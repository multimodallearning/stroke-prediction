import torch
import torch.nn as nn
from common import data, metrics
import matplotlib.pyplot as plt


class Criterion(nn.Module):
    def __init__(self, weights):
        super(Criterion, self).__init__()
        self.dc = metrics.BatchDiceLoss(weights)  # weighted inversely by each volume proportion

    def forward(self, pred, target, pred_seq):
        loss = self.dc(pred, target)

        for i in range(pred_seq.size()[1]-1):
            diff = pred_seq[:, i+1] - pred_seq[:, i]
            loss += torch.mean(torch.abs(diff) - diff)  # monotone

        loss += 0.1 * (1 - torch.mean(torch.abs(torch.tanh(pred_seq))))  # high contrast (avoid fading)

        return loss


class PaddedUnet(nn.Module):
    def _block_def(self, ch_in, ch_out, input2d):
        kernel = 3
        padding = 1
        if input2d:
            kernel = (1, 3, 3)
            padding = (0, 1, 1)
        return nn.Sequential(
            nn.InstanceNorm3d(ch_in),
            nn.Conv3d(ch_in, ch_out, kernel, stride=1, padding=padding),
            nn.ReLU(),
            nn.InstanceNorm3d(ch_out),
            nn.Conv3d(ch_out, ch_out, kernel, stride=1, padding=padding),
            nn.ReLU()
        )

    def __init__(self, channels=[2, 16, 32, 16, 2], channel_dim=1, input2d=False):
        super(PaddedUnet, self).__init__()
        kernel = 2
        if input2d:
            kernel = (1, 2, 2)

        n_ch_in, ch_b1, ch_b2, ch_b3, n_classes = channels

        self.channel_dim = channel_dim

        self.block1 = self._block_def(n_ch_in, ch_b1, input2d)
        self.pool12 = nn.MaxPool3d(kernel, kernel)
        self.block2 = self._block_def(ch_b1, ch_b2, input2d)
        self.upsample = nn.Upsample(scale_factor=kernel, mode='trilinear')
        self.block3 = self._block_def(ch_b2 + ch_b1, ch_b3, input2d)

        self.classify = nn.Sequential(
            nn.Conv3d(ch_b3, n_classes, 1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, input):
        block1_result = self.block1(input)
        block2_input = self.pool12(block1_result)
        block2_result = self.block2(block2_input)

        block2_unpool = self.upsample(block2_result)
        block3_input = torch.cat((block2_unpool, block1_result), dim=self.channel_dim)
        block3_result = self.block3(block3_input)

        return self.classify(block3_result)


class RnnGridsampler(nn.Module):
    def _theta(self, num_directions):
        return nn.Sequential(
            # num affine parameters: 4*3 (12) for 3D, 3*2 (6) for 2D
            nn.Linear(dim_hidden * num_directions, 6 + (dim_hidden * num_directions) // 2),
            nn.ReLU(True),
            nn.Linear(6 + (dim_hidden * num_directions) // 2, 6 + (dim_hidden * num_directions) // 2),
            nn.ReLU(True),
            nn.Linear(6 + (dim_hidden * num_directions) // 2, 12)
        )

    def __init__(self, seq_len, num_layers=1, dim_hidden = 64, dim_in_img = 2, dim_in_vec = 2, zsize=28):
        super(RnnGridsampler, self).__init__()

        ksize = 3
        psize = 1
        dsize = 2
        if zsize == 1:
            ksize = (1,3,3)
            psize = (0,1,1)
            dsize = (1,2,2)

        num_directions = 2
        self.len = seq_len
        self.num_layers = num_layers

        self.rnn = nn.GRU(input_size=dim_hidden,
                          hidden_size=dim_hidden,
                          num_layers=num_layers,
                          dropout=0.3333,
                          bias=True,
                          bidirectional=(num_directions == 2))  #TODO: other outputs!

        dim_hidden_step = dim_hidden // 6

        self.img2vec = nn.Sequential(
            nn.InstanceNorm3d(dim_in_img),  # 128, 28
            nn.Conv3d(dim_in_img, dim_hidden_step, kernel_size=ksize, padding=psize),  # 128, 28
            nn.ReLU(),
            nn.MaxPool3d(dsize,dsize),  # 64, 14
            nn.InstanceNorm3d(dim_hidden_step),
            nn.Conv3d(dim_hidden_step, 2 * dim_hidden_step, kernel_size=ksize),  # 62, 12
            nn.ReLU(),
            nn.MaxPool3d(dsize,dsize),  # 31, 6
            nn.InstanceNorm3d(2 * dim_hidden_step),
            nn.Conv3d(2 * dim_hidden_step, 3 * dim_hidden_step, kernel_size=ksize),  # 29, 4
            nn.ReLU(),
            nn.InstanceNorm3d(3 * dim_hidden_step),
            nn.Conv3d(3 * dim_hidden_step, 4 * dim_hidden_step, kernel_size=ksize),  # 27, 2
            nn.ReLU(),
            nn.MaxPool3d((1, 3, 3), (1, 3, 3)),  # 9, 2
            nn.InstanceNorm3d(4 * dim_hidden_step),
            nn.Conv3d(4 * dim_hidden_step, 5 * dim_hidden_step, kernel_size=(1, 3, 3)),  # 7, 2
            nn.ReLU(),
            nn.InstanceNorm3d(5 * dim_hidden_step),
            nn.Conv3d(5 * dim_hidden_step, dim_hidden - dim_in_vec, kernel_size=(1, 3, 3)),  # 5, 2
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(output_size=(1,1,1))  # 1, 1
        )

        self.vec2theta_0 = self._theta(num_directions)
        self.vec2theta_0[4].weight.data.zero_()
        self.vec2theta_0[4].bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float))

        self.vec2theta_1 = self._theta(num_directions)
        self.vec2theta_1[4].weight.data.zero_()
        self.vec2theta_1[4].bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float))

        self.weighting = nn.Sequential(
            nn.Linear(num_directions * dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, input_img, input_vec):
        vec_img = self.img2vec(input_img)
        vec_cat = torch.cat((vec_img, input_vec), dim=1)
        vec_seq = torch.stack([vec_cat]*self.len)

        output, _ = self.rnn(vec_seq.squeeze())

        input_0 = input_img[:, 0].unsqueeze(1)
        input_1 = input_img[:, 1].unsqueeze(1)

        result = []
        for i in range(self.len):
            theta_0 = self.vec2theta_0(output[i]).view(-1, 3, 4)
            theta_1 = self.vec2theta_1(output[i]).view(-1, 3, 4)
            grid_0 = nn.functional.affine_grid(theta_0, input_0.size())
            grid_1 = nn.functional.affine_grid(theta_1, input_1.size())
            out_0 = nn.functional.grid_sample(input_0, grid_0)
            out_1 = nn.functional.grid_sample(input_1, grid_1)
            weights = self.weighting(output[i]).view(-1, 1, 1, 1, 1)
            result.append(out_0 * weights + out_1 * (1 - weights))

        return torch.cat(result, dim=1)


def get_title(prefix, idx, batch, row):
    suffix = ''
    if idx == int(batch[data.KEY_GLOBAL][row, 0, :, :, :]):
        suffix += ' core'
    elif idx == int(batch[data.KEY_GLOBAL][row, 1, :, :, :]):
        suffix += ' fuct'
    elif idx == 23:
        suffix += ' penu'
    return prefix + '[' + str(idx) + ']' + suffix


def visualize(axarr, gt, output, n_visual_samples, offset=0):
    for row in range(n_visual_samples):
        titles = []
        for i in range(sequence_length//2):
            axarr[row + offset, i].imshow(gt.cpu().detach().numpy()[row, i*2, zslice, :, :], vmin=0, vmax=1, cmap='gray')
            titles.append(get_title('GT', i*2, batch, row))
        for i in range(sequence_length//2):
            axarr[row + offset, i + sequence_length//2].imshow(output.cpu().detach().numpy()[row, i*2, zslice, :, :], vmin=0, vmax=1, cmap='gray')
            titles.append(get_title('Pr', i*2, batch, row))
        for ax, title in zip(axarr[row + offset], titles):
            ax.set_title(title)
    return axarr

modalities = ['_CBV_reg1_downsampled',
              '_TTD_reg1_downsampled']
labels = ['_CBVmap_subset_reg1_downsampled',
          '_FUCT_MAP_T_Samplespace_subset_reg1_downsampled',
          '_TTDmap_subset_reg1_downsampled']

zsize = 1  # change here for 2D/3D: 1 or 28
input2d = (zsize == 1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sequence_length = 24
num_layers = 5
dim_hidden = 64
batchsize = 4
zslice = zsize // 2
pad = (20, 20, 20)
n_visual_samples = min(4, batchsize)

train_trafo = [data.UseLabelsAsImages(),
               data.HemisphericFlip(),
               data.ElasticDeform2D(apply_to_images=True, random=0.95),
               data.ToTensor()]
valid_trafo = [data.UseLabelsAsImages(),
               data.ElasticDeform2D(apply_to_images=True, random=0.67, seed=0),
               data.ToTensor()]

ds_train, ds_valid = data.get_toy_seq_shape_training_data(train_trafo, valid_trafo,
                                                          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                                                          [16, 17, 18, 19],
                                                          batchsize=batchsize, normalize=sequence_length, growth='fast',
                                                          zsize=zsize)

rnn = RnnGridsampler(seq_len=sequence_length, num_layers=num_layers, dim_hidden=dim_hidden,
                     dim_in_img = 2, dim_in_vec = 2, zsize=zsize).to(device)

params = [p for p in rnn.parameters() if p.requires_grad]
print('# optimizing params', sum([p.nelement() * p.requires_grad for p in params]),
      '/ total: RNN', sum([p.nelement() for p in rnn.parameters()]))

criterion = Criterion([1.0])
optimizer = torch.optim.Adam(params, lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.1)

loss_train = []
loss_valid = []

for epoch in range(0, 175):
    scheduler.step()
    f, axarr = plt.subplots(n_visual_samples * 2, sequence_length)
    loss_mean = 0
    inc = 0

    ### Train ###

    is_train = True
    rnn.train(is_train)
    with torch.set_grad_enabled(is_train):

        for batch in ds_train:
            gt = batch[data.KEY_LABELS].to(device)

            mask = torch.zeros(gt.size()).byte()
            for b in range(batchsize):
                mask[b, int(batch[data.KEY_GLOBAL][b, 0, :, :, :]), :, :, :] = 1  # core
                mask[b, -1, :, :, :] = 1  # penumbra

            input_img = torch.ones(batchsize, 2, zsize, 128, 128)
            input_img[:, :, :, :, :] = gt[mask].view(batchsize, 2, zsize, 128, 128)
            input_img = input_img.to(device).requires_grad_()

            mask = torch.zeros(gt.size()).byte()
            for b in range(batchsize):
                mask[b, int(batch[data.KEY_GLOBAL][b, 1, :, :, :]), :, :, :] = 1  # lesion

            input_vec = batch[data.KEY_GLOBAL]
            input_vec = input_vec.to(device).requires_grad_()

            output = rnn(input_img, input_vec)

            loss = criterion(output[mask].view(batchsize, 1, zsize, 128, 128),
                             gt[mask].view(batchsize, 1, zsize, 128, 128),
                             output)
            loss_mean += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            inc += 1

        loss_train.append(loss_mean/inc)

        axarr = visualize(axarr, gt, output, n_visual_samples)
        del batch

    del output
    del input_img
    del loss
    del gt

    ### Validate ###

    inc = 0
    loss_mean = 0
    is_train = False
    optimizer.zero_grad()
    rnn.train(is_train)
    with torch.set_grad_enabled(is_train):

        for batch in ds_valid:
            gt = batch[data.KEY_LABELS].to(device)

            mask = torch.zeros(gt.size()).byte()
            for b in range(batchsize):
                mask[b, int(batch[data.KEY_GLOBAL][b, 0, :, :, :]), :, :, :] = 1  # core
                mask[b, -1, :, :, :] = 1  # penumbra

            input_img = torch.ones(batchsize, 2, zsize, 128, 128)
            input_img[:, :, :, :, :] = gt[mask].view(batchsize, 2, zsize, 128, 128)
            input_img = input_img.to(device).requires_grad_()

            mask = torch.zeros(gt.size()).byte()
            for b in range(batchsize):
                mask[b, int(batch[data.KEY_GLOBAL][b, 1, :, :, :]), :, :, :] = 1  # lesion

            input_vec = batch[data.KEY_GLOBAL]
            input_vec = input_vec.to(device).requires_grad_()

            output = rnn(input_img, input_vec)

            loss = criterion(output[mask].view(batchsize, 1, zsize, 128, 128),
                             gt[mask].view(batchsize, 1, zsize, 128, 128),
                             output)
            loss_mean += loss.item()

            inc += 1

        loss_valid.append(loss_mean/inc)

        axarr = visualize(axarr, gt, output, n_visual_samples, offset=n_visual_samples)
        del batch

    print('Epoch', epoch, 'last batch training loss:', loss_train[-1], '\tvalidation batch loss:', loss_valid[-1])

    if epoch % 5 == 0:
        torch.save(rnn, '/share/data_zoe1/lucas/NOT_IN_BACKUP/tmp/gru_gridsampler.model')

    for ax in axarr.flatten():
        ax.title.set_fontsize(3)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    f.subplots_adjust(hspace=0.05)
    f.savefig('/share/data_zoe1/lucas/NOT_IN_BACKUP/tmp/gru_gridsampler' + str(epoch) + '.png', bbox_inches='tight', dpi=300)

    del f
    del axarr

    if epoch > 0:
        fig, plot = plt.subplots()
        epochs = range(1, epoch + 2)
        plot.plot(epochs, loss_train, 'r-')
        plot.plot(epochs, loss_valid, 'b-')
        plot.set_ylabel('Loss Training (r) & Validation (b)')
        fig.savefig('/share/data_zoe1/lucas/NOT_IN_BACKUP/tmp/gru_gridsampler_plots.png', bbox_inches='tight', dpi=300)
        del plot
        del fig
