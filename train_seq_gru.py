import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from common import data, metrics
from convgru_unet import ConvGRU_Unet
import matplotlib.pyplot as plt


class Criterion(nn.Module):
    def __init__(self, weights):
        super(Criterion, self).__init__()
        self.dc = metrics.BatchDiceLoss(weights)  # weighted inversely by each volume proportion

    def forward(self, pred, target):
        loss = self.dc(pred, target)

        for i in range(pred.size()[1]-1):
            diff = pred[:, i+1] - pred[:, i]
            loss += torch.mean(torch.abs(diff) - diff)  # monotone

        loss += 0.01 * (1 - torch.mean(torch.abs(torch.tanh(pred))))  # high contrast (avoid fading)

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
        self.block3 = self._block_def(ch_b2 + ch_b1, ch_b3, input2d)

        self.upsample = nn.Upsample(scale_factor=kernel, mode='trilinear')

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


class RnnCheckpointingWrapper(nn.Module):
    def __init__(self, rnn, seq_len):
        super(RnnCheckpointingWrapper, self).__init__()
        self.rnn = rnn
        self.len = seq_len
        assert seq_len > 0

    def cp_func(self, module):
        def custom_forward(*inputs):
            assert 0 < len(inputs) < 3
            if len(inputs) == 2:
                inputs = module(inputs[0], inputs[1])
            else:
                inputs = module(inputs[0])
            return inputs
        return custom_forward

    def forward(self, input):
        hidden = checkpoint(self.cp_func(self.rnn), input)
        output = [hidden[:, -1, :, :, :].unsqueeze(1)]  # only works for 1 channels output of last layer in GRU
        if self.len > 1:
            for i in range(1, self.len):
                hidden = checkpoint(self.cp_func(self.rnn), input, hidden)
                output.append(hidden[:, -1, :, :, :].unsqueeze(1))
        output = torch.cat(output, dim=1)
        return output


def get_title(prefix, idx, batch):
    suffix = ''
    if idx == int(batch[data.KEY_GLOBAL][row, 0, :, :, :]):
        suffix += ' core'
    elif idx == int(batch[data.KEY_GLOBAL][row, 1, :, :, :]):
        suffix += ' fuct'
    elif idx == 8:
        suffix += ' penu'
    return prefix + '[' + str(idx) + ']' + suffix


modalities = ['_CBV_reg1_downsampled',
              '_TTD_reg1_downsampled']
labels = ['_CBVmap_subset_reg1_downsampled',
          '_FUCT_MAP_T_Samplespace_subset_reg1_downsampled',
          '_TTDmap_subset_reg1_downsampled']

zsize = 1  # change here for 2D/3D: 1 or 28
input2d = (zsize == 1)
convgru_kernel = 3
if input2d:
    convgru_kernel = (1, 3, 3)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sequence_length = 9
num_layers = 3
num_input = 4
batchsize = 4
zslice = zsize // 2
pad = (20, 20, 20)
n_visual_samples = min(4, batchsize)

train_trafo = [data.UseLabelsAsImages(),
               data.PadImages(pad[0], pad[1], pad[2], pad_value=0),
               data.HemisphericFlip(),
               data.ElasticDeform(apply_to_images=True, random=0.95),
               data.ToTensor()]
valid_trafo = [data.UseLabelsAsImages(),
               data.PadImages(pad[0], pad[1], pad[2], pad_value=0),
               data.ElasticDeform(apply_to_images=True, random=0.67, seed=0),
               data.ToTensor()]

ds_train, ds_valid = data.get_toy_seq_shape_training_data(train_trafo, valid_trafo,
                                                          [0, 1, 2, 3],  #4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                                                          [4, 5, 6, 7],  #[16, 17, 18, 19],
                                                          batchsize=batchsize, normalize=sequence_length, growth='fast',
                                                          zsize=zsize)

channels = 16
unet_out = 16
shared_unet = PaddedUnet([num_input, channels, 32, channels, unet_out], input2d=input2d)
convgru = ConvGRU_Unet(input_size=unet_out,
                       hidden_sizes=[channels]*(num_layers-1) + [1],
                       kernel_sizes=[convgru_kernel]*(num_layers-1) + [1],
                       n_layers=num_layers,
                       shared_unet=shared_unet).to(device)
rnn = RnnCheckpointingWrapper(convgru, sequence_length)

params = [p for p in convgru.parameters() if p.requires_grad]
print('# optimizing params', sum([p.nelement() * p.requires_grad for p in params]),
      '/ total: Unet-GRU-RNN', sum([p.nelement() for p in convgru.parameters()]))

criterion = Criterion([0.1, 0.8, 0.1])
optimizer = torch.optim.Adam(params, lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.1)

loss_train = []
loss_valid = []

for epoch in range(0, 175):
    scheduler.step()
    f, axarr = plt.subplots(n_visual_samples * 2, sequence_length * 2)
    loss_mean = 0
    inc = 0

    ### Train ###

    is_train = True
    convgru.train(is_train)
    with torch.set_grad_enabled(is_train):

        for batch in ds_train:
            gt = batch[data.KEY_LABELS].to(device)

            mask = torch.zeros(gt.size()).byte()
            for b in range(batchsize):
                mask[b, int(batch[data.KEY_GLOBAL][b, 0, :, :, :]), :, :, :] = 1  # core
                mask[b, -1, :, :, :] = 1  # penumbra

            input = torch.ones(batchsize, 4, zsize, 128 ,128)
            input[:, :2, :, :, :] = gt[mask].view(batchsize, 2, zsize, 128, 128)
            input[:, 2:, :, :, :] = input[:, 2:, :, :, :] * batch[data.KEY_GLOBAL]
            input = input.to(device).requires_grad_()
            for b in range(batchsize):
                mask[b, int(batch[data.KEY_GLOBAL][b, 1, :, :, :]), :, :, :] = 1  # lesion

            output = rnn(input)

            loss = criterion(output[mask].view(batchsize, 3, zsize, 128, 128),
                             gt[mask].view(batchsize, 3, zsize, 128, 128))
            loss_mean += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            inc += 1

        loss_train.append(loss_mean/inc)

        for row in range(n_visual_samples):
            titles = []
            for i in range(sequence_length):
                axarr[row, i].imshow(gt.cpu().detach().numpy()[row, i, zslice, :, :], vmin=0, vmax=1, cmap='gray')
                titles.append(get_title('GT', i, batch))
            for i in range(sequence_length):
                axarr[row, i + sequence_length].imshow(output.cpu().detach().numpy()[row, i, zslice, :, :], vmin=0, vmax=1, cmap='gray')
                titles.append(get_title('Pr', i, batch))
            for ax, title in zip(axarr[row], titles):
                ax.set_title(title)
        del batch

    del output
    del input
    del loss
    del gt

    ### Validate ###

    inc = 0
    loss_mean = 0
    is_train = False
    optimizer.zero_grad()
    convgru.train(is_train)
    with torch.set_grad_enabled(is_train):

        for batch in ds_valid:
            gt = batch[data.KEY_LABELS].to(device)

            mask = torch.zeros(gt.size()).byte()
            for b in range(batchsize):
                mask[b, int(batch[data.KEY_GLOBAL][b, 0, :, :, :]), :, :, :] = 1  # core
                mask[b, -1, :, :, :] = 1  # penumbra

            input = torch.ones(batchsize, 4, zsize, 128 ,128)
            input[:, :2, :, :, :] = gt[mask].view(batchsize, 2, zsize, 128, 128)
            input[:, 2:, :, :, :] = input[:, 2:, :, :, :] * batch[data.KEY_GLOBAL]
            input = input.to(device).requires_grad_()
            for b in range(batchsize):
                mask[b, int(batch[data.KEY_GLOBAL][b, 1, :, :, :]), :, :, :] = 1  # lesion

            output = rnn(input)

            loss = criterion(output[mask].view(batchsize, 3, zsize, 128, 128),
                             gt[mask].view(batchsize, 3, zsize, 128, 128))
            loss_mean += loss.item()

            inc += 1

        loss_valid.append(loss_mean/inc)

        for row in range(n_visual_samples):
            titles = []
            for i in range(sequence_length):
                axarr[row + n_visual_samples, i].imshow(gt.cpu().detach().numpy()[row, i, zslice, :, :], vmin=0, vmax=1, cmap='gray')
                titles.append(get_title('GT', i, batch))
            for i in range(sequence_length):
                axarr[row + n_visual_samples, i + sequence_length].imshow(output.cpu().detach().numpy()[row, i, zslice, :, :], vmin=0, vmax=1, cmap='gray')
                titles.append(get_title('Pr', i, batch))
            for ax, title in zip(axarr[row + n_visual_samples], titles):
                ax.set_title(title)
        del batch

    print('Epoch', epoch, 'last batch training loss:', loss_train[-1], '\tvalidation batch loss:', loss_valid[-1])

    if epoch % 5 == 0:
        torch.save(convgru, '/share/data_zoe1/lucas/NOT_IN_BACKUP/tmp/convgru.model')

    for ax in axarr.flatten():
        ax.title.set_fontsize(3)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    f.subplots_adjust(hspace=0.05)
    f.savefig('/share/data_zoe1/lucas/NOT_IN_BACKUP/tmp/convgru' + str(epoch) + '.png', bbox_inches='tight', dpi=300)

    del f
    del axarr

    if epoch > 0:
        fig, plot = plt.subplots()
        epochs = range(1, epoch + 2)
        plot.plot(epochs, loss_train, 'r-')
        plot.plot(epochs, loss_valid, 'b-')
        plot.set_ylabel('Loss Training (r) & Validation (b)')
        fig.savefig('/share/data_zoe1/lucas/NOT_IN_BACKUP/tmp/convgru_plots.png', bbox_inches='tight', dpi=300)
        del plot
        del fig
