import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from common import data, metrics
from convgru_unet import GRU_Unet
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

        loss += 0.1 * (1 - torch.mean(torch.abs(torch.tanh(pred))))  # high contrast (avoid fading)

        return loss


class RnnGridsampler(nn.Module):
    def __init__(self, seq_len, num_layers=1, dim_in_vec = 2, dim_in_img = 2):
        super(RnnGridsampler, self).__init__()

        self.len = seq_len

        self.rnn = nn.GRU(input_size=64,
                          hidden_size=64,
                          num_layers=num_layers,
                          dropout=0.5,
                          bias=True,
                          bidirectional=True)  #TODO: other outputs!

        self.img2vec = nn.Sequential(
            nn.InstanceNorm3d(dim_in_img),  # 128, 28
            nn.Conv3d(dim_in_img, 10, kernel_size=3, padding=1),  # 128, 28
            nn.ReLU(),
            nn.MaxPool3d(2,2),  # 64, 14
            nn.InstanceNorm3d(10),
            nn.Conv3d(10, 20, kernel_size=3),  # 62, 12
            nn.ReLU(),
            nn.MaxPool3d(2,2),  # 31, 6
            nn.InstanceNorm3d(20),
            nn.Conv3d(20, 30, kernel_size=3),  # 29, 4
            nn.ReLU(),
            nn.InstanceNorm3d(30),
            nn.Conv3d(30, 40, kernel_size=3),  # 27, 2
            nn.ReLU(),
            nn.MaxPool3d((1, 3, 3), (1, 3, 3)),  # 9, 2
            nn.InstanceNorm3d(40),
            nn.Conv3d(40, 50, kernel_size=(1, 3, 3)),  # 7, 2
            nn.ReLU(),
            nn.InstanceNorm3d(50),
            nn.Conv3d(50, 60, kernel_size=(1, 3, 3)),  # 5, 2
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(output_size=(1,1,1))  # 1, 1
        )
        self.vec2vec = nn.Sequential(
            nn.InstanceNorm3d(dim_in_vec),
            nn.Conv3d(dim_in_vec, 4, kernel_size=1),
            nn.ReLU(),
            nn.InstanceNorm3d(4),
            nn.Conv3d(4, 4, kernel_size=1),
            nn.ReLU(),
        )

        self.vec2theta = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 4*3)  # 4*3 for 3D, 3*2 for 2D
        )
        self.vec2theta[2].weight.data.zero_()
        self.vec2theta[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, input_img, input_vec):
        vec_img = self.img2vec(input_img)
        vec_vec = self.vec2vec(input_vec)
        vec_cat = torch.cat((vec_img, vec_vec), dim=1)

        output = []
        hidden = None
        for i in range(self.len):
            result, hidden = self.rnn(vec_cat.view(-1, 64), hidden)  # TODO: correct return values?
            theta = self.vec2theta(result[-1]).view(-1, 3, 4)
            grid = nn.functional.affine_grid(theta, input_img[:, 0].size())
            output.append(nn.functional.grid_sample(input_img[:, 0], grid))

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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sequence_length = 9
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
                                                          [0, 1, 2, 3],  #4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                                                          [4, 5, 6, 7],  #[16, 17, 18, 19],
                                                          batchsize=batchsize, normalize=sequence_length, growth='fast',
                                                          zsize=zsize)

rnn = RnnGridsampler(seq_len=sequence_length)

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
    f, axarr = plt.subplots(n_visual_samples * 2, sequence_length * 2)
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

            input_image = torch.ones(batchsize, 4, zsize, 128, 128)
            input_image[:, :2, :, :, :] = gt[mask].view(batchsize, 2, zsize, 128, 128)
            input_image[:, 2:, :, :, :] = input_image[:, 2:, :, :, :] * batch[data.KEY_GLOBAL]
            input_image = input_image.to(device).requires_grad_()
            for b in range(batchsize):
                mask[b, int(batch[data.KEY_GLOBAL][b, 1, :, :, :]), :, :, :] = 1  # lesion

            output = rnn(input_image, batch[data.KEY_GLOBAL])

            loss = criterion(output, gt[:, 1].unsqueeze(1))  # loss on follow-up lesion only
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
    del input_image
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

            input_image = torch.ones(batchsize, 4, zsize, 128, 128)
            input_image[:, :2, :, :, :] = gt[mask].view(batchsize, 2, zsize, 128, 128)
            input_image[:, 2:, :, :, :] = input_image[:, 2:, :, :, :] * batch[data.KEY_GLOBAL]
            input_image = input_image.to(device).requires_grad_()
            for b in range(batchsize):
                mask[b, int(batch[data.KEY_GLOBAL][b, 1, :, :, :]), :, :, :] = 1  # lesion

            output = rnn(input_image, batch[data.KEY_GLOBAL])

            loss = criterion(output, gt[:, 1].unsqueeze(1))  # loss on follow-up lesion only
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
