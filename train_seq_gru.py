import torch
import torch.nn as nn
from torch.autograd import Variable
from common import data, metrics
from convgru_unet import ConvGRU_Unet
import matplotlib.pyplot as plt


class Criterion(nn.Module):
    def __init__(self, weights, seq_len=10):
        super(Criterion, self).__init__()
        self.dc = metrics.BatchDiceLoss(weights)  # weighted inversely by each volume proportion
        self.seq_len = seq_len

    def forward(self, pred, target):
        loss = self.dc(pred, target)

        for i in range(self.seq_len-1):
            diff = pred[:, i+1] - pred[:, i]
            loss += torch.mean(torch.abs(diff) - diff)  # monotone

        return loss


class PaddedUnet(nn.Module):
    def _block_def(self, ch_in, ch_out):
        return nn.Sequential(
            nn.BatchNorm3d(ch_in),
            nn.Conv3d(ch_in, ch_out, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(ch_out),
            nn.Conv3d(ch_out, ch_out, 3, stride=1, padding=1),
            nn.ReLU()
        )

    def __init__(self, channels=[2, 16, 32, 16, 2], channel_dim=1):
        super(PaddedUnet, self).__init__()
        n_ch_in, ch_b1, ch_b2, ch_b3, n_classes = channels

        self.channel_dim = channel_dim

        self.block1 = self._block_def(n_ch_in, ch_b1)
        self.pool12 = nn.MaxPool3d(2, 2)
        self.block2 = self._block_def(ch_b1, ch_b2)

        self.upsa23 = nn.Upsample(scale_factor=2, mode='trilinear')
        self.block3 = self._block_def(ch_b2 + ch_b1, ch_b3)

        self.classify = nn.Sequential(
            nn.Conv3d(ch_b3, n_classes, 1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, input):
        block1_result = self.block1(input)
        block2_input = self.pool12(block1_result)
        block2_result = self.block2(block2_input)

        block2_unpool = self.upsa23(block2_result)
        block3_input = torch.cat((block2_unpool, block1_result), dim=self.channel_dim)
        block3_result = self.block3(block3_input)

        return self.classify(block3_result)


modalities = ['_CBV_reg1_downsampled',
              '_TTD_reg1_downsampled']
labels = ['_CBVmap_subset_reg1_downsampled',
          '_FUCT_MAP_T_Samplespace_subset_reg1_downsampled',
          '_TTDmap_subset_reg1_downsampled']

sequence_length = 4
num_layers = 3
batchsize = 2
zslice = 14
pad = (20, 20, 20)
n_visual_samples = min(4, batchsize)

train_trafo = [data.UseLabelsAsImages(),
               data.PadImages(pad[0], pad[1], pad[2], pad_value=0),
               data.HemisphericFlip(),
               data.ElasticDeform(apply_to_images=True, random=0.95),
               data.ToTensor()]
valid_trafo = [data.UseLabelsAsImages(),
               data.PadImages(pad[0], pad[1], pad[2], pad_value=0),
               data.ToTensor()]

ds_train, _ = data.get_toy_seq_shape_training_data(train_trafo, valid_trafo,
                                                   [0, 1, 2, 3, 4, 5, 6, 7],  #, 8, 9, 10, 11, 12, 13, 14, 15],
                                                   [],  #16, 17, 18, 19],
                                                   batchsize=batchsize, normalize=sequence_length)

ch_unet_out = 16
shared_unet = PaddedUnet([2, 16, 32, 16, ch_unet_out])
convgru = ConvGRU_Unet(input_size=ch_unet_out,
                       hidden_sizes=[16]*(num_layers-1) + [1],
                       kernel_sizes=[3]*(num_layers-1) + [1],
                       n_layers=num_layers,
                       shared_unet=shared_unet).cuda()

params = [p for p in convgru.parameters() if p.requires_grad]
print('# optimizing params', sum([p.nelement() * p.requires_grad for p in params]),
      '/ total: Unet-GRU-RNN', sum([p.nelement() for p in convgru.parameters()]))

criterion = Criterion([0.25, 0.25, 0.25, 0.25], seq_len=sequence_length)
optimizer = torch.optim.Adam(params, lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.1)

loss_train = []
loss_valid = []

for epoch in range(0, 175):
    scheduler.step()
    f, axarr = plt.subplots(n_visual_samples, 8)
    loss_mean = 0
    inc = 0
    train = True
    convgru.train(train)

    for batch in ds_train:
        gt = Variable(batch[data.KEY_LABELS], volatile=not train).cuda()

        del batch

        hidden = None
        output = []
        for i in range(sequence_length):
            input = torch.cat((gt[:, 0, :, :, :].unsqueeze(1), gt[:, -1, :, :, :].unsqueeze(1)), dim=1)
            hidden = convgru(input, hidden=hidden)
            output.append(hidden[-1])
        output = torch.cat(output, dim=1)

        loss = criterion(output, gt)
        loss_mean += float(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        inc += 1

    loss_train.append(loss_mean/inc)

    for row in range(n_visual_samples):
        axarr[row, 0].imshow(gt.cpu().data.numpy()[row, 0, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        axarr[row, 1].imshow(gt.cpu().data.numpy()[row, 1, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        axarr[row, 2].imshow(gt.cpu().data.numpy()[row, 2, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        axarr[row, 3].imshow(gt.cpu().data.numpy()[row, 3, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        axarr[row, 4].imshow(output.cpu().data.numpy()[row, 0, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        axarr[row, 5].imshow(output.cpu().data.numpy()[row, 1, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        axarr[row, 6].imshow(output.cpu().data.numpy()[row, 2, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        axarr[row, 7].imshow(output.cpu().data.numpy()[row, 3, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        titles = ['GT[0]', 'GT[1]', 'GT[2]', 'GT[3]', 'Pr[0]', 'Pr[1]', 'Pr[2]', 'Pr[3]']
        for ax, title in zip(axarr[row], titles):
            ax.set_title(title)

    del output
    del hidden
    del loss
    del gt

    optimizer.zero_grad()

    train = False
    convgru.train(train)

    loss_valid.append(0.0)

    print('Epoch', epoch, 'last batch training loss:', loss_train[-1], '\tvalidation batch loss:', loss_valid[-1])

    if epoch % 5 == 0:
        torch.save(convgru, '/share/data_zoe1/lucas/NOT_IN_BACKUP/tmp/convgru.model')
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
