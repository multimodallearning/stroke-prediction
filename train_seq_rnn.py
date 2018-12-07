import torch
import torch.nn as nn
from torch.autograd import Variable
from common import data, metrics
import common.model.Unet3D as UnetHelper
import matplotlib.pyplot as plt


class Criterion(nn.Module):
    def __init__(self, weights, normalize, alpha=[196. / 888., 192. / 888., 59. / 888., 49. / 888.], seq_len=10):
        super(Criterion, self).__init__()
        self.dc = metrics.BatchDiceLoss(weights)  # weighted inversely by each volume proportion
        self.normalize = normalize
        self.alpha = alpha
        self.seq_len = seq_len

    def forward(self, pred, target):
        batchsize = target.size()[0]

        loss = 0

        for b in range(batchsize):
            loss += self.alpha[0] * self.dc(pred[b, 0], target[b, 0])#int(round(float(time_core[b])))], target[b, 0])
            loss += self.alpha[1] * self.dc(pred[b, 1], target[b, 1])#int(round(float(time_lesion[b])))], target[b, 1])
            loss += self.alpha[2] * self.dc(pred[b, 2], target[b, 2])

            for i in range(self.seq_len-1):
                diff = pred[b, i+1] - pred[b, i]
                loss += self.alpha[3] * torch.mean(torch.abs(diff) - diff)  # monotone

        return loss/batchsize


class Unet(nn.Module):
    def __init__(self, channels=[2, 32, 64, 128, 64, 32, 2], channel_dim=1, channels_crop=[2,3,4]):
        super(Unet, self).__init__()
        n_ch_in, ch_b1, ch_b2, ch_b3, ch_b4, ch_b5, n_classes = channels

        self.channel_dim = channel_dim
        self.channels_crop = channels_crop

        self.block1 = UnetHelper.Block3x3x3(n_ch_in, ch_b1)
        self.pool12 = nn.MaxPool3d(2, 2)
        self.block2 = UnetHelper.Block3x3x3(ch_b1, ch_b2)
        self.pool23 = nn.MaxPool3d(2, 2)
        self.block3 = UnetHelper.Block3x3x3(ch_b2, ch_b3)

        self.upsa34 = nn.Upsample(scale_factor=2, mode='trilinear')
        self.block4 = UnetHelper.Block3x3x3(ch_b3 + ch_b2, ch_b4)
        self.upsa45 = nn.Upsample(scale_factor=2, mode='trilinear')
        self.block5 = UnetHelper.Block3x3x3(ch_b4 + ch_b1, ch_b5)

        self.classify = nn.Sequential(
            nn.Conv3d(ch_b5, n_classes, 1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, input):
        block1_result = self.block1(input)

        block2_input = self.pool12(block1_result)
        block2_result = self.block2(block2_input)

        block3_input = self.pool23(block2_result)
        block3_result = self.block3(block3_input)
        block3_unpool = self.upsa34(block3_result)

        block2_crop = UnetHelper.crop(block2_result, block3_unpool, dims=self.channels_crop)
        block4_input = torch.cat((block3_unpool, block2_crop), dim=self.channel_dim)
        block4_result = self.block4(block4_input)
        block4_unpool = self.upsa45(block4_result)

        block1_crop = UnetHelper.crop(block1_result, block4_unpool, dims=self.channels_crop)
        block5_input = torch.cat((block4_unpool, block1_crop), dim=self.channel_dim)
        block5_result = self.block5(block5_input)

        return self.classify(block5_result)


class RnnCell(nn.Module):
    def __init__(self, channels=16):
        super(RnnCell, self).__init__()

        self.hh = nn.Sequential(
            nn.InstanceNorm3d(channels),
            nn.Conv3d(channels, channels, 1),
            nn.ReLU(True),
            nn.InstanceNorm3d(channels),
            nn.Conv3d(channels, channels, 1),
            nn.Tanh()
        )

        self.ho = nn.Sequential(
            nn.InstanceNorm3d(channels),
            nn.Conv3d(channels, channels//2, 3, padding=1),
            nn.ReLU(True),
            nn.InstanceNorm3d(channels//2),
            nn.Conv3d(channels//2, 1, 3, padding=1),
            nn.ReLU(True)
        )

    def forward(self, h_state):
        return self.hh(h_state), self.ho(h_state)


class RnnUnet(nn.Module):
    def __init__(self, unet, cell, seq_len):
        super(RnnUnet, self).__init__()
        self.unet = unet
        self.cell = cell
        self.seq_len = seq_len

    def forward(self, input):
        h_states = []
        predicts = []
        h_state = self.unet(input)
        for i in range(self.seq_len):
            h_states.append(h_state)
            h_state, predict = self.cell(h_state)
            predicts.append(predict)
        return predicts, h_states

    def freeze(self, freeze=False):
        requires_grad = not freeze
        for param in self.parameters():
            param.requires_grad = requires_grad


modalities = ['_CBV_reg1_downsampled',
              '_TTD_reg1_downsampled']
labels = ['_CBVmap_subset_reg1_downsampled',
          '_FUCT_MAP_T_Samplespace_subset_reg1_downsampled',
          '_TTDmap_subset_reg1_downsampled']

sequence_length = 3
channels_rnn = 12
channels_unet = [5, 16, 24, 48, 24, 16, channels_rnn]
batchsize = 4
normalize = 10
zslice = 14
pad = (20, 20, 20)
n_visual_samples = 3

train_trafo = [data.UseLabelsAsImages(),
               data.PadImages(pad[0], pad[1], pad[2], pad_value=0),
               data.HemisphericFlip(),
               data.ElasticDeform(apply_to_images=True),
               data.ToTensor()]
valid_trafo = [data.UseLabelsAsImages(),
               data.PadImages(pad[0], pad[1], pad[2], pad_value=0),
               data.ToTensor()]

ds_train, ds_valid = data.get_toy_shape_training_data(train_trafo, valid_trafo,
                                                      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                                                      [16, 17, 18, 19], batchsize=batchsize)
#ds_train, ds_valid = data.get_stroke_shape_training_data(modalities, labels, train_trafo, valid_trafo,
#                                                         list(range(32)), 0.275, seed=4, batchsize=batchsize,
#                                                         split=True)

rnn = RnnUnet(Unet(channels_unet), RnnCell(channels_rnn), seq_len=sequence_length).cuda()
#rnn = torch.load('/share/data_zoe1/lucas/NOT_IN_BACKUP/tmp/rnn_217.model')

params = [p for p in rnn.parameters() if p.requires_grad]
print('# optimizing params', sum([p.nelement() * p.requires_grad for p in params]),
      '/ total: Unet-RNN', sum([p.nelement() for p in rnn.parameters()]))

criterion = Criterion([1], 168*168*68, seq_len=sequence_length)  # Criterion([195. / 444., 191. / 444., 58. / 444.], 168*168*68)
optimizer = torch.optim.Adam(params, lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.1)

loss_train = []
loss_valid = []

for epoch in range(0, 225):
    scheduler.step()
    f, axarr = plt.subplots(6, 10)
    loss_mean = 0
    inc = 0
    train = True
    rnn.freeze(not train)
    rnn.train(train)

    for batch in ds_train:
        to_ta = batch[data.KEY_GLOBAL][:, 0, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5).cuda()
        ta_tr = batch[data.KEY_GLOBAL][:, 1, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5).cuda()
        ta = Variable(to_ta.type(torch.FloatTensor)).cuda()
        tn = Variable(torch.ones(batch[data.KEY_GLOBAL].size()[0], 1, 1, 1, 1).type(torch.FloatTensor) * normalize).cuda()
        tr = Variable((to_ta + ta_tr).type(torch.FloatTensor)).cuda()
        additional = torch.ones(4, 2, 68, 168, 168) * batch[data.KEY_GLOBAL]
        inputs = torch.cat((batch[data.KEY_IMAGES], additional), dim=1)
        inputs = Variable(inputs).cuda()
        core_gt = Variable(batch[data.KEY_LABELS][:, 0, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5)).cuda()
        fuct_gt = Variable(batch[data.KEY_LABELS][:, 1, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5)).cuda()
        penu_gt = Variable(batch[data.KEY_LABELS][:, 2, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5)).cuda()

        del batch

        predicts, h_states = rnn(inputs)

        loss = criterion(torch.cat(predicts, dim=1),
                         torch.cat([core_gt, fuct_gt, penu_gt], dim=1))
        loss_mean += float(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.save(rnn, '/share/data_zoe1/lucas/NOT_IN_BACKUP/tmp/rnn_seq_simple.model')
        inc += 1

    loss_train.append(loss_mean/inc)

    for row in range(n_visual_samples):
        axarr[row, 0].imshow(inputs.cpu().data.numpy()[row, 0, zslice+pad[2], pad[1]:-pad[1], pad[0]:-pad[0]], vmin=0, vmax=12, cmap='jet')
        axarr[row, 1].imshow(core_gt.cpu().data.numpy()[row, 0, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        axarr[row, 2].imshow(predicts[0].cpu().data.numpy()[row, 0, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        axarr[row, 3].imshow(fuct_gt.cpu().data.numpy()[row, 0, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        axarr[row, 4].imshow(predicts[1].cpu().data.numpy()[row, 0, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        axarr[row, 5].imshow(penu_gt.cpu().data.numpy()[row, 0, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        axarr[row, 6].imshow(predicts[-1].cpu().data.numpy()[row, -1, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        axarr[row, 7].imshow(inputs.cpu().data.numpy()[row, 1, zslice+pad[2], pad[1]:-pad[1], pad[0]:-pad[0]], vmin=0, vmax=35, cmap='jet')
        axarr[row, 8].imshow(inputs.cpu().data.numpy()[row, 2, zslice+pad[2], pad[1]:-pad[1], pad[0]:-pad[0]], vmin=0, cmap='gray')
        axarr[row, 9].imshow(h_states[-1].cpu().data.numpy()[row, 0, zslice, :, :], cmap='gray')
        titles = ['CBV',
                  'Core', 'p({:02.1f})'.format(float(ta[row, :, :, :, :])),
                  'Lesion', 'p({:02.1f})'.format(float(tr[row, :, :, :, :])),
                  'Penumbra', 'p({:02.1f})'.format(float(tn[row, :, :, :, :])),
                  'TTD',
                  'Clinical',
                  'hidden']
        for ax, title in zip(axarr[row], titles):
            ax.set_title(title)

    del loss
    del inputs
    del core_gt
    del fuct_gt
    del penu_gt
    del predicts
    del to_ta
    del ta_tr
    del ta
    del tr
    del tn
    del h_states
    optimizer.zero_grad()

    train = False
    rnn.train(train)
    rnn.freeze(not train)

    # in validation: PENU_PR BASED ON FUCT_PR!!!!!!
    for batch in ds_valid:
        to_ta = batch[data.KEY_GLOBAL][:, 0, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5).cuda()
        ta_tr = batch[data.KEY_GLOBAL][:, 1, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5).cuda()
        ta = Variable(to_ta.type(torch.FloatTensor), volatile=not train).cuda()
        tn = Variable(torch.ones(batch[data.KEY_GLOBAL].size()[0], 1, 1, 1, 1).type(torch.FloatTensor) * normalize, volatile=not train).cuda()
        tr = Variable((to_ta + ta_tr).type(torch.FloatTensor), volatile=not train).cuda()
        additional = torch.ones(4, 2, 68, 168, 168) * batch[data.KEY_GLOBAL]
        inputs = torch.cat((batch[data.KEY_IMAGES], additional), dim=1)
        inputs = Variable(inputs, volatile=not train).cuda()
        core_gt = Variable(batch[data.KEY_LABELS][:, 0, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5), volatile=not train).cuda()
        fuct_gt = Variable(batch[data.KEY_LABELS][:, 1, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5), volatile=not train).cuda()
        penu_gt = Variable(batch[data.KEY_LABELS][:, 2, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5), volatile=not train).cuda()

        del batch

        predicts, h_states = rnn(inputs)

        loss = criterion(torch.cat(predicts, dim=1),
                         torch.cat([core_gt, fuct_gt, penu_gt], dim=1))

    for row in range(n_visual_samples):
        axarr[row+n_visual_samples, 0].imshow(inputs.cpu().data.numpy()[row, 0, zslice + 20, 20:-20, 20:-20], vmin=0, vmax=12,
                             cmap='jet')
        axarr[row+n_visual_samples, 1].imshow(core_gt.cpu().data.numpy()[row, 0, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        axarr[row+n_visual_samples, 2].imshow(predicts[0].cpu().data.numpy()[row, 0, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        axarr[row+n_visual_samples, 3].imshow(fuct_gt.cpu().data.numpy()[row, 0, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        axarr[row+n_visual_samples, 4].imshow(predicts[1].cpu().data.numpy()[row, 0, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        axarr[row+n_visual_samples, 5].imshow(penu_gt.cpu().data.numpy()[row, 0, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        axarr[row+n_visual_samples, 6].imshow(predicts[-1].cpu().data.numpy()[row, 0, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        axarr[row+n_visual_samples, 7].imshow(inputs.cpu().data.numpy()[row, 1, zslice + 20, 20:-20, 20:-20], vmin=0, vmax=35,
                             cmap='jet')
        axarr[row+n_visual_samples, 8].imshow(inputs.cpu().data.numpy()[row, 2, zslice + 20, 20:-20, 20:-20], vmin=0, cmap='gray')
        axarr[row+n_visual_samples, 9].imshow(h_states[-1].cpu().data.numpy()[row, row+n_visual_samples, zslice, :, :], cmap='gray')
        titles = ['CBV',
                  'Core', 'p({:02.1f})'.format(float(ta[row, :, :, :, :])),
                  'Lesion', 'p({:02.1f})'.format(float(tr[row, :, :, :, :])),
                  'Penumbra', 'p({:02.1f})'.format(float(tn[row, :, :, :, :])),
                  'TTD',
                  'Clinical',
                  'hidden']
        for ax, title in zip(axarr[row+2], titles):
            ax.set_title(title)

    for ax in axarr.flatten():
        ax.title.set_fontsize(3)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    loss_valid.append(float(loss))

    print('Epoch', epoch, 'last batch training loss:', loss_train[-1], '\tvalidation batch loss:', loss_valid[-1])

    f.subplots_adjust(hspace=0.05)
    f.savefig('/share/data_zoe1/lucas/NOT_IN_BACKUP/tmp/rnn_seq_simple' + str(epoch) + '.png', bbox_inches='tight', dpi=300)

    del f
    del axarr

    if epoch > 0:
        fig, plot = plt.subplots()
        epochs = range(1, epoch + 2)
        plot.plot(epochs, loss_train, 'r-')
        plot.plot(epochs, loss_valid, 'b-')
        plot.set_ylabel('Loss Training (r) & Validation (b)')
        plot.set_ylim(0, 0.8)
        fig.savefig('/share/data_zoe1/lucas/NOT_IN_BACKUP/tmp/rnn_seq_simple_plots.png', bbox_inches='tight', dpi=300)
        del plot
        del fig
