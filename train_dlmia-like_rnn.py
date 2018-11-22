import torch
import torch.nn as nn
from torch.autograd import Variable
from common import data, metrics
import common.model.Unet3D as UnetHelper
import matplotlib.pyplot as plt
from torch.nn.modules.loss import _Loss as LossModule


class CombinedLoss(LossModule):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.criterion1 = metrics.BatchDiceLoss([1./3., 1./3., 1./3.])
        self.criterion2 = nn.L1Loss()

    def forward(self, core_pr, fuct_pr, penu_pr, t_core, t_fuct, t_penu,
                core_gt, fuct_gt, penu_gt, time_core, time_fuct, time_penu):
        dice = self.criterion1(torch.cat([core_pr, fuct_pr, penu_pr], dim=1),
                               torch.cat([core_gt, fuct_gt, penu_gt], dim=1))
        time = self.criterion2(torch.cat([t_core, t_fuct, t_penu], dim=1),
                               torch.cat([time_core, time_fuct, time_penu], dim=1))
        return dice #* time


def get_normalization(batch, normalization=24):
    to_to_ta = batch[data.KEY_GLOBAL][:, 0, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5).type(torch.FloatTensor)
    normalization = torch.ones(to_to_ta.size()[0], 1).type(torch.FloatTensor) * \
                    normalization - to_to_ta.squeeze().unsqueeze(data.DIM_CHANNEL_TORCH3D_5)
    return normalization


def get_time_to_treatment(batch, normalize=24, step=None):
    normalization = get_normalization(batch, normalize)
    if step is None:
        ta_to_tr = batch[data.KEY_GLOBAL][:, 1, :, :, :].squeeze().unsqueeze(data.DIM_CHANNEL_TORCH3D_5)
        time_to_treatment = Variable(ta_to_tr.type(torch.FloatTensor) / normalization)
    else:
        time_to_treatment = Variable((step * torch.ones(batch[data.KEY_GLOBAL].size()[0], 1)) / normalization)
    return time_to_treatment.unsqueeze(2).unsqueeze(3).unsqueeze(4)


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
            nn.ReLU(True),
            nn.InstanceNorm3d(channels),
            nn.Conv3d(channels, channels, 1),
            nn.Tanh()
        )

        self.ho = nn.Sequential(
            nn.InstanceNorm3d(channels),
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.ReLU(True),
            nn.InstanceNorm3d(channels),
            nn.Conv3d(channels, channels//2, 3, padding=1),
            nn.ReLU(True),
            nn.InstanceNorm3d(channels // 2),
            nn.Conv3d(channels // 2, channels // 2, 1),
            nn.ReLU(True),
            nn.InstanceNorm3d(channels//2),
            nn.Conv3d(channels//2, 1, 3, padding=1),
            nn.Sigmoid()
        )

        self.ht = nn.Sequential(
            nn.InstanceNorm3d(channels),
            nn.Conv3d(channels, channels//2, 1),
            nn.ReLU(True),
            nn.InstanceNorm3d(channels//2),
            nn.AdaptiveAvgPool3d((8, 8, 2)),
            nn.ReLU(True),
            nn.InstanceNorm3d(channels//2),
            nn.Conv3d(channels//2, channels//2, 1),
            nn.ReLU(True),
            nn.InstanceNorm3d(channels//2),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.ReLU(True),
            nn.Conv3d(channels // 2, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, h_state):
        return self.hh(h_state), self.ho(h_state), self.ht(h_state)


class RnnUnet(nn.Module):
    def __init__(self, unet, cell):
        super(RnnUnet, self).__init__()
        self.unet = unet
        self.cell = cell

    def forward(self, input):
        h_state_unet = self.unet(input)
        h_state, pr_core, t_core = self.cell(h_state_unet)
        h_state, pr_fuct, t_fuct = self.cell(h_state)
        _, pr_penu, t_penu = self.cell(h_state)
        return pr_core, pr_fuct, pr_penu, t_core, t_fuct, t_penu, h_state_unet

    def freeze(self, freeze=False):
        requires_grad = not freeze
        for param in self.parameters():
            param.requires_grad = requires_grad


modalities = ['_CBV_reg1_downsampled',
              '_TTD_reg1_downsampled']
labels = ['_CBVmap_subset_reg1_downsampled',
          '_FUCT_MAP_T_Samplespace_subset_reg1_downsampled',
          '_TTDmap_subset_reg1_downsampled']

channels_rnn = 16
channels_unet = [3, 16, 32, 64, 32, 16, channels_rnn]
batchsize = 4
normalize = 24
zslice = 14
pad = (20, 20, 20)
n_visual_samples = 3

train_trafo = [data.ResamplePlaneXY(0.5),
               data.PadImages(pad[0], pad[1], pad[2], pad_value=0),
               data.HemisphericFlip(),
               data.ElasticDeform(apply_to_images=True),
               data.ToTensor()]
valid_trafo = [data.ResamplePlaneXY(0.5),
               data.PadImages(pad[0], pad[1], pad[2], pad_value=0),
               data.HemisphericFlipFixedToCaseId(14),
               data.ToTensor()]
ds_train, ds_valid = data.get_stroke_shape_training_data(modalities, labels, train_trafo, valid_trafo,
                                                         list(range(32)), 0.3, seed=4, batchsize=batchsize,
                                                         split=True)

rnn = RnnUnet(Unet(channels_unet), RnnCell(channels_rnn)).cuda()
#rnn = torch.load('/share/data_zoe1/lucas/NOT_IN_BACKUP/tmp/rnn_217.model')

params = [p for p in rnn.parameters() if p.requires_grad]
print('# optimizing params', sum([p.nelement() * p.requires_grad for p in params]),
      '/ total: Unet-RNN', sum([p.nelement() for p in rnn.parameters()]))

criterion = CombinedLoss()
optimizer = torch.optim.Adam(params, lr=0.001)

for epoch in range(0, 100):
    f, axarr = plt.subplots(6, 11)
    loss_mean = 0
    inc = 0
    train = True
    rnn.freeze(not train)
    rnn.train(train)

    for batch in ds_train:
        to_ta = batch[data.KEY_GLOBAL][:, 0, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5).cuda()
        ta_tr = batch[data.KEY_GLOBAL][:, 1, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5).cuda()
        ta = to_ta
        tn = torch.ones(batch[data.KEY_GLOBAL][:, 0, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5).size()) * 24
        tr = to_ta + ta_tr
        input_size = batch[data.KEY_IMAGES][:, 0, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5).size()
        inputs = torch.cat((batch[data.KEY_IMAGES], torch.ones(input_size)), dim=1)
        assert batch[data.KEY_GLOBAL].size()[1] == 5
        inputs[:, 2, :, :, :33] *= batch[data.KEY_GLOBAL][:, 0, :, :, :]
        inputs[:, 2, :, :, 33:67] *= batch[data.KEY_GLOBAL][:, 1, :, :, :]
        inputs[:, 2, :, :, 67:101] *= batch[data.KEY_GLOBAL][:, 2, :, :, :]/42  #NIHSS
        inputs[:, 2, :, :, 101:135] *= batch[data.KEY_GLOBAL][:, 3, :, :, :]/100  # age
        inputs[:, 2, :, :, 135:] *= batch[data.KEY_GLOBAL][:, 4, :, :, :]
        inputs = Variable(inputs).cuda()
        core_gt = Variable(batch[data.KEY_LABELS][:, 0, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5)).cuda()
        fuct_gt = Variable(batch[data.KEY_LABELS][:, 1, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5)).cuda()
        penu_gt = Variable(batch[data.KEY_LABELS][:, 2, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5)).cuda()

        core_pr, fuct_pr, penu_pr, t_core, t_fuct, t_penu, h_state_unet = rnn(inputs)

        time_core = get_time_to_treatment(batch, normalize, step=0).cuda()
        time_fuct = get_time_to_treatment(batch, normalize).cuda()
        time_penu = Variable(torch.ones(batch[data.KEY_GLOBAL][:, 0, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5).size())).cuda()

        loss = criterion(core_pr, fuct_pr, penu_pr, t_core, t_fuct, t_penu,
                         core_gt, fuct_gt, penu_gt, time_core, time_fuct, time_penu)
        loss_mean += float(loss)

        del batch

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.save(rnn, '/share/data_zoe1/lucas/NOT_IN_BACKUP/tmp/rnn_DCMonly.model')
        inc += 1

    for row in range(n_visual_samples):
        axarr[row, 3].imshow(inputs.cpu().data.numpy()[row, 0, zslice+pad[2], pad[1]:-pad[1], pad[0]:-pad[0]], vmin=0, vmax=12, cmap='jet')
        axarr[row, 0].imshow(core_gt.cpu().data.numpy()[row, 0, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        axarr[row, 5].imshow(core_pr.cpu().data.numpy()[row, 0, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        axarr[row, 1].imshow(fuct_gt.cpu().data.numpy()[row, 0, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        axarr[row, 6].imshow(fuct_pr.cpu().data.numpy()[row, 0, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        axarr[row, 2].imshow(penu_gt.cpu().data.numpy()[row, 0, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        axarr[row, 7].imshow(penu_pr.cpu().data.numpy()[row, 0, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        axarr[row, 4].imshow(inputs.cpu().data.numpy()[row, 1, zslice+pad[2], pad[1]:-pad[1], pad[0]:-pad[0]], vmin=0, vmax=35, cmap='jet')
        axarr[row, 8].imshow(inputs.cpu().data.numpy()[row, 2, zslice+pad[2], pad[1]:-pad[1], pad[0]:-pad[0]], vmin=0, cmap='gray')
        axarr[row, 9].imshow(h_state_unet.cpu().data.numpy()[row, row, zslice, :, :], cmap='gray')
        axarr[row, 10].imshow(h_state_unet.cpu().data.numpy()[row, 1*6+row, zslice, :, :], cmap='gray')
        titles = ['{:01.0f}: Core'.format(float(time_core[row, :, :, :, :])),
                  '{:01.2f}: Lesion'.format(float(time_fuct[row, :, :, :, :])),
                  '{:01.0f}: Penumbra'.format(float(time_penu[row, :, :, :, :])),
                  'CBV',
                  'TTD',
                  '{:02.2f}: p({:02.1f}h)'.format(float(t_core[row, :, :, :, :]), float(ta[row, :, :, :, :])),
                  '{:02.2f}: p({:02.1f}h)'.format(float(t_fuct[row, :, :, :, :]), float(tr[row, :, :, :, :])),
                  '{:02.2f}: p({:02.1f}h)'.format(float(t_penu[row, :, :, :, :]), float(tn[row, :, :, :, :])),
                  'Clinical',
                  'hidden', 'hidden', 'hidden', 'hidden']
        for ax, title in zip(axarr[row], titles):
            ax.set_title(title)

    del loss
    del inputs
    del core_gt
    del core_pr
    del fuct_gt
    del fuct_pr
    del penu_gt
    del penu_pr
    del to_ta
    del ta_tr
    del ta
    del tr
    del tn
    del t_core
    del t_fuct
    del t_penu
    del input_size
    del h_state_unet
    optimizer.zero_grad()

    train = False
    rnn.train(train)
    rnn.freeze(not train)

    for batch in ds_valid:
        to_ta = batch[data.KEY_GLOBAL][:, 0, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5).cuda()
        ta_tr = batch[data.KEY_GLOBAL][:, 1, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5).cuda()
        ta = to_ta
        tn = torch.ones(batch[data.KEY_GLOBAL][:, 0, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5).size()) * 24
        tr = to_ta + ta_tr
        input_size = batch[data.KEY_IMAGES][:, 0, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5).size()
        inputs = torch.cat((batch[data.KEY_IMAGES], torch.ones(input_size)), dim=1)
        assert batch[data.KEY_GLOBAL].size()[1] == 5
        inputs[:, 2, :, :, :33] *= batch[data.KEY_GLOBAL][:, 0, :, :, :]
        inputs[:, 2, :, :, 33:67] *= batch[data.KEY_GLOBAL][:, 1, :, :, :]
        inputs[:, 2, :, :, 67:101] *= batch[data.KEY_GLOBAL][:, 2, :, :, :]/42  #NIHSS
        inputs[:, 2, :, :, 101:135] *= batch[data.KEY_GLOBAL][:, 3, :, :, :]/100  # age
        inputs[:, 2, :, :, 135:] *= batch[data.KEY_GLOBAL][:, 4, :, :, :]
        inputs = Variable(inputs, volatile=not train).cuda()
        core_gt = Variable(batch[data.KEY_LABELS][:, 0, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5), volatile=not train).cuda()
        fuct_gt = Variable(batch[data.KEY_LABELS][:, 1, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5), volatile=not train).cuda()
        penu_gt = Variable(batch[data.KEY_LABELS][:, 2, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5), volatile=not train).cuda()

        core_pr, fuct_pr, penu_pr, t_core, t_fuct, t_penu, h_state_unet = rnn(inputs)

        time_core = get_time_to_treatment(batch, normalize, step=0).cuda()
        time_fuct = get_time_to_treatment(batch, normalize).cuda()
        time_penu = Variable(torch.ones(batch[data.KEY_GLOBAL][:, 0, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5).size())).cuda()

        loss = criterion(core_pr, fuct_pr, penu_pr, t_core, t_fuct, t_penu,
                         core_gt, fuct_gt, penu_gt, time_core, time_fuct, time_penu)

        del batch

    for row in range(n_visual_samples):
        axarr[row+n_visual_samples, 3].imshow(inputs.cpu().data.numpy()[row, 0, zslice+pad[2], pad[1]:-pad[1], pad[0]:-pad[0]], vmin=0, vmax=12, cmap='jet')
        axarr[row+n_visual_samples, 0].imshow(core_gt.cpu().data.numpy()[row, 0, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        axarr[row+n_visual_samples, 5].imshow(core_pr.cpu().data.numpy()[row, 0, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        axarr[row+n_visual_samples, 1].imshow(fuct_gt.cpu().data.numpy()[row, 0, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        axarr[row+n_visual_samples, 6].imshow(fuct_pr.cpu().data.numpy()[row, 0, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        axarr[row+n_visual_samples, 2].imshow(penu_gt.cpu().data.numpy()[row, 0, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        axarr[row+n_visual_samples, 7].imshow(penu_pr.cpu().data.numpy()[row, 0, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        axarr[row+n_visual_samples, 4].imshow(inputs.cpu().data.numpy()[row, 1, zslice+pad[2], pad[1]:-pad[1], pad[0]:-pad[0]], vmin=0, vmax=35, cmap='jet')
        axarr[row+n_visual_samples, 8].imshow(inputs.cpu().data.numpy()[row, 2, zslice+pad[2], pad[1]:-pad[1], pad[0]:-pad[0]], vmin=0, cmap='gray')
        axarr[row+n_visual_samples, 9].imshow(h_state_unet.cpu().data.numpy()[row, row+n_visual_samples, zslice, :, :], cmap='gray')
        axarr[row+n_visual_samples, 10].imshow(h_state_unet.cpu().data.numpy()[row, 1*6+row+n_visual_samples, zslice, :, :], cmap='gray')
        titles = ['{:01.0f}: Core'.format(float(time_core[row, :, :, :, :])),
                  '{:01.2f}: Lesion'.format(float(time_fuct[row, :, :, :, :])),
                  '{:01.0f}: Penumbra'.format(float(time_penu[row, :, :, :, :])),
                  'CBV',
                  'TTD',
                  '{:02.2f}: p({:02.1f}h)'.format(float(t_core[row, :, :, :, :]), float(ta[row, :, :, :, :])),
                  '{:02.2f}: p({:02.1f}h)'.format(float(t_fuct[row, :, :, :, :]), float(tr[row, :, :, :, :])),
                  '{:02.2f}: p({:02.1f}h)'.format(float(t_penu[row, :, :, :, :]), float(tn[row, :, :, :, :])),
                  'Clinical',
                  'hidden', 'hidden', 'hidden', 'hidden']
        for ax, title in zip(axarr[row+n_visual_samples], titles):
            ax.set_title(title)

    for ax in axarr.flatten():
        ax.title.set_fontsize(3)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    print('Epoch', epoch, 'last batch training loss:', loss_mean/inc, '\tvalidation batch loss:', float(loss))

    f.subplots_adjust(hspace=0.05)
    f.savefig('/share/data_zoe1/lucas/NOT_IN_BACKUP/tmp/rnn_DCMonly' + str(epoch) + '.png', bbox_inches='tight', dpi=300)

    del f
    del axarr

