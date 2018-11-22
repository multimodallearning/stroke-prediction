import torch
import torch.nn as nn
from torch.autograd import Variable
from common import data, metrics
import common.model.Unet3D as UnetHelper
import matplotlib.pyplot as plt


def crop(tensor_in, crop_as, dims=[]):
    assert len(dims) > 0, "Specify dimensions to be cropped"
    result = tensor_in
    for dim in dims:
        result = result.narrow(dim, (tensor_in.size()[dim] - crop_as.size()[dim]) // 2, crop_as.size()[dim])
    return result


class Unet(nn.Module):
    def __init__(self, channels, final_activation, channel_dim=1, channels_crop=[2,3,4]):
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
            final_activation
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


class Rnn(nn.Module):
    def __init__(self, unet_channels=[3, 20, 24, 64, 24, 20, 16], channels_clinical=5, padding=(0,0,0)):
        super(Rnn, self).__init__()

        channels_input = unet_channels[0]
        channels_hidden = unet_channels[-1]

        self.hh = Unet(unet_channels, nn.Tanh())

        self.ho = nn.Sequential(
            nn.InstanceNorm3d(channels_hidden),
            nn.Conv3d(channels_hidden, channels_hidden, 1),
            nn.ReLU(True),
            nn.InstanceNorm3d(channels_hidden),
            nn.Conv3d(channels_hidden, channels_hidden // 2, 1),
            nn.ReLU(True),
            nn.InstanceNorm3d(channels_hidden // 2),
            nn.Conv3d(channels_hidden // 2, 1, 1),
            nn.Sigmoid()
        )

        ch_intermediate = (channels_clinical + channels_input) // 2

        self.hc = nn.Sequential(
            nn.InstanceNorm3d(channels_input),
            nn.Conv3d(channels_input, ch_intermediate, 3, stride=2, padding=1),  # 64x64x14
            nn.ReLU(True),
            nn.AdaptiveAvgPool3d((3,3,3)),  # 3x3x3
            nn.InstanceNorm3d(ch_intermediate),
            nn.Conv3d(ch_intermediate, channels_clinical, 3, stride=1, padding=0),  # 1x1x1
            nn.ReLU(True)
        )

        ch_intermediate = (2*channels_clinical + channels_hidden) // 2

        self.weight = nn.Sequential(
            nn.Conv3d(channels_clinical * 2, ch_intermediate, 1),
            nn.ReLU(True),
            nn.Conv3d(ch_intermediate, channels_hidden, 1),
            nn.Sigmoid()
        )

        self.input_padding = nn.modules.ConstantPad3d((padding[2], padding[2],
                                                       padding[1], padding[1],
                                                       padding[0], padding[0]), value=0)

    def forward_single(self, inputs, clinicals, pre_output, hidden):
        combined = torch.cat((inputs, self.input_padding(pre_output), self.input_padding(hidden)), dim=data.DIM_CHANNEL_TORCH3D_5)
        hidden = self.hh(combined)
        crop_combined = crop(combined, hidden, [2,3,4])
        weight = self.weight(torch.cat((self.hc(crop_combined), clinicals), dim=data.DIM_CHANNEL_TORCH3D_5))
        return self.ho(hidden * weight), hidden

    def forward(self, inputs, clinicals, pre_output_init, hidden_init):
        pr_core, hidden = self.forward_single(inputs, clinicals, pre_output_init, hidden_init)
        pr_fuct, hidden = self.forward_single(inputs, clinicals, pr_core, hidden)
        pr_penu, _ = self.forward_single(inputs, clinicals, pr_fuct, hidden)
        return pr_core, pr_fuct, pr_penu

    def freeze(self, freeze=False):
        requires_grad = not freeze
        for param in self.parameters():
            param.requires_grad = requires_grad


class Criterion(nn.Module):
    def __init__(self, weights, normalize, alpha=[799./1000., 100./1000., 100./1000., 1./1000.]):
        super(Criterion, self).__init__()
        self.dc = metrics.BatchDiceLoss(weights)  # weighted inversely by each volume proportion
        self.normalize = normalize
        self.alpha = alpha
        #self.mp4 = nn.MaxPool3d(4, 4)
        #self.mp2 = nn.MaxPool3d(2, 2)
        #self.up4 = nn.Upsample(scale_factor=4)
        #self.up2 = nn.Upsample(scale_factor=2)

        self.non_neg = Variable(torch.FloatTensor([0])).cuda()

    def forward(self, pred, target):
        loss = 0

        loss += self.alpha[0] * self.dc(pred, target)

        diff_penu_fuct = pred[:, 2] - pred[:, 1]
        diff_penu_core = pred[:, 2] - pred[:, 0]
        loss += self.alpha[1] * torch.mean(torch.abs(diff_penu_fuct) - diff_penu_fuct)  # monotone
        loss += self.alpha[2] * torch.mean(torch.abs(diff_penu_core) - diff_penu_core)  # monotone

        diff_left_right = pred[:, :, :, :, :64] - pred[:, :, :, :, 64:]
        diff_right_left = pred[:, :, :, :, 64:] - pred[:, :, :, :, :64]
        tmp_lr = torch.max(pred[:, :, :, :, :64] - torch.abs(diff_left_right), self.non_neg.expand_as(diff_left_right))
        tmp_rl = torch.max(pred[:, :, :, :, 64:] - torch.abs(diff_right_left), self.non_neg.expand_as(diff_right_left))
        loss += self.alpha[3] * (torch.mean(tmp_lr) + torch.mean(tmp_rl))/2  # exlusive hemisphere

        return loss


modalities = ['_CBV_reg1_downsampled',
              '_TTD_reg1_downsampled']
labels = ['_CBVmap_subset_reg1_downsampled',
          '_FUCT_MAP_T_Samplespace_subset_reg1_downsampled',
          '_TTDmap_subset_reg1_downsampled']

channels_clinical = 5
channels_hidden = 5
channels_input = 3 + channels_hidden  # CBV, TTD, previous output
channels_unet = [channels_input, 32, 48, 96, 48, 32, channels_hidden]
batchsize = 2
normalize = 24
zslice = 14
pad = (20, 20, 20)
n_visual_samples = 2

train_trafo = [data.ResamplePlaneXY(0.5),
               data.PadImages(pad[0], pad[1], pad[2], pad_value=0),
               data.HemisphericFlip(),
               data.ElasticDeform(apply_to_images=True, random=0.5),
               data.ToTensor()]
valid_trafo = [data.ResamplePlaneXY(0.5),
               data.PadImages(pad[0], pad[1], pad[2], pad_value=0),
               data.HemisphericFlipFixedToCaseId(14),
               data.ToTensor()]
ds_train, ds_valid = data.get_stroke_shape_training_data(modalities, labels, train_trafo, valid_trafo,
                                                         list(range(32)), 0.3, seed=4, batchsize=batchsize,
                                                         split=True)

rnn = Rnn(channels_unet, channels_clinical=channels_clinical, padding=pad).cuda()
#rnn = torch.load('/share/data_zoe1/lucas/NOT_IN_BACKUP/tmp/rnn_217.model')

params = [p for p in rnn.parameters() if p.requires_grad]
print('# optimizing params', sum([p.nelement() * p.requires_grad for p in params]),
      '/ total: Unet-RNN', sum([p.nelement() for p in rnn.parameters()]))

criterion = Criterion([195. / 444., 191. / 444., 58. / 444.], 168*168*68)

optimizer = torch.optim.Adam(params, lr=0.001)

for epoch in range(0, 100):
    f, axarr = plt.subplots(n_visual_samples*2, 9)
    loss_mean = 0
    inc = 0
    train = True
    rnn.freeze(not train)
    rnn.train(train)

    for batch in ds_train:
        to_ta = batch[data.KEY_GLOBAL][:, 0, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5).cuda()
        ta_tr = batch[data.KEY_GLOBAL][:, 1, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5).cuda()
        ta = to_ta
        tn = normalize * torch.ones(batchsize, 1, 1, 1, 1)
        tr = to_ta + ta_tr
        core_gt = Variable(batch[data.KEY_LABELS][:, 0, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5), volatile=not train).cuda()
        fuct_gt = Variable(batch[data.KEY_LABELS][:, 1, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5), volatile=not train).cuda()
        penu_gt = Variable(batch[data.KEY_LABELS][:, 2, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5), volatile=not train).cuda()
        inputs = Variable(batch[data.KEY_IMAGES], volatile=not train).cuda()
        output_init = Variable(torch.zeros(batchsize, 1, 28, 128, 128), volatile=not train).cuda()
        hidden_init = Variable(torch.zeros(batchsize, channels_hidden, 28, 128, 128), volatile=not train).cuda()
        clinicals = Variable(batch[data.KEY_GLOBAL], volatile=not train).cuda()

        core_pr, fuct_pr, penu_pr = rnn(inputs, clinicals, output_init, hidden_init)

        loss = criterion(torch.cat([core_pr, fuct_pr, penu_pr], dim=1),
                         torch.cat([core_gt, fuct_gt, penu_gt], dim=1))
        loss_mean += float(loss)

        del batch

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.save(rnn, '/share/data_zoe1/lucas/NOT_IN_BACKUP/tmp/rnn_iterative_weightedDice_monoLoss_smallBatch_hiddenInput_sideLoss.model')
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
        axarr[row, 8].imshow(output_init.cpu().data.numpy()[row, 0, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        titles = ['Core',
                  'Lesion',
                  'Penumbra',
                  'CBV',
                  'TTD',
                  'p({:02.1f}h)'.format(float(ta[row, :, :, :, :])),
                  'p({:02.1f}h)'.format(float(tr[row, :, :, :, :])),
                  'p({:02.1f}h)'.format(float(tn[row, :, :, :, :])),
                  'InitPrev']
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
    optimizer.zero_grad()

    train = False
    rnn.train(train)
    rnn.freeze(not train)
    loss_string = 'custom'

    for batch in ds_valid:
        to_ta = batch[data.KEY_GLOBAL][:, 0, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5).cuda()
        ta_tr = batch[data.KEY_GLOBAL][:, 1, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5).cuda()
        ta = to_ta
        tn = normalize * torch.ones(batchsize, 1, 1, 1, 1)
        tr = to_ta + ta_tr
        core_gt = Variable(batch[data.KEY_LABELS][:, 0, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5), volatile=not train).cuda()
        fuct_gt = Variable(batch[data.KEY_LABELS][:, 1, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5), volatile=not train).cuda()
        penu_gt = Variable(batch[data.KEY_LABELS][:, 2, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5), volatile=not train).cuda()
        inputs = Variable(batch[data.KEY_IMAGES], volatile=not train).cuda()
        output_init = Variable(torch.zeros(batchsize, 1, 28, 128, 128), volatile=not train).cuda()
        hidden_init = Variable(torch.zeros(batchsize, channels_hidden, 28, 128, 128), volatile=not train).cuda()
        clinicals = Variable(batch[data.KEY_GLOBAL], volatile=not train).cuda()

        core_pr, fuct_pr, penu_pr = rnn(inputs, clinicals, output_init, hidden_init)

        loss = criterion(torch.cat([core_pr, fuct_pr, penu_pr], dim=1),
                         torch.cat([core_gt, fuct_gt, penu_gt], dim=1))

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
        axarr[row+n_visual_samples, 8].imshow(output_init.cpu().data.numpy()[row, 0, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        titles = ['Core',
                  'Lesion',
                  'Penumbra',
                  'CBV',
                  'TTD',
                  'p({:02.1f}h)'.format(float(ta[row, :, :, :, :])),
                  'p({:02.1f}h)'.format(float(tr[row, :, :, :, :])),
                  'p({:02.1f}h)'.format(float(tn[row, :, :, :, :])),
                  'InitPrev']
        for ax, title in zip(axarr[row+n_visual_samples], titles):
            ax.set_title(title)

    for ax in axarr.flatten():
        ax.title.set_fontsize(3)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    print('Epoch {:02} ({}) last batch training loss: {:02.3f}\tvalidation batch loss: {:02.3f}'.format(epoch, loss_string, loss_mean/inc, float(loss)))

    f.subplots_adjust(hspace=0.05)
    f.savefig('/share/data_zoe1/lucas/NOT_IN_BACKUP/tmp/rnn_iterative_weightedDice_monoLoss_smallBatch_hiddenInput_sideLoss_' + str(epoch) + '.png', bbox_inches='tight', dpi=300)

    del f
    del axarr

