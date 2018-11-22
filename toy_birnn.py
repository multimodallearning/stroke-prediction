import torch
import torch.nn as nn
from torch.autograd import Variable
from common import data, metrics
import matplotlib.pyplot as plt


def get_normalization(batch, normalization=24):
    to_to_ta = batch[data.KEY_GLOBAL][:, 0, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5).type(torch.FloatTensor)
    normalization = torch.ones(to_to_ta.size()[0], 1).type(torch.FloatTensor) * \
                    normalization - to_to_ta.squeeze().unsqueeze(data.DIM_CHANNEL_TORCH3D_5)
    return normalization


def get_time_to_treatment(batch, global_variables, step):
    normalization = get_normalization(batch)
    if step is None:
        ta_to_tr = batch[data.KEY_GLOBAL][:, 1, :, :, :].squeeze().unsqueeze(data.DIM_CHANNEL_TORCH3D_5)
        time_to_treatment = Variable(ta_to_tr.type(torch.FloatTensor) / normalization)
    else:
        time_to_treatment = Variable((step * torch.ones(global_variables.size()[0], 1)) / normalization)
    return time_to_treatment.unsqueeze(2).unsqueeze(3).unsqueeze(4)


class CaeBase(nn.Module):
    def __init__(self, size_input_xy=128, size_input_z=28, channels=[1, 16, 32, 64, 128, 1024, 128, 1], n_ch_global=2,
                 alpha=0.01, inner_xy=12, inner_z=3):
        super().__init__()
        assert size_input_xy % 4 == 0 and size_input_z % 4 == 0
        self.n_ch_origin = channels[1]
        self.n_ch_down2x = channels[2]
        self.n_ch_down4x = channels[3]
        self.n_ch_down8x = channels[4]
        self.n_ch_fc = channels[5]

        self._inner_ch = self.n_ch_down8x
        self._inner_xy = inner_xy
        self._inner_z = inner_z

        self.n_ch_global = n_ch_global
        self.n_input = channels[0]
        self.n_classes = channels[-1]
        self.alpha = alpha

    def freeze(self, freeze=False):
        requires_grad = not freeze
        for param in self.parameters():
            param.requires_grad = requires_grad


class Enc3D(CaeBase):
    def __init__(self, size_input_xy, size_input_z, channels, n_ch_global, alpha):
        super().__init__(size_input_xy, size_input_z, channels, n_ch_global, alpha, inner_xy=10, inner_z=3)

        self.encoder = nn.Sequential(
            nn.BatchNorm3d(self.n_input),
            nn.Conv3d(self.n_input, self.n_ch_origin, 3, stride=1, padding=(1, 0, 0)),
            nn.ReLU(True),
            nn.BatchNorm3d(self.n_ch_origin),
            nn.Conv3d(self.n_ch_origin, self.n_ch_origin, 3, stride=1, padding=(1, 0, 0)),
            nn.ReLU(True),

            nn.BatchNorm3d(self.n_ch_origin),
            nn.Conv3d(self.n_ch_origin, self.n_ch_down2x, 3, stride=2, padding=1),
            nn.ReLU(True),

            nn.BatchNorm3d(self.n_ch_down2x),
            nn.Conv3d(self.n_ch_down2x, self.n_ch_down2x, 3, stride=1, padding=(1, 0, 0)),
            nn.ReLU(True),
            nn.BatchNorm3d(self.n_ch_down2x),
            nn.Conv3d(self.n_ch_down2x, self.n_ch_down2x, 3, stride=1, padding=(1, 0, 0)),
            nn.ReLU(True),

            nn.BatchNorm3d(self.n_ch_down2x),
            nn.Conv3d(self.n_ch_down2x, self.n_ch_down4x, 3, stride=2, padding=1),
            nn.ReLU(True),

            nn.BatchNorm3d(self.n_ch_down4x),
            nn.Conv3d(self.n_ch_down4x, self.n_ch_down4x, 3, stride=1, padding=(1, 0, 0)),
            nn.ReLU(True),
            nn.BatchNorm3d(self.n_ch_down4x),
            nn.Conv3d(self.n_ch_down4x, self.n_ch_down4x, 3, stride=1, padding=(1, 0, 0)),
            nn.ReLU(True),

            nn.BatchNorm3d(self.n_ch_down4x),
            nn.Conv3d(self.n_ch_down4x, self.n_ch_down8x, 3, stride=2, padding=0),
            nn.ReLU(True),

            nn.BatchNorm3d(self.n_ch_down8x),
            nn.Conv3d(self.n_ch_down8x, self.n_ch_down8x, 3, stride=1, padding=0),
            nn.ReLU(True),
        )

        self.r1 = nn.Sequential(
            nn.BatchNorm3d(self.n_ch_down8x),
            nn.Conv3d(self.n_ch_down8x, self.n_ch_down8x, (1, 2, 2), stride=(1, 2, 2), padding=0),
            nn.ReLU(True),
        )

        self.r2 = nn.Sequential(
            nn.BatchNorm3d(self.n_ch_down8x),
            nn.Conv3d(self.n_ch_down8x, self.n_ch_fc, (1, 5, 5), stride=1, padding=0),
            nn.ReLU(True),
        )

    def forward(self, input_image):
        if input_image is None:
            return None
        tmp = self.encoder(input_image)
        tmp = self.r1(tmp)
        return self.r2(tmp)


class Dec3D(CaeBase):
    def __init__(self, size_input_xy, size_input_z, channels, n_ch_global, alpha):
        super().__init__(size_input_xy, size_input_z, channels, n_ch_global, alpha, inner_xy=10, inner_z=3)

        self.decoder = nn.Sequential(
            nn.BatchNorm3d(self.n_ch_fc),
            nn.ConvTranspose3d(self.n_ch_fc, self.n_ch_down8x, (1, 5, 5), stride=1, padding=0, output_padding=0),
            nn.ELU(alpha, True),

            nn.BatchNorm3d(self.n_ch_down8x),
            nn.ConvTranspose3d(self.n_ch_down8x, self.n_ch_down8x, (1, 2, 2), stride=2, padding=0, output_padding=0),
            nn.ELU(alpha, True),

            nn.BatchNorm3d(self.n_ch_down8x),
            nn.ConvTranspose3d(self.n_ch_down8x, self.n_ch_down8x, 3, stride=1, padding=0, output_padding=0),
            nn.ELU(alpha, True),

            nn.BatchNorm3d(self.n_ch_down8x),
            nn.ConvTranspose3d(self.n_ch_down8x, self.n_ch_down4x, 3, stride=2, padding=0, output_padding=0),
            nn.ELU(alpha, True),

            nn.BatchNorm3d(self.n_ch_down4x),
            nn.Conv3d(self.n_ch_down4x, self.n_ch_down4x, 3, stride=1, padding=(1, 2, 2)),
            nn.ELU(alpha, True),
            nn.BatchNorm3d(self.n_ch_down4x),
            nn.Conv3d(self.n_ch_down4x, self.n_ch_down2x, 3, stride=1, padding=(1, 2, 2)),
            nn.ELU(alpha, True),

            nn.BatchNorm3d(self.n_ch_down2x),
            nn.ConvTranspose3d(self.n_ch_down2x, self.n_ch_down2x, 2, stride=2, padding=0, output_padding=0),
            nn.ELU(alpha, True),

            nn.BatchNorm3d(self.n_ch_down2x),
            nn.Conv3d(self.n_ch_down2x, self.n_ch_down2x, 3, stride=1, padding=(1, 2, 2)),
            nn.ELU(alpha, True),
            nn.BatchNorm3d(self.n_ch_down2x),
            nn.Conv3d(self.n_ch_down2x, self.n_ch_origin, 3, stride=1, padding=(1, 2, 2)),
            nn.ELU(alpha, True),

            nn.BatchNorm3d(self.n_ch_origin),
            nn.ConvTranspose3d(self.n_ch_origin, self.n_ch_origin, 2, stride=2, padding=0, output_padding=0),
            nn.ELU(alpha, True),

            nn.BatchNorm3d(self.n_ch_origin),
            nn.Conv3d(self.n_ch_origin, self.n_ch_origin, 3, stride=1, padding=(1, 2, 2)),
            nn.ELU(alpha, True),
            nn.BatchNorm3d(self.n_ch_origin),
            nn.Conv3d(self.n_ch_origin, self.n_ch_origin, 3, stride=1, padding=(1, 2, 2)),
            nn.ELU(alpha, True),

            nn.BatchNorm3d(self.n_ch_origin),
            nn.Conv3d(self.n_ch_origin, self.n_ch_origin, 1, stride=1, padding=0),
            nn.ELU(alpha, True),
            nn.BatchNorm3d(self.n_ch_origin),
            nn.Conv3d(self.n_ch_origin, self.n_classes, 1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, input_latent):
        if input_latent is None:
            return None
        return self.decoder(input_latent)


class Cae3dRnn(nn.Module):
    def __init__(self, enc: Enc3D, dec: Dec3D, low_dim=100, globals_dim=1, batchsize=4):
        super().__init__()
        self.enc = enc
        self.dec = dec

        self.hh = nn.Sequential(
            nn.BatchNorm3d(2 * low_dim + globals_dim),
            nn.Conv3d(2 * low_dim + globals_dim, low_dim, 1),
            nn.ReLU(True)
        )

        self.reduce2dec = nn.Sequential(
            nn.BatchNorm3d(2 * low_dim + globals_dim),
            nn.Conv3d(2 * low_dim + globals_dim, low_dim, 1),
            nn.ReLU(True)
        )

        self.tanh = nn.Tanh()

        self.low_dim = low_dim
        self.batchsize = batchsize

    def init_hstate(self):
        return torch.zeros([self.batchsize, self.low_dim, 1, 1, 1]).cuda()

    def forward_step1(self, image, globals, h_state):
        latent_image = self.enc(image)
        combined = torch.cat((latent_image, globals, h_state), dim=1)
        h_state = self.tanh(self.hh(combined))
        return combined, h_state

    def forward(self, combined_this, combined_other):
        combined = combined_this * combined_other
        output = self.dec(self.reduce2dec(combined))
        return output


channels_enc = [1, 10, 12, 16, 32, 64, 1]  #[1, 16, 20, 24, 32, 96, 1]
n_ch_global = 1
low_dim = channels_enc[5]
channels_dec = channels_enc
batchsize = 2
normalize = 24
zslice = 14

train_trafo = [data.HemisphericFlip(), data.ElasticDeform(), data.ToTensor()]
valid_trafo = [data.ToTensor()]
ds_train, ds_valid = data.get_toy_shape_training_data(train_trafo, valid_trafo,
                                                      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                                                      [16, 17, 18, 19],
                                                      batchsize=batchsize)

enc = Enc3D(size_input_xy=128, size_input_z=28, channels=channels_enc, n_ch_global=n_ch_global, alpha=0.1)
dec = Dec3D(size_input_xy=128, size_input_z=28, channels=channels_dec, n_ch_global=n_ch_global, alpha=0.1)
rnn0 = Cae3dRnn(enc, dec, low_dim, n_ch_global, batchsize=batchsize).cuda()
rnn1 = Cae3dRnn(enc, dec, low_dim, n_ch_global, batchsize=batchsize).cuda()
#rnn = torch.load('/share/data_zoe1/lucas/NOT_IN_BACKUP/tmp/rnn_217.model')

params = [p for p in rnn0.parameters() if p.requires_grad] + [p for p in rnn1.parameters() if p.requires_grad]
print('# optimizing params', sum([p.nelement() * p.requires_grad for p in params]),
      '/ total: Bi-RNN', sum([p.nelement() for p in rnn0.parameters()] + [p.nelement() for p in rnn1.parameters()]))

criterion = metrics.BatchDiceLoss([1 / 2, 1 / 2])  #metrics.BatchDiceLoss([1 / 3, 1 / 3, 1 / 3])
optimizer = torch.optim.Adam(params, lr=0.001)

for epoch in range(0, 1000):
    f, axarr = plt.subplots(4, 6)
    loss_mean = 0
    inc = 0
    rnn0.train()
    rnn1.train()

    for batch in ds_train:
        to_ta = batch[data.KEY_GLOBAL][:, 0, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5).cuda()
        ta_tr = batch[data.KEY_GLOBAL][:, 1, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5).cuda()
        time_core = Variable(to_ta.type(torch.FloatTensor)).cuda()
        time_penu = Variable(
            torch.ones(batch[data.KEY_GLOBAL].size()[0], 1, 1, 1, 1).type(torch.FloatTensor) * normalize).cuda()
        time_zero = Variable(
            torch.zeros(batch[data.KEY_GLOBAL].size()[0], 1, 1, 1, 1).type(torch.FloatTensor)).cuda()
        time_fuct = Variable((to_ta + ta_tr).type(torch.FloatTensor)).cuda()
        core_gt = Variable(batch[data.KEY_LABELS][:, 0, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5)).cuda()
        penu_gt = Variable(batch[data.KEY_LABELS][:, 1, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5)).cuda()
        fuct_gt = Variable(batch[data.KEY_LABELS][:, 2, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5)).cuda()

        h_state0 = Variable(rnn0.init_hstate()).cuda()
        h_state1 = Variable(rnn1.init_hstate()).cuda()

        init_gt = Variable(torch.zeros(core_gt.size())).cuda()

        ### 1

        core_combined0, h_state0 = rnn0.forward_step1(init_gt, time_core, h_state0)
        fuct_combined1, h_state1 = rnn1.forward_step1(penu_gt, time_fuct, h_state1)

        fuct_combined0, h_state0 = rnn0.forward_step1(core_gt, time_fuct, h_state0)
        core_combined1, h_state1 = rnn1.forward_step1(fuct_gt, time_core, h_state1)

        core_pr = rnn0(core_combined0, core_combined1)
        fuct_pr = rnn0(fuct_combined0, fuct_combined1)

        loss = criterion(torch.cat([core_pr, fuct_pr], dim=1),
                         torch.cat([core_gt, fuct_gt], dim=1))
        loss_mean += float(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        inc += 1

        del loss
        del core_combined0
        del core_combined1
        del fuct_combined0
        del fuct_combined1
        del core_pr
        del fuct_pr
        del time_core
        del time_fuct

        ### 2

        penu_combined0, _ = rnn0.forward_step1(fuct_gt, time_penu, h_state0)
        del fuct_gt
        del time_penu
        del h_state0
        penu_pr = rnn0(penu_combined0, penu_gt)
        del penu_combined0

        zero_combined1, _ = rnn1.forward_step1(core_gt, time_zero, h_state1)
        del core_gt
        del time_zero
        del h_state1
        zero_pr = rnn0(zero_combined1, init_gt)
        del zero_combined1

        loss = criterion(torch.cat([zero_pr, penu_pr], dim=1),
                         torch.cat([init_gt, penu_gt], dim=1))
        loss_mean += float(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        inc += 1

        ###

        torch.save(rnn0, '/share/data_zoe1/lucas/NOT_IN_BACKUP/tmp/rnn0.model')
        torch.save(rnn1, '/share/data_zoe1/lucas/NOT_IN_BACKUP/tmp/rnn1.model')

    for row in range(2):

        axarr[row, 0].imshow(core_gt.cpu().data.numpy()[row, 0, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        axarr[row, 1].imshow(core_pr.cpu().data.numpy()[row, 0, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        axarr[row, 2].imshow(fuct_gt.cpu().data.numpy()[row, 0, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        axarr[row, 3].imshow(fuct_pr.cpu().data.numpy()[row, 0, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        axarr[row, 4].imshow(penu_gt.cpu().data.numpy()[row, 0, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        axarr[row, 5].imshow(penu_pr.cpu().data.numpy()[row, 0, zslice, :, :], vmin=0, vmax=1, cmap='gray')

        titles = ['Core', 'p({:02.1f})'.format(float(time_core[row, :, :, :, :])),
                  'Lesion', 'p({:02.1f})'.format(float(time_fuct[row, :, :, :, :])),
                  'Penumbra', 'p({:02.1f})'.format(float(time_penu[row, :, :, :, :]))]

        for ax, title in zip(axarr[row], titles):
            ax.set_title(title)

    rnn0.eval()
    rnn1.eval()

    # in validation: PENU_PR BASED ON FUCT_PR!!!!!!
    for batch in ds_valid:
        to_ta = batch[data.KEY_GLOBAL][:, 0, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5).cuda()
        ta_tr = batch[data.KEY_GLOBAL][:, 1, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5).cuda()
        time_core = Variable(to_ta.type(torch.FloatTensor)).cuda()
        time_penu = Variable(
            torch.ones(batch[data.KEY_GLOBAL].size()[0], 1, 1, 1, 1).type(torch.FloatTensor) * normalize).cuda()
        time_zero = Variable(
            torch.zeros(batch[data.KEY_GLOBAL].size()[0], 1, 1, 1, 1).type(torch.FloatTensor)).cuda()
        time_fuct = Variable((to_ta + ta_tr).type(torch.FloatTensor)).cuda()
        core_gt = Variable(batch[data.KEY_LABELS][:, 0, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5)).cuda()
        penu_gt = Variable(batch[data.KEY_LABELS][:, 1, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5)).cuda()
        fuct_gt = Variable(batch[data.KEY_LABELS][:, 2, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5)).cuda()

        h_state0 = Variable(rnn0.init_hstate()).cuda()
        h_state1 = Variable(rnn1.init_hstate()).cuda()

        init_gt = Variable(torch.zeros(core_gt.size())).cuda()

        core_combined0, h_state0 = rnn0.forward_step1(init_gt, time_core, h_state0)
        fuct_combined1, h_state1 = rnn1.forward_step1(penu_gt, time_fuct, h_state1)

        fuct_combined0, h_state0 = rnn0.forward_step1(core_gt, time_fuct, h_state0)
        core_combined1, h_state1 = rnn1.forward_step1(fuct_gt, time_core, h_state1)

        penu_combined0, h_state0 = rnn0.forward_step1(fuct_gt, time_penu, h_state0)
        zero_combined1, h_state1 = rnn1.forward_step1(core_gt, time_zero, h_state1)

        core_pr = rnn0(core_combined0, core_combined1)
        fuct_pr = rnn0(fuct_combined0, fuct_combined1)
        penu_pr = rnn0(penu_combined0, penu_gt)

        loss = criterion(torch.cat([core_pr, fuct_pr, penu_pr], dim=1),
                         torch.cat([core_gt, fuct_gt, penu_gt], dim=1))

    for row in range(2):

        axarr[row + 2, 0].imshow(core_gt.cpu().data.numpy()[row, 0, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        axarr[row + 2, 1].imshow(core_pr.cpu().data.numpy()[row, 0, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        axarr[row + 2, 2].imshow(fuct_gt.cpu().data.numpy()[row, 0, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        axarr[row + 2, 3].imshow(fuct_pr.cpu().data.numpy()[row, 0, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        axarr[row + 2, 4].imshow(penu_gt.cpu().data.numpy()[row, 0, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        axarr[row + 2, 5].imshow(penu_pr.cpu().data.numpy()[row, 0, zslice, :, :], vmin=0, vmax=1, cmap='gray')

        titles = ['Core', 'p({:02.1f})'.format(float(time_core[row, :, :, :, :])),
                  'Lesion', 'p({:02.1f})'.format(float(time_fuct[row, :, :, :, :])),
                  'Penumbra', 'p({:02.1f})'.format(float(time_penu[row, :, :, :, :]))]

        for ax, title in zip(axarr[row + 2], titles):
            ax.set_title(title)

    for ax in axarr.flatten():
        ax.title.set_fontsize(3)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    print('Epoch', epoch, 'last batch training loss:', loss_mean / inc, '\tvalidation batch loss:', float(loss))

    f.subplots_adjust(hspace=0.05)
    f.savefig('/share/data_zoe1/lucas/NOT_IN_BACKUP/tmp/rnn' + str(epoch) + '.png', bbox_inches='tight', dpi=300)

    del f
    del axarr

