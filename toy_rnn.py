import torch
import torch.nn as nn
from torch.autograd import Variable
from common import data
from common.model.Cae3D import Enc3D, Dec3D


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


class Cae3dRnn(nn.Module):
    def __init__(self, enc: Enc3D, dec: Dec3D, low_dim=100):
        super().__init__()
        self.enc = enc
        self.dec = dec

        self.h_state = torch.zeros([-1, low_dim, 1, 1, 1])

        self.hh = nn.Sequential(
            nn.BatchNorm3d(low_dim),
            nn.Conv3d(low_dim, low_dim, 1),
            nn.ReLU(True)
        )

        self.tanh = nn.Tanh()

    def forward(self, image, globals):
        latent_image = self.enc._forward_single(image)
        latent_code = torch.cat((latent_image, globals), dim=1)
        self.h_state = self.tanh(self.hh(latent_code) + self.h_state)
        return self.dec._forward_single(self.h_state)


channels_cae = [20, 24, 32, 64, 128, 1]
batchsize = 4
normalize = 24

train_trafo = [data.HemisphericFlip(), data.ElasticDeform(), data.ToTensor()]
valid_trafo = [data.ToTensor()]
ds_train, ds_valid = data.get_toy_shape_training_data(train_trafo, valid_trafo,
                                                      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                                                      [16 ,17 ,18, 19],
                                                      batchsize=batchsize)

enc = Enc3D(size_input_xy=128, size_input_z=28, channels=channels_cae, n_ch_global=2, alpha=0.1)
dec = Dec3D(size_input_xy=128, size_input_z=28, channels=channels_cae, n_ch_global=2, alpha=0.1)
rnn = Cae3dRnn(enc, dec, 129).cuda()

params = [p for p in rnn.parameters() if p.requires_grad]
print('# optimizing params', sum([p.nelement() * p.requires_grad for p in params]),
      '/ total: RNN', sum([p.nelement() for p in rnn.parameters()]))

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(params, lr=0.001)

for epoch in range(100):
    print('Epoch', epoch)
    for batch in ds_train:
        to_ta = batch[data.KEY_GLOBAL][:, 0, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5).cuda()
        ta_tr = batch[data.KEY_GLOBAL][:, 1, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5).cuda()
        time_core = Variable(to_ta.type(torch.FloatTensor)).cuda()
        time_penu = Variable(torch.ones(batch[data.KEY_GLOBAL].size()[0], 1, 1, 1, 1).type(torch.FloatTensor) * normalize).cuda()
        time_lesion = Variable((to_ta + ta_tr).type(torch.FloatTensor)).cuda()
        core_gt = Variable(batch[data.KEY_LABELS][:, 0, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5)).cuda()
        penu_gt = Variable(batch[data.KEY_LABELS][:, 1, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5)).cuda()
        lesion_gt = Variable(batch[data.KEY_LABELS][:, 2, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5)).cuda()

        core_pred = rnn(torch.zeros(core_gt.size()), time_core)
        lesion_pred = rnn(core_pred, time_lesion)
        penu_pred = rnn(penu_pred, time_penu)

        loss = criterion(torch.cat([core_pred, lesion_pred, penu_pred], dim=1),
                         torch.cat([core_gt, lesion_gt, penu_gt], dim=1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

