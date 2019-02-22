"""
Based on:
https://github.com/jacobkimmel/pytorch_convgru
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from torch.nn.functional import affine_grid, grid_sample


def affine_identity(n=1):
    result = []
    for i in range(n):
        result += [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
    return torch.tensor(result, dtype=torch.float)


def grid_identity(batch_size=4, n=1, out_size=(4, 1, 28, 128, 128)):
    result = affine_identity(1)
    result = result.view(-1, 3, 4).expand(batch_size, 3, 4).cuda()
    return torch.cat([nn.functional.affine_grid(result, out_size) for _ in range(n)], dim=4)


def def_vec2vec(n_dim, final_activation=None, init_fn=lambda x: x):
    assert len(n_dim) > 1

    result = []
    for i in range(1, len(n_dim) - 1):
        result += [nn.Linear(n_dim[i - 1], n_dim[i]), nn.ReLU(True), nn.Dropout()]
    result += [nn.Linear(n_dim[len(n_dim) - 2], n_dim[len(n_dim) - 1])]

    if final_activation:
        if final_activation.lower() == 'relu':
            result.append(nn.ReLU())
        elif final_activation.lower() == 'sigmoid':
            result.append(nn.Sigmoid())
        elif final_activation.lower() == 'tanh':
            result.append(nn.Tanh())
        else:
            raise AssertionError('Unknown final activation function')

    return nn.Sequential(*init_fn(result))


def def_img2vec(n_dim, depth2d=False):
    assert len(n_dim) == 5
    ksize = 3
    ksize2 = (1, 3, 3)
    psize = 1
    dsize = (1, 2, 2)
    if depth2d:
        ksize = (1, ksize, ksize)
        psize = (0, psize, psize)
    return nn.Sequential(
        nn.InstanceNorm3d(n_dim[0]),  # 128x128x28
        nn.Conv3d(n_dim[0], n_dim[1], kernel_size=ksize, padding=psize),  # 128x128x28
        nn.ReLU(),
        nn.MaxPool3d(4, 4),  # 32x32x7
        nn.InstanceNorm3d(n_dim[1]),
        nn.Conv3d(n_dim[1], n_dim[2], kernel_size=ksize, padding=psize),  # 32x32x7
        nn.ReLU(),
        nn.MaxPool3d(dsize, dsize),  # 16x16x7
        nn.InstanceNorm3d(n_dim[2]),
        nn.Conv3d(n_dim[2], n_dim[3], kernel_size=ksize2),  # 14x14x7
        nn.ReLU(),
        nn.MaxPool3d(dsize, dsize),  # 7x7x7
        nn.InstanceNorm3d(n_dim[3]),
        nn.Conv3d(n_dim[3], n_dim[4], kernel_size=1),  # 7x7x7
        nn.ReLU(),
        nn.AvgPool3d(7)  # 1x1x1
    )


def time2index(time, thresholds):
    assert thresholds
    idx = 0
    while time > thresholds[idx]:
        idx += 1
    if idx > len(thresholds) :
        raise Exception('Invalid time >' + thresholds[-1])
    return idx


def tensor2index(time_tensor, thresholds):
    assert thresholds
    indices = -1 * torch.ones(time_tensor.size())
    batchsize = time_tensor.size(0)
    for b in range(batchsize):
        indices[b] = time2index(time_tensor[b], thresholds)
    return indices


class UnetBlock(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, kernel_size):
        super().__init__()

        if (isinstance(kernel_size, tuple) or isinstance(kernel_size, list)) and len(kernel_size) == 3:
            padding = (kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2)
        else:
            padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.InstanceNorm3d(input_size),
            nn.Conv3d(input_size, hidden_size, kernel_size, padding=padding),
            nn.ReLU(),
            nn.InstanceNorm3d(hidden_size),
            nn.Conv3d(hidden_size, output_size, kernel_size, padding=padding),
            nn.ReLU()
        )

    def forward(self, input):
        return self.block(input)


class Unet(nn.Module):
    def unet_def(self, h_sizes, k_sizes):
        return [UnetBlock(h_sizes[0], h_sizes[0], h_sizes[0], k_sizes[0]),
                UnetBlock(h_sizes[0], h_sizes[1], h_sizes[1], k_sizes[1]),
                UnetBlock(h_sizes[1], h_sizes[2], h_sizes[2], k_sizes[2]),

                UnetBlock(h_sizes[2], h_sizes[3], h_sizes[2], k_sizes[3]),

                UnetBlock(h_sizes[2] + h_sizes[2], h_sizes[4], h_sizes[1], k_sizes[4]),
                UnetBlock(h_sizes[1] + h_sizes[1], h_sizes[5], h_sizes[0], k_sizes[5]),
                UnetBlock(h_sizes[0] + h_sizes[0], h_sizes[6], h_sizes[0], k_sizes[6])]

    def __init__(self, hidden_sizes, kernel_sizes,):
        self.N_BLOCKS = 7

        super().__init__()

        if type(hidden_sizes) != list:
            self.hidden_sizes = [hidden_sizes] * self.N_BLOCKS
        else:
            assert len(hidden_sizes) == self.N_BLOCKS, '`hidden_sizes` must have the same length as n_layers'
            self.hidden_sizes = hidden_sizes
        if type(kernel_sizes) != list:
            self.kernel_sizes = [kernel_sizes] * self.N_BLOCKS
        else:
            assert len(kernel_sizes) == self.N_BLOCKS, '`kernel_sizes` must have the same length as n_layers'
            self.kernel_sizes = kernel_sizes

        self.blocks = self.unet_def(hidden_sizes, kernel_sizes)
        for i in range(len(self.blocks)):
            setattr(self, 'UnetBlock' + str(i).zfill(2), self.blocks[i])

        # pooling between blocks / levels
        pool = 2
        if type(kernel_sizes[0]) == tuple:
            pool = (1, 2, 2)
        self.pool = nn.MaxPool3d(pool, pool, return_indices=True)
        self.pool_anisotroph = nn.MaxPool3d((1, 2, 2), (1, 2, 2), return_indices=True)
        self.unpool_anisotroph = nn.MaxUnpool3d((1, 2, 2), (1, 2, 2))
        self.unpool = nn.MaxUnpool3d(pool, pool)

    def forward(self, input_rep):

        output0 = self.blocks[0](input_rep)
        input_rep, indices0 = self.pool(output0)

        output1 = self.blocks[1](input_rep)
        input_rep, indices1 = self.pool(output1)

        output2 = self.blocks[2](input_rep)
        input_rep, indices2 = self.pool_anisotroph(output2)

        output = self.blocks[self.N_BLOCKS // 2](input_rep)

        unpool = self.unpool_anisotroph(output, indices2)
        input_rep = torch.cat((unpool, output2), dim=1)
        output = self.blocks[4](input_rep)

        unpool = self.unpool(output, indices1)
        input_rep = torch.cat((unpool, output1), dim=1)
        output = self.blocks[5](input_rep)

        unpool = self.unpool(output, indices0)
        input_rep = torch.cat((unpool, output0), dim=1)
        output = self.blocks[6](input_rep)

        del unpool
        del input_rep
        del output0
        del output1
        del output2
        del indices0
        del indices1
        del indices2

        return output


class ConvGRU(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()

        # Allow for anisotropic inputs
        if (isinstance(kernel_size, tuple) or isinstance(kernel_size, list)) and len(kernel_size) == 3:
            self.padding = (kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2)
        else:
            self.padding = kernel_size // 2

        self.hidden_size = hidden_size
        self.reset_gate = nn.Conv3d(input_size + hidden_size, hidden_size, kernel_size, padding=self.padding)
        self.update_gate = nn.Conv3d(input_size + hidden_size, hidden_size, kernel_size, padding=self.padding)
        self.out_gate = nn.Conv3d(input_size + hidden_size, hidden_size, kernel_size, padding=self.padding)

        # Appropriate initialization
        nn.init.orthogonal_(self.reset_gate.weight)
        nn.init.orthogonal_(self.update_gate.weight)
        nn.init.orthogonal_(self.out_gate.weight)
        nn.init.constant_(self.reset_gate.bias, 0.)
        nn.init.constant_(self.update_gate.bias, 0.)
        nn.init.constant_(self.out_gate.bias, 0.)

    def forward(self, input, prev_state):
        # Get batch and spatial sizes
        batch_size = input.data.size()[0]
        spatial_size = input.data.size()[2:]

        # Generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if torch.cuda.is_available():
                prev_state = torch.zeros(state_size).cuda()
            else:
                prev_state = torch.zeros(state_size)

        # Data size: [batch, channel, depth, height, width]
        stacked_inputs = torch.cat([input, prev_state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = torch.tanh(self.out_gate(torch.cat([input, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state


class GRUnetBlock(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, output_size=None, output_activation=True):
        super().__init__()

        # Allow for anisotropic inputs
        if (isinstance(kernel_size, tuple) or isinstance(kernel_size, list)) and len(kernel_size) == 3:
            padding = (kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2)
        else:
            padding = kernel_size // 2
        self.input_size = input_size

        # GRU convolution with incorporation of hidden state
        self.hidden_size = hidden_size
        self.reset_gate = nn.Conv3d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv3d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = nn.Conv3d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)

        # Additional "normal" convolution as in vanilla Unet to map to another channel number
        if output_size is None:
            output_size = hidden_size
        if output_activation:
            self.conv3d = nn.Sequential(
                nn.Conv3d(hidden_size, output_size, kernel_size, padding=padding),
                nn.ReLU()
            )
            nn.init.xavier_normal(self.conv3d[0].weight)
            nn.init.normal(self.conv3d[0].bias)
        else:
            self.conv3d = nn.Conv3d(hidden_size, output_size, kernel_size, padding=padding)
            nn.init.xavier_normal(self.conv3d.weight)
            nn.init.normal(self.conv3d.bias)

        # Appropriate initialization
        nn.init.orthogonal_(self.reset_gate.weight)
        nn.init.orthogonal_(self.update_gate.weight)
        nn.init.orthogonal_(self.out_gate.weight)
        nn.init.constant_(self.reset_gate.bias, 0.)
        nn.init.constant_(self.update_gate.bias, 0.)
        nn.init.constant_(self.out_gate.bias, 0.)

    def forward(self, input_, prev_state):
        # Get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # Generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if torch.cuda.is_available():
                prev_state = torch.zeros(state_size).cuda()
            else:
                prev_state = torch.zeros(state_size)

        # Data size: [batch, channel, depth, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = torch.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update

        output = self.conv3d(new_state)

        return new_state, output


class GRUnet(nn.Module):
    def grunet_def(self, h_sizes, k_sizes):
        return [GRUnetBlock(h_sizes[0], h_sizes[0], k_sizes[0], output_size=h_sizes[0]),
                GRUnetBlock(h_sizes[0], h_sizes[1], k_sizes[1], output_size=h_sizes[1]),
                GRUnetBlock(h_sizes[1], h_sizes[2], k_sizes[2], output_size=h_sizes[1]),
                GRUnetBlock(h_sizes[1] + h_sizes[1], h_sizes[3], k_sizes[3], output_size=h_sizes[0]),
                GRUnetBlock(h_sizes[0] + h_sizes[0], h_sizes[4], k_sizes[4], output_size=h_sizes[0])]

    def __init__(self, hidden_sizes, kernel_sizes, down_scaling):
        self.N_BLOCKS = 5

        super().__init__()

        if type(hidden_sizes) != list:
            self.hidden_sizes = [hidden_sizes] * self.N_BLOCKS
        else:
            assert len(hidden_sizes) == self.N_BLOCKS, '`hidden_sizes` must have the same length as n_layers'
            self.hidden_sizes = hidden_sizes
        if type(kernel_sizes) != list:
            self.kernel_sizes = [kernel_sizes] * self.N_BLOCKS
        else:
            assert len(kernel_sizes) == self.N_BLOCKS, '`kernel_sizes` must have the same length as n_layers'
            self.kernel_sizes = kernel_sizes

        self.blocks = self.grunet_def(hidden_sizes, kernel_sizes)
        for i in range(len(self.blocks)):
            setattr(self, 'GRUnetBlock' + str(i).zfill(2), self.blocks[i])

        # pooling between blocks / levels
        pool = 2
        if type(kernel_sizes[0]) == tuple:
            pool = (1, 2, 2)
        self.pool = nn.MaxPool3d(pool, pool, return_indices=True)
        self.unpool = nn.MaxUnpool3d(pool, pool)

        # Grid offset prediction
        self.grid_offset = nn.Conv3d(self.blocks[-1].conv3d.out_channels, 6, 1)
        torch.nn.init.normal(self.grid_offset.weight, 0, 0.001)
        torch.nn.init.normal(self.grid_offset.bias, 0, 0.001)

        self.downscaling = down_scaling

    def forward(self, input_rep, *hidden):
        input_rep = F.interpolate(input_rep, scale_factor=(1, 1/self.downscaling, 1/self.downscaling))

        outputs = [None] * (self.N_BLOCKS // 2 + 1)
        indices = [None] * (self.N_BLOCKS // 2)
        upd_hidden = [None] * self.N_BLOCKS
        if len(hidden) == 1 and hidden[0] is None:
            hidden = [None] * (self.N_BLOCKS + 2)

        #
        # Non-lin deform

        for i in range(self.N_BLOCKS // 2):
           # upd_block_hidden, output = checkpoint(lambda a, b: self.blocks[i](a, b), input_rep, hidden[i])
            upd_block_hidden, output = self.blocks[i](input_rep, hidden[i])
            upd_hidden[i] = upd_block_hidden
            outputs[i] = output

            input_rep, indices_ = self.pool(output)
            indices[i] = indices_
        del indices_

        upd_block_hidden, output = self.blocks[self.N_BLOCKS // 2](input_rep, hidden[self.N_BLOCKS // 2])
        upd_hidden[self.N_BLOCKS // 2] = upd_block_hidden
        outputs[self.N_BLOCKS // 2] = output

        for i in range(self.N_BLOCKS // 2):
            unpool = self.unpool(output, indices[self.N_BLOCKS // 2 - (i + 1)])
            skip = outputs[self.N_BLOCKS // 2 - (i + 1)]
            input_rep = torch.cat((unpool, skip), dim=1)

            j = self.N_BLOCKS // 2 + (i + 1)
            upd_block_hidden, output = self.blocks[j](input_rep, hidden[j])
            upd_hidden[j] = upd_block_hidden

        del upd_block_hidden
        del outputs
        del input_rep
        del unpool
        del skip

        output = F.interpolate(self.grid_offset(output), scale_factor=(1, self.downscaling, self.downscaling))

        return output.permute(0, 2, 3, 4, 1), upd_hidden[0], upd_hidden[1], upd_hidden[2], upd_hidden[3], upd_hidden[4]


class AffineModule(nn.Module):
    def init_theta(self, pos):
        def _init(sequential):
            sequential[pos].weight.data.zero_()
            sequential[pos].bias.data.copy_(affine_identity(2))
            return sequential

        return _init

    def __init__(self, dim_img2vec, dim_vec2vec, dim_clinical, kernel_size, seq_len, depth2d=False):
        super().__init__()

        dim_in_img = dim_img2vec[0]
        dim_hidden = dim_img2vec[-1] + dim_clinical
        assert dim_vec2vec[0] == dim_hidden
        assert dim_vec2vec[-1] == 24  # core and penumbra affine parameters

        self.len = seq_len

        self.affine1 = GRUnetBlock(dim_in_img, dim_in_img, kernel_size)
        self.affine2 = def_img2vec(n_dim=dim_img2vec, depth2d=depth2d)
        self.affine3 = nn.GRUCell(dim_hidden, dim_hidden, bias=True)
        self.affine4 = def_vec2vec(n_dim=dim_vec2vec, init_fn=self.init_theta(-1))
        self.affine5 = nn.GRUCell(24, 24, bias=True)

    def forward(self, input_img, clinical, core, hidden_affine1, hidden_affine3, hidden_affine5):
        out_size = core.size()
        del core
        hidden_affine1, affine1 = self.affine1(input_img, hidden_affine1)
        del input_img
        affine2 = self.affine2(affine1)
        hidden_affine3 = self.affine3(torch.cat((affine2, clinical), dim=1).squeeze(), hidden_affine3)
        del clinical
        affine4 = self.affine4(hidden_affine3)
        hidden_affine5 = self.affine5(affine4, hidden_affine5)
        grid_core = nn.functional.affine_grid(hidden_affine5[:, :12].view(-1, 3, 4), out_size)
        grid_penu = nn.functional.affine_grid(hidden_affine5[:, 12:].view(-1, 3, 4), out_size)

        return torch.cat((grid_core, grid_penu), dim=4), hidden_affine1, hidden_affine3, hidden_affine5


class LesionPositionModule(nn.Module):
    def __init__(self, dim_img2vec, dim_vec2vec, dim_clinical, kernel_size, seq_len, depth2d=False):
        super().__init__()

        dim_in_img = dim_img2vec[0]
        dim_hidden = dim_img2vec[-1] + dim_clinical
        assert dim_vec2vec[0] == dim_hidden

        self.len = seq_len

        self.affine1 = GRUnetBlock(dim_in_img, dim_in_img, kernel_size)
        self.affine2 = def_img2vec(n_dim=dim_img2vec, depth2d=depth2d)
        self.affine3 = nn.GRUCell(dim_hidden, dim_hidden, bias=True)
        self.affine4 = def_vec2vec(n_dim=dim_vec2vec, final_activation='sigmoid')

        torch.nn.init.normal(self.affine4[-2].weight, 0, 0.001)
        torch.nn.init.normal(self.affine4[-2].bias, 0, 0.1)


    def forward(self, input_img, clinical, hidden_affine1, hidden_affine3, hidden_affine5):

        hidden_affine1, affine1 = self.affine1(input_img, hidden_affine1)
        affine2 = self.affine2(affine1)
        hidden_affine3 = self.affine3(torch.cat((affine2, clinical), dim=1).squeeze(), hidden_affine3)
        affine4 = self.affine4(hidden_affine3)

        return affine4, hidden_affine1, hidden_affine3, hidden_affine5


class UnidirectionalSequence(nn.Module):
    def _init_zero_normal(self, pos):
        def _init(sequential):
            sequential[pos].weight.data.normal_(mean=0, std=0.001)  # for Sigmoid()=0.5 init
            sequential[pos].bias.data.normal_(mean=0, std=0.1)  # for Sigmoid()=0.5 init
            return sequential

        return _init

    def __init__(self, n_ch_grunet, dim_img2vec_affine, dim_vec2vec_affine, dim_img2vec_time, dim_vec2vec_time,
                 dim_clinical, dim_feat_rnn, kernel_size, seq_len, batchsize=4, out_size=6, depth2d=False,
                 reverse=False, add_factor=False, clinical_grunet=False):
        super().__init__()

        self.len = seq_len
        self.batchsize = batchsize
        self.out_size = out_size
        self.reverse = reverse
        self.add_factor = add_factor
        self.clinical_grunet = clinical_grunet

        self.grid_identity = grid_identity(batchsize, out_size=(batchsize, out_size, 28, 128, 128))

        #
        # Separate (hidden) features for core / penumbra
        self.core_rep = ConvGRU(dim_feat_rnn, dim_feat_rnn, kernel_size)
        self.penu_rep = ConvGRU(dim_feat_rnn, dim_feat_rnn, kernel_size)

        #
        # Affine
        self.affine = None
        if dim_img2vec_affine and dim_vec2vec_affine:
            assert dim_img2vec_affine[0] == 2 * dim_feat_rnn + 4, '{} != 2 * {} + 4'.format(int(dim_img2vec_affine[0]), int(dim_feat_rnn))  # ... + 4 = ... + 2 core/penumbra + 2 previous deform
            assert dim_img2vec_affine[-1] + dim_clinical == dim_vec2vec_affine[0]
            self.affine = AffineModule(dim_img2vec_affine, dim_vec2vec_affine, dim_clinical, kernel_size, seq_len, depth2d=depth2d)

        #
        # Non-lin.
        self.grunet = None
        if n_ch_grunet:
            self.xy_downscaling = 4

            self.refine1 = GRUnetBlock(n_ch_grunet[0], 20, kernel_size)
            assert n_ch_grunet[0] < self.refine1.hidden_size
            self.maxpool = nn.MaxPool3d(2, 2, return_indices=True)
            self.refine2 = GRUnetBlock(20, 30, kernel_size, output_size=20)
            self.munpool = nn.MaxUnpool3d(2, 2)
            self.refine3 = GRUnetBlock(20, 20, kernel_size)
            self.grid_offset = GRUnetBlock(20, 10, kernel_size, output_size=6, output_activation=False)

            '''
            self.unet_in = GRUnetBlock(n_ch_grunet[0], n_ch_grunet[0], kernel_size)

            self.grunet = Unet(hidden_sizes=n_ch_grunet, kernel_sizes=[kernel_size] * 7)
            # self.grunet = GRUnet(hidden_sizes=n_ch_grunet, kernel_sizes=[kernel_size] * 5, down_scaling=2)

            self.grid_offset = GRUnetBlock(n_ch_grunet[-1], n_ch_grunet[-1], kernel_size, output_size=6, output_activation=False)
            '''
            torch.nn.init.normal(self.grid_offset.conv3d.weight, 0, 0.001)
            torch.nn.init.normal(self.grid_offset.conv3d.bias, 0, 0.001)

        assert self.grunet or self.affine, 'Either affine or non-lin. deformation parameter numbers must be given'

        #
        # Time position
        self.lesion_pos = None
        if dim_img2vec_time and dim_vec2vec_time:
            assert dim_img2vec_time[-1] + dim_clinical == dim_vec2vec_time[0]
            self.lesion_pos = LesionPositionModule(dim_img2vec_time, dim_vec2vec_time, dim_clinical, kernel_size,
                                                   seq_len, depth2d=depth2d)

    def forward(self, core, penu, core_rep, penu_rep, clinical, factor):
        def _nonlin(input_grunet, hidden_refine1, hidden_refine2, hidden_refine3, hidden_unet_out):
            hidden_refine1, output = self.refine1(input_grunet, hidden_refine1)
            output, indices = self.maxpool(output)
            hidden_refine2, output = self.refine2(output, hidden_refine2)
            output = self.munpool(output, indices)
            hidden_refine3, output = self.refine3(output, hidden_refine3)
            hidden_unet_out, nonlin_grids = self.grid_offset(output, hidden_unet_out)
            return nonlin_grids, hidden_refine1, hidden_refine2, hidden_refine3, hidden_unet_out

        offset = []
        pr_time = []

        if self.reverse:
            factor = 1 - factor

        hidden_core = torch.zeros(self.batchsize, self.core_rep.hidden_size, 28, 128, 128).cuda()
        hidden_penu = torch.zeros(self.batchsize, self.penu_rep.hidden_size, 28, 128, 128).cuda()

        if self.munpool is not None:  #self.grunet:
            #hidden_unet_in = torch.zeros(self.batchsize, self.unet_in.hidden_size, 28, 64, 64).cuda()
            hidden_refine1 = torch.zeros(self.batchsize, self.refine1.hidden_size, 28, 32, 32).cuda()
            hidden_refine2 = torch.zeros(self.batchsize, self.refine2.hidden_size, 14, 16, 16).cuda()
            hidden_refine3 = torch.zeros(self.batchsize, self.refine3.hidden_size, 28, 32, 32).cuda()
            hidden_unet_out = torch.zeros(self.batchsize, self.grid_offset.hidden_size, 28, 32, 32).cuda()
            '''
            hidden_grunet = [torch.zeros([self.batchsize, self.grunet.blocks[0].hidden_size, 28, 64, 64]).cuda(),
                             torch.zeros([self.batchsize, self.grunet.blocks[1].hidden_size, 14, 32, 32]).cuda(),
                             torch.zeros([self.batchsize, self.grunet.blocks[2].hidden_size, 7, 16, 16]).cuda(),
                             torch.zeros([self.batchsize, self.grunet.blocks[3].hidden_size, 14, 32, 32]).cuda(),
                             torch.zeros([self.batchsize, self.grunet.blocks[4].hidden_size, 28, 64, 64]).cuda()]
            '''
        if self.affine:
            h_affine1 = torch.zeros(self.batchsize, self.affine.affine1.hidden_size, 28, 128, 128).cuda()
            h_affine3 = torch.zeros((self.batchsize, self.affine.affine3.hidden_size)).cuda()
            h_affine5 = torch.zeros((self.batchsize, 24)).cuda()
        if self.lesion_pos:
            h_time1 = None
            h_time3 = torch.zeros((self.batchsize, self.lesion_pos.affine3.hidden_size)).cuda()
            h_time5 = torch.zeros((self.batchsize, 1)).cuda()

        for i in range(self.len):
            if i == 0:
                if self.reverse:
                    previous_result = torch.cat((penu, penu), dim=1)
                else:
                    #previous_result = torch.cat((core, core), dim=1)
                    previous_result = torch.cat((torch.zeros(core.size()), torch.zeros(core.size())), dim=1).cuda()
            else:
                previous_result = torch.cat((nn.functional.grid_sample(core, self.grid_identity + offset[-1][:, :, :, :, :3]),
                                             nn.functional.grid_sample(penu, self.grid_identity + offset[-1][:, :, :, :, 3:])), dim=1)

            if self.add_factor:
                clinical_step = torch.cat((clinical, factor[:, i]), dim=1)
            else:
                clinical_step = clinical

            hidden_core = checkpoint(lambda a, b: self.core_rep(a, b), core_rep, hidden_core)
            hidden_penu = checkpoint(lambda a, b: self.penu_rep(a, b), penu_rep, hidden_penu)
            input_img = torch.cat((hidden_core, hidden_penu, core, penu, previous_result), dim=1)

            if self.affine:
                affine_grids, h_affine1, h_affine3, h_affine5 = checkpoint(lambda a, b, c, d, e, f: self.affine(a, b, c, d, e, f), input_img, clinical_step, core, h_affine1, h_affine3, h_affine5)
                input_grunet = torch.cat((input_img, affine_grids.permute(0, 4, 1, 2, 3)), dim=1)
            else:
                input_grunet = input_img

            if self.munpool is not None:  #self.grunet:
                if self.clinical_grunet:
                    input_grunet = torch.cat((F.interpolate(clinical_step, input_grunet.size()[2:5]), input_grunet), dim=1)
                #nonlin_grids, h0, h1, h2, h3, h4 = checkpoint(lambda a, b, c, d, e, f: self.grunet(a, b, c, d, e, f), input_grunet, *hidden_grunet)
                #hidden_grunet = [h0, h1, h2, h3, h4]
                input_grunet = F.interpolate(input_grunet, scale_factor=(1, 1 / self.xy_downscaling, 1 / self.xy_downscaling))

                '''
                hidden_refine1, output = checkpoint(lambda a, b: self.refine1(a, b), input_grunet, hidden_refine1)
                output, indices = self.maxpool(output)
                hidden_refine2, output = checkpoint(lambda a, b: self.refine2(a, b), output, hidden_refine2)
                output = self.munpool(output, indices)
                hidden_refine3, output = checkpoint(lambda a, b: self.refine3(a, b), output, hidden_refine3)
                hidden_unet_out, nonlin_grids = checkpoint(lambda a, b: self.grid_offset(a, b), output, hidden_unet_out)
                '''
                nonlin_grids, hidden_refine1, hidden_refine2, hidden_refine3, hidden_unet_out = checkpoint(lambda a, b, c, d, e: _nonlin(a, b, c, d, e), input_grunet, hidden_refine1, hidden_refine2, hidden_refine3, hidden_unet_out)

                '''
                hidden_unet_in, input_grunet = checkpoint(lambda a, b: self.unet_in(a, b), input_grunet, hidden_unet_in)
                nonlin_grids = checkpoint(lambda a: self.grunet(a), input_grunet)
                hidden_unet_out, nonlin_grids = checkpoint(lambda a, b: self.grid_offset(a, b), output, hidden_unet_out)
                '''

                nonlin_grids = F.interpolate(nonlin_grids, scale_factor=(1, self.xy_downscaling, self.xy_downscaling))
                offset.append(nonlin_grids.permute(0, 2, 3, 4, 1))
            else:
                offset.append(affine_grids)

            if self.lesion_pos:
                lesion_pos, h_time1, h_time3, h_time5 = self.lesion_pos(
                    torch.cat((input_img, offset[-1].permute(0, 4, 1, 2, 3)), dim=1),
                    clinical_step,
                    h_time1,
                    h_time3,
                    h_time5
                )
                pr_time.append(lesion_pos)

        return offset, pr_time  # lesion cannot be first or last!


class BidirectionalSequence(nn.Module):
    def common_rep(self, n_in, n_out, k_size=(1,3,3), p_size=(0,1,1)):
        return nn.Sequential(
            nn.InstanceNorm3d(n_in),
            nn.Conv3d(n_in, n_out, kernel_size=k_size, padding=p_size),
            nn.ReLU(),
            nn.InstanceNorm3d(n_out),
            nn.Conv3d(n_out, n_out, kernel_size=k_size, padding=p_size),
            nn.ReLU()
        )

    def visualise_grid(self, batchsize):
        visual_grid = torch.ones(batchsize, 1, 28, 128, 128, requires_grad=False).cuda()
        visual_grid[:, :, 1::4, :, :] = 0.75
        visual_grid[:, :, 2::4, :, :] = 0.5
        visual_grid[:, :, 3::4, :, :] = 0.75
        visual_grid[:, :, :, 3::24, :] = 0
        visual_grid[:, :, :, 4::24, :] = 0
        visual_grid[:, :, :, 5::24, :] = 0
        visual_grid[:, :, :, :, 3::24] = 0
        visual_grid[:, :, :, :, 4::24] = 0
        visual_grid[:, :, :, :, 5::24] = 0
        return visual_grid

    def __init__(self, n_ch_feature_single, n_ch_affine_img2vec, n_ch_affine_vec2vec, dim_img2vec_time,
                 dim_vec2vec_time, n_ch_grunet, n_ch_clinical, kernel_size, seq_len, batch_size=4, out_size=6,
                 depth2d=False, add_factor=False, soften_kernel=(3, 13, 13), clinical_grunet=False):
        super().__init__()
        self.len = seq_len
        assert seq_len > 0

        self.add_factor = add_factor

        self.grid_identity = grid_identity(batch_size, out_size=(batch_size, out_size, 28, 128, 128))

        self.visual_grid = self.visualise_grid(batch_size)

        ##############################################################
        # Part 1: Commonly used separate core/penumbra representations
        self.common_core = self.common_rep(1, n_ch_feature_single)
        self.common_penu = self.common_rep(1, n_ch_feature_single)

        ##################################
        # Part 2: Bidirectional Recurrence
        self.rnn1 = UnidirectionalSequence(n_ch_grunet, n_ch_affine_img2vec, n_ch_affine_vec2vec, dim_img2vec_time,
                                           dim_vec2vec_time, n_ch_clinical, n_ch_feature_single, kernel_size, seq_len,
                                           batchsize=batch_size, out_size=out_size, depth2d=depth2d,
                                           add_factor=add_factor, clinical_grunet=clinical_grunet)
        self.rnn2 = UnidirectionalSequence(n_ch_grunet, n_ch_affine_img2vec, n_ch_affine_vec2vec, dim_img2vec_time,
                                           dim_vec2vec_time, n_ch_clinical, n_ch_feature_single, kernel_size, seq_len,
                                           batchsize=batch_size, out_size=out_size, depth2d=depth2d, reverse=True,
                                           add_factor=add_factor, clinical_grunet=clinical_grunet)

        ################################################
        # Part 3: Combine predictions of both directions
        if len(soften_kernel) < 1:
            soften_kernel = [soften_kernel] * 3
        if depth2d:
            soften_kernel[2] = 1

        self.soften = nn.AvgPool3d(soften_kernel, (1, 1, 1), padding=[i//2 for i in soften_kernel])

        self.combine = GRUnetBlock(4, 10, kernel_size, output_activation=None, output_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, core, penu, clinical, factor):
        hidden_combine = torch.zeros(self.rnn1.batchsize, self.combine.hidden_size, 28, 128, 128).cuda()

        factor = factor.unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5)  # one additional dim for later when squeeze

        ##############################################################
        # Part 1: Commonly used separate core/penumbra representations
        core_rep = self.common_core(core)
        penu_rep = self.common_penu(penu)

        ##################################
        # Part 2: Bidirectional Recurrence
        offset1, lesion_pos1 = self.rnn1(core, penu, core_rep, penu_rep, clinical, factor)
        offset2, lesion_pos2 = self.rnn2(core, penu, core_rep, penu_rep, clinical, factor)

        ################################################
        # Part 3: Combine predictions of both directions
        offsets = [factor[:, i] * offset1[i] + (1 - factor[:, i]) * offset2[self.len - i - 1] for i in range(self.len)]
        if lesion_pos1 and lesion_pos2:
            lesion_pos = [factor[:, i].squeeze() * lesion_pos1[i].squeeze()
                          + (1 - factor[:, i]).squeeze() * lesion_pos2[self.len - i - 1].squeeze() for i in range(self.len)]
            lesion_pos = torch.stack(lesion_pos, dim=1)
        else:
            lesion_pos = None
        del offset1
        del offset2
        del lesion_pos1
        del lesion_pos2

        output_by_core = []
        output_by_penu = []
        grids_by_core = []
        grids_by_penu = []
        offsets_core = []
        offsets_penu = []
        combined = []

        for i in range(self.len):
            offsets[i] = self.soften(self.soften(offsets[i].permute(0, 4, 1, 2, 3))).permute(0, 2, 3, 4, 1)
            offsets_core.append(offsets[i][:, :, :, :, :3])
            offsets_penu.append(offsets[i][:, :, :, :, 3:])

            pred_by_core = nn.functional.grid_sample(core, self.grid_identity + offsets_core[-1])
            pred_by_penu = nn.functional.grid_sample(penu, self.grid_identity + offsets_penu[-1])
            output_by_core.append(pred_by_core)
            output_by_penu.append(pred_by_penu)
            input_combine = torch.cat((F.interpolate(clinical, pred_by_core.size()[2:5]), pred_by_core, pred_by_penu), dim=1)
            hidden_combine, combine = self.combine(input_combine, hidden_combine)
            combined.append(self.sigmoid(combine))
            del pred_by_core
            del pred_by_penu

            grid_by_core = nn.functional.grid_sample(self.visual_grid, self.grid_identity + offsets_core[-1])
            grid_by_penu = nn.functional.grid_sample(self.visual_grid, self.grid_identity + offsets_penu[-1])
            grids_by_core.append(grid_by_core)
            grids_by_penu.append(grid_by_penu)
            del grid_by_core
            del grid_by_penu

        return torch.cat(output_by_core, dim=1), torch.cat(output_by_penu, dim=1), lesion_pos,\
               torch.cat(grids_by_core, dim=1), torch.cat(grids_by_penu, dim=1),\
               torch.stack(offsets_core, dim=1), torch.stack(offsets_penu, dim=1),\
               torch.cat(combined, dim=1)


class BiNet(nn.Module):
    def _feature(self, channels):
        assert len(channels) == 3
        return nn.Sequential(
            nn.Conv3d(channels[0], channels[1], kernel_size=(3, 3, 3), dilation=(1, 7, 7), padding=(0, 0, 0)),  # 26x114x114
            nn.ReLU(),
            nn.LayerNorm([channels[1], 26, 114, 114]),
            nn.Conv3d(channels[1], channels[2], kernel_size=(3, 3, 3), dilation=(1, 5, 5), padding=(0, 0, 0)),  # 24x104x104
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))  # 12x52x52
        )

    def _refine(self, channels):
        assert len(channels) == 4
        assert channels[-1] == 3
        return nn.Sequential(
            nn.LayerNorm([channels[0], 12, 52, 52]),
            nn.Conv3d(channels[0], channels[1], kernel_size=(3, 3, 3), dilation=(1, 4, 4), padding=(1, 4, 4)),  # 12x52x52
            nn.ReLU(),
            nn.LayerNorm([channels[1], 12, 52, 52]),
            nn.Conv3d(channels[1], channels[2], kernel_size=(3, 3, 3), dilation=(1, 2, 2), padding=(1, 2, 2)),  # 12x52x52
            nn.ReLU(),
            nn.LayerNorm([channels[2], 12, 52, 52]),
            nn.Conv3d(channels[2], channels[3], kernel_size=(3, 3, 3), padding=(1, 1, 1)),  # 12x52x52
            nn.Tanh()
        )

    def _img2vec(self, channels):
        assert len(channels) == 4
        return nn.Sequential(
            nn.LayerNorm([channels[0], 12, 52, 52]),
            nn.Conv3d(channels[0], channels[1], kernel_size=(3, 3, 3), padding=(0, 0, 0)),  # 10x50x50
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),  # 5x25x25
            nn.LayerNorm([channels[1], 5, 25, 25]),
            nn.Conv3d(channels[1], channels[2], kernel_size=(3, 3, 3), padding=(0, 0, 0)),  # 3x21x21
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(3, 3, 3)),  # 1x7x7
            nn.LayerNorm([channels[2], 1, 7, 7]),
            nn.Conv3d(channels[2], channels[3], kernel_size=(1, 3, 3), padding=(0, 0, 0)),  # 1x5x5
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 5, 5), stride=(1, 5, 5)),  # 1x1x1
        )

    def _vec2vec(self, channels, dropout=0.5, final_activation=None, init_fn=lambda x: x):
        assert len(channels) > 1

        result = []
        for i in range(1, len(channels) - 1):
            result += [nn.Linear(channels[i - 1], channels[i]), nn.ReLU(True), nn.Dropout(dropout)]
        result += [nn.Linear(channels[len(channels) - 2], channels[len(channels) - 1])]

        if final_activation:
            if final_activation.lower() == 'relu':
                result.append(nn.ReLU())
            elif final_activation.lower() == 'sigmoid':
                result.append(nn.Sigmoid())
            elif final_activation.lower() == 'tanh':
                result.append(nn.Tanh())
            else:
                raise AssertionError('Unknown final activation function')

        return nn.Sequential(*init_fn(result))

    def _affine_identity(self):
        return torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float)

    def _affine_inverse(self, m):
        assert m.size(1) == 3 and m.size(2) == 4
        homogeneous = torch.cat((m, torch.tensor([0, 0, 0, 1] * 2).view(-1, 1, 4).float()), dim=1)
        homogeneous_inverse = torch.inverse(homogeneous)
        return homogeneous_inverse[:, :3, :]

    def _grid_identity(self, batch_size=4, out_size=(4, 1, 28, 128, 128)):
        assert batch_size == out_size[0]
        result = self._affine_identity()
        result = result.view(-1, 3, 4).expand(batch_size, 3, 4).cuda()
        return nn.functional.affine_grid(result, out_size)

    def visualise_grid(self, batchsize):
        visual_grid = torch.ones(batchsize, 1, 28, 128, 128, requires_grad=False).cuda()
        visual_grid[:, :, 1::4, :, :] = 0.75
        visual_grid[:, :, 2::4, :, :] = 0.5
        visual_grid[:, :, 3::4, :, :] = 0.75
        visual_grid[:, :, :, 3::24, :] = 0
        visual_grid[:, :, :, 4::24, :] = 0
        visual_grid[:, :, :, 5::24, :] = 0
        visual_grid[:, :, :, :, 3::24] = 0
        visual_grid[:, :, :, :, 4::24] = 0
        visual_grid[:, :, :, :, 5::24] = 0
        return visual_grid

    def _init_xavier(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal(m.weight)
            torch.nn.init.xavier_normal(m.bias)

    def _init_normal(self, m):
        if type(m) == nn.Conv3d:
            m.weight.data.normal_(0, 0.000001)
            m.bias.data.normal_(0, 0.000001)

    def __init__(self, seq_thr, batch_size):
        super().__init__()

        self.seq_thr = seq_thr
        self.len = len(seq_thr)
        assert self.len > 0

        self.grid_identity_core = self._grid_identity(batch_size, out_size=(batch_size, 3, 28, 128, 128))
        self.grid_identity_penu = self._grid_identity(batch_size, out_size=(batch_size, 3, 28, 128, 128))

        self.visual_grid = self.visualise_grid(batch_size)

        channels_clinical = 4  # 5 if add time_step
        channels_img_low = 42
        channels_gru = 48  # ideally >= channels_img_low + 5 ?
        channels_feature = [2, 16, 32]
        channels_img2vec = [32, 48, 64, 72]
        channels_vec2low = [72, (72 + channels_img_low)//2, channels_img_low]
        channels_low2gru = [channels_img_low + channels_clinical, channels_gru]
        channels_gru2phi = [channels_gru + channels_img_low, (channels_gru + channels_img_low + 12)//2, 12]
        channels_prl2vec = [1, channels_img_low//3, 2*(channels_img_low//3), channels_img_low]
        channels_vec2gru = [2 * channels_img_low + channels_clinical, channels_gru]  # +5 if add time_step

        self.feature_core = self._feature(channels_feature)      # spatial image features
        self.img2vec_core = self._img2vec(channels_img2vec)      # vector  representation of image features
        self.vec2low_core = self._vec2vec(channels_vec2low)      # low-dim representation of image features
        self.low2gru_core = self._vec2vec(channels_low2gru)      # low-dim representation of image features + clinical
        self.gru_cl = nn.GRUCell(channels_gru, channels_gru)     # recurrent abstraction
        self.gru_lp = nn.GRUCell(channels_gru, channels_gru)     # recurrent abstraction
        self.gru2phi_cl = self._vec2vec(channels_gru2phi)        # self.gru2aff_core / recurrent abstraction + image vector to affine params
        self.gru2phi_lp = self._vec2vec(channels_gru2phi)        #
        self.gru2phi_cl[-1].weight.data.normal_(0, 0.0001)
        self.gru2phi_cl[-1].bias.data.copy_(self._affine_identity())
        self.gru2phi_lp[-1].weight.data.normal_(0, 0.0001)
        self.gru2phi_lp[-1].bias.data.copy_(self._affine_identity())
        self.prl2vec_core = self._img2vec(channels_prl2vec)
        self.vec2gru_core = self._vec2vec(channels_vec2gru)

        self.feature_penu = self._feature(channels_feature)      # spatial image features
        self.img2vec_penu = self._img2vec(channels_img2vec)      # vector  representation of image features
        self.vec2low_penu = self._vec2vec(channels_vec2low)      # low-dim representation of image features
        self.low2gru_penu = self._vec2vec(channels_low2gru)      # low-dim representation of image features + clinical
        self.gru_pl = nn.GRUCell(channels_gru, channels_gru)  # recurrent abstraction
        self.gru_lc = nn.GRUCell(channels_gru, channels_gru)  # recurrent abstraction
        self.gru2phi_pl = self._vec2vec(channels_gru2phi)  # self.gru2aff_penu / recurrent abstraction + image vector to affine params
        self.gru2phi_lc = self._vec2vec(channels_gru2phi)  #
        self.gru2phi_pl[-1].weight.data.normal_(0, 0.0001)
        self.gru2phi_pl[-1].bias.data.copy_(self._affine_identity())
        self.gru2phi_lc[-1].weight.data.normal_(0, 0.0001)
        self.gru2phi_lc[-1].bias.data.copy_(self._affine_identity())
        self.prl2vec_penu = self._img2vec(channels_prl2vec)
        self.vec2gru_penu = self._vec2vec(channels_vec2gru)

        # Final up-pooling grid to full resolution
        self.up_pool = nn.AdaptiveAvgPool3d(output_size=(28, 128, 128))

    def forward(self, core, penu, clinical):
        pr_l_core = []
        pr_l_penu = []
        pr_p_core = []
        pr_c_penu = []
        gr_l_core = []
        gr_l_penu = []
        phis_cl = []
        phis_lp = []
        phis_pl = []
        phis_lc = []

        h_gru_cl = torch.zeros(self.grid_identity_core.size(0), self.gru_cl.hidden_size).cuda()
        h_gru_lp = torch.zeros(self.grid_identity_core.size(0), self.gru_lp.hidden_size).cuda()

        h_gru_pl = torch.zeros(self.grid_identity_penu.size(0), self.gru_pl.hidden_size).cuda()
        h_gru_lc = torch.zeros(self.grid_identity_penu.size(0), self.gru_lc.hidden_size).cuda()

        input = torch.cat((core, penu), dim=1)

        feat_core = self.feature_core(input)
        fvec_core = self.img2vec_core(feat_core)
        ldim_core = self.vec2low_core(fvec_core.squeeze())

        feat_penu = self.feature_penu(input)
        fvec_penu = self.img2vec_penu(feat_penu)
        ldim_penu = self.vec2low_penu(fvec_penu.squeeze())

        del input
        del fvec_core
        del fvec_penu

        prev_step = torch.zeros(self.grid_identity_core.size(0), 1).cuda()
        for i in range(self.len):
            time_step = torch.ones(self.grid_identity_core.size(0), 1).cuda() * self.seq_thr[i]
            clinical_plus = torch.cat((clinical.squeeze(), time_step - prev_step), dim=1)  # TODO add time_step?

            # Deform core to lesion
            vec_core = torch.cat((ldim_core, clinical_plus), dim=1)
            vec_core = self.low2gru_core(vec_core)
            h_gru_cl = self.gru_cl(vec_core, h_gru_cl)
            phi_cl = self.gru2phi_cl(torch.cat((h_gru_cl, ldim_core), dim=1))
            aff_cl = affine_grid(phi_cl.view(-1, 3, 4), feat_core.size())
            prl_core = grid_sample(core, aff_cl)

            # Register deformed core onto penumbra
            prl2vec_core = self.prl2vec_core(prl_core).squeeze()
            vec_core = torch.cat((prl2vec_core, ldim_core, clinical_plus), dim=1)
            vec_core = self.vec2gru_core(vec_core)
            h_gru_lp = self.gru_lp(vec_core, h_gru_lp)
            phi_lp = self.gru2phi_lp(torch.cat((h_gru_lp, prl2vec_core), dim=1))
            aff_lp = affine_grid(phi_lp.view(-1, 3, 4), feat_core.size())
            prp_core = grid_sample(prl_core, aff_lp)

            pr_l_core.append(self.up_pool(prl_core))
            pr_p_core.append(self.up_pool(prp_core))
            gr_l_core.append(self.up_pool(grid_sample(self.visual_grid, aff_cl)))
            phis_cl.append(phi_cl)
            phis_lp.append(phi_lp)

            del vec_core
            del aff_cl
            del aff_lp
            del phi_cl
            del phi_lp
            del prl_core
            del prp_core

            # Deform penumbra to lesion
            vec_penu = torch.cat((ldim_penu, clinical_plus), dim=1)
            vec_penu = self.low2gru_penu(vec_penu)
            h_gru_pl = self.gru_pl(vec_penu, h_gru_pl)
            phi_pl = self.gru2phi_pl(torch.cat((h_gru_pl, ldim_penu), dim=1))
            aff_pl = affine_grid(phi_pl.view(-1, 3, 4), feat_penu.size())
            prl_penu = grid_sample(penu, aff_pl)

            # Register deformed penumbra onto core
            prl2vec_penu = self.prl2vec_penu(prl_penu).squeeze()
            vec_penu = torch.cat((prl2vec_penu, ldim_penu, clinical_plus), dim=1)
            vec_penu = self.vec2gru_penu(vec_penu)
            h_gru_lc = self.gru_lc(vec_penu, h_gru_lc)
            phi_lc = self.gru2phi_lc(torch.cat((h_gru_lc, prl2vec_penu), dim=1))
            aff_lc = affine_grid(phi_lc.view(-1, 3, 4), feat_penu.size())
            prc_penu = grid_sample(prl_penu, aff_lc)

            pr_l_penu.append(self.up_pool(prl_penu))
            pr_c_penu.append(self.up_pool(prc_penu))
            gr_l_penu.append(self.up_pool(grid_sample(self.visual_grid, aff_pl)))
            phis_pl.append(phi_pl)
            phis_lc.append(phi_lc)

            del vec_penu
            del aff_pl
            del aff_lc
            del phi_pl
            del phi_lc
            del prl_penu
            del prc_penu

            prev_step = time_step

        return torch.cat(pr_l_core, dim=1), torch.cat(pr_l_penu[::-1], dim=1), \
               torch.cat(pr_p_core, dim=1), torch.cat(pr_c_penu[::-1], dim=1), \
               torch.cat(gr_l_core, dim=1), torch.cat(gr_l_penu[::-1], dim=1), \
               torch.cat(phis_cl, dim=1), torch.cat(phis_pl[::-1], dim=1), \
               torch.cat(phis_lp, dim=1), torch.cat(phis_lc[::-1], dim=1)