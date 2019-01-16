"""
Based on:
https://github.com/jacobkimmel/pytorch_convgru
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUnetBlock(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, output_size=None):
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
        self.conv3d = nn.Conv3d(hidden_size, output_size, kernel_size, padding=padding)

        # Appropriate initialization
        nn.init.orthogonal_(self.reset_gate.weight)
        nn.init.orthogonal_(self.update_gate.weight)
        nn.init.orthogonal_(self.out_gate.weight)
        nn.init.xavier_normal(self.conv3d.weight)
        nn.init.normal(self.conv3d.bias)
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
    def unet_def(self, h_sizes, k_sizes):
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

        self.blocks = self.unet_def(hidden_sizes, kernel_sizes)
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


    def forward(self, input_rep, hidden=None):
        input_rep = F.interpolate(input_rep, scale_factor=(1, 1/self.downscaling, 1/self.downscaling))

        outputs = [None] * (self.N_BLOCKS // 2 + 1)
        indices = [None] * (self.N_BLOCKS // 2)
        upd_hidden = [None] * self.N_BLOCKS
        if hidden is None:
            hidden = [None] * (self.N_BLOCKS + 2)

        #
        # Non-lin deform

        for i in range(self.N_BLOCKS // 2):
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

        return upd_hidden, output.permute(0, 2, 3, 4, 1)


class AffineModule(nn.Module):
    def affine_identity(self):
        return torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
                             1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float)

    def init_theta(self, pos):
        def _init(sequential):
            sequential[pos].weight.data.zero_()
            sequential[pos].bias.data.copy_(self.affine_identity())
            return sequential

        return _init

    def def_img2vec(self, n_dim=[10, 15, 20, 25, 30], ksize=(1, 3, 3), psize=(0, 1, 1), dsize=(1, 2, 2)):
        assert len(n_dim) == 5
        return nn.Sequential(
            nn.InstanceNorm3d(n_dim[0]),  # 128x128x28
            nn.Conv3d(n_dim[0], n_dim[1], kernel_size=ksize, padding=psize),  # 128x128x28
            nn.ReLU(),
            nn.MaxPool3d(4, 4),  # 32x32x7
            nn.InstanceNorm3d(n_dim[1]),
            nn.Conv3d(n_dim[1], n_dim[2], kernel_size=ksize, padding=psize),  # 32x32x7
            nn.ReLU(),
            nn.MaxPool3d(dsize,dsize),  # 16x16x7
            nn.InstanceNorm3d(n_dim[2]),
            nn.Conv3d(n_dim[2], n_dim[3], kernel_size=ksize),  # 14x14x7
            nn.ReLU(),
            nn.MaxPool3d(dsize, dsize),  # 7x7x7
            nn.InstanceNorm3d(n_dim[3]),
            nn.Conv3d(n_dim[3], n_dim[4], kernel_size=1),  # 7x7x7
            nn.ReLU(),
            nn.MaxPool3d(7)  # 1x1x1
        )

    def def_vec2vec(self, n_dim=[16, 16], final_activation=None, init_fn=lambda x: x):
        assert len(n_dim) > 1

        result = []
        for i in range(1, len(n_dim)-1):
            result += [nn.Linear(n_dim[i-1], n_dim[i]), nn.ReLU(True)]
        result += [nn.Linear(n_dim[len(n_dim)-2], n_dim[len(n_dim)-1])]

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

    def __init__(self, dim_in_img, dim_hidden, kernel_size, seq_len):
        super().__init__()

        self.len = seq_len

        self.affine1 = GRUnetBlock(dim_in_img, dim_in_img, kernel_size)
        self.affine2 = self.def_img2vec(n_dim=[dim_in_img, 16, 20, 25, 30])
        self.affine3 = nn.GRUCell(dim_hidden, dim_hidden, bias=True)
        self.affine4 = self.def_vec2vec(n_dim=[dim_hidden, dim_hidden // 2, dim_hidden // 2, 24], init_fn=self.init_theta(-1))
        self.affine5 = nn.GRUCell(24, 24, bias=True)

    def forward(self, input_img, clinical, out_size, hidden_affine1, hidden_affine3, hidden_affine5):

        hidden_affine1, affine1 = self.affine1(input_img, hidden_affine1)
        affine2 = self.affine2(affine1)
        hidden_affine3 = self.affine3(torch.cat((affine2, clinical), dim=1).squeeze(), hidden_affine3)
        affine4 = self.affine4(hidden_affine3)
        hidden_affine5 = self.affine5(affine4, hidden_affine5)
        grid_core = nn.functional.affine_grid(hidden_affine5[:, :12].view(-1, 3, 4), out_size)
        grid_penu = nn.functional.affine_grid(hidden_affine5[:, 12:].view(-1, 3, 4), out_size)

        return torch.cat((grid_core, grid_penu), dim=4), hidden_affine1, hidden_affine3, hidden_affine5


class UnidirectionalNet(nn.Module):
    def _init_zero_normal(self, pos):
        def _init(sequential):
            sequential[pos].weight.data.normal_(mean=0, std=0.001)  # for Sigmoid()=0.5 init
            sequential[pos].bias.data.normal_(mean=0, std=0.1)  # for Sigmoid()=0.5 init
            return sequential

        return _init

    '''
    def def_marker_module(self, dim_in_img, dim_hidden):
        marker1 = GRUnetBlock(dim_in_img, dim_in_img)
        marker2 = self.def_img2vec(n_dim=[dim_in_img, 16, 20, 25, 30])
        marker3 = nn.GRUCell(dim_hidden, dim_hidden, bias=True)

        marker = self.def_vec2vec(n_dim=[dim_hidden, dim_hidden // 2, dim_hidden // 4, 1],
                                  final_activation='sigmoid',
                                  init_fn=self._init_zero_normal(-2))

        return marker1, marker2, marker3, marker
    '''

    def __init__(self, grunet, rep_size, kernel_size, seq_len, dim_hidden=32):
        super().__init__()

        self.len = seq_len

        dim_in_img = 16 - 6

        #
        # Separate (hidden) features for core / penumbra
        self.core_rep = GRUnetBlock(rep_size // 2 -4, rep_size // 2 -4, kernel_size)
        self.penu_rep = GRUnetBlock(rep_size // 2 -4, rep_size // 2 -4, kernel_size)

        #
        # Affine
        self.affine = AffineModule(dim_in_img, dim_hidden, kernel_size, seq_len)

        #
        # Non-lin.
        self.grunet = grunet

        '''
        #
        # Marker
        self.marker1, self.marker2, self.marker3, self.marker = self.def_marker_module(30?, dim_hidden)

        self.vec2vec1 = nn.GRUCell(dim_hidden, dim_hidden, bias=True)
        self.vec2vec2a = nn.GRUCell(dim_hidden, dim_hidden, bias=True)

        self.vec2scalar = self.def_vec2vec(n_dim=[dim_hidden, dim_h2, dim_h4, 1],
                                           final_activation='sigmoid',
                                           init_fn=self._init_zero_normal(-2))

        self.vec2vec2b = nn.GRUCell(dim_hidden, dim_hidden, bias=True)

        self.vec2theta_core = self.def_vec2vec(n_dim=[dim_hidden, dim_h2, dim_h2, 12],
                                               init_fn=self._init_theta(-1))
        self.vec2theta_penu = self.def_vec2vec(n_dim=[dim_hidden, dim_h2, dim_h2, 12],
                                               init_fn=self._init_theta(-1))
        '''

    def forward(self, core, penu, core_rep, penu_rep, clinical):
        offset = []

        hidden_core = None
        hidden_penu = None
        hidden_grunet = None
        hidden_affine1 = None
        hidden_affine3 = torch.zeros((4, 32)).cuda()
        hidden_affine5 = torch.zeros((4, 24)).cuda()

        for i in range(self.len):
            hidden_core, core_rep = self.core_rep(core_rep, hidden_core)
            hidden_penu, penu_rep = self.penu_rep(penu_rep, hidden_penu)
            input_img = torch.cat((core_rep, penu_rep, core, penu), dim=1)

            affine_grids, hidden_affine1, hidden_affine3, hidden_affine5 = self.affine(input_img, clinical, core.size(),
                                                                                       hidden_affine1, hidden_affine3,
                                                                                       hidden_affine5)

            input_grunet = torch.cat((input_img, affine_grids.permute(0,4,1,2,3)), dim=1)
            hidden_grunet, last_output = self.grunet(input_grunet, hidden_grunet)
            offset.append(last_output)

            '''
            # marker
            hidden_vec2 = self.vec2vec2a(hidden_vec1, hidden_vec2)
            scalar = self.vec2scalar(hidden_vec2).unsqueeze(2).unsqueeze(3).unsqueeze(4)
            marker.append(scalar)
            '''

        return offset


class BidirectionalSequence(nn.Module):
    def common_rep(self, n_in, n_out, k_size=(1,3,3), p_size=(0,1,1)):
        n_middle = (n_in + n_out) // 2
        return nn.Sequential(
            nn.InstanceNorm3d(n_in),
            nn.Conv3d(n_in, n_middle, kernel_size=k_size, padding=p_size),
            nn.ReLU(),
            nn.InstanceNorm3d(n_middle),
            nn.Conv3d(n_middle, n_out, kernel_size=k_size, padding=p_size),
            nn.ReLU()
        )

    def __init__(self, rep_size, kernel_size, seq_len, batch_size=4, out_size=6, convgru_kernel=3):
        super().__init__()
        self.len = seq_len
        assert seq_len > 0

        #
        # Part 1: Commonly used separate core/penumbra representations
        self.common_core = self.common_rep(1, rep_size // 2 - 4)
        self.common_penu = self.common_rep(1, rep_size // 2 - 4)

        #
        # Part 2: Bidirectional Recurrence
        grunet1 = GRUnet(hidden_sizes=[16, 32, 64, 32, 16], kernel_sizes=[convgru_kernel] * 5, down_scaling=2)
        grunet2 = GRUnet(hidden_sizes=[16, 32, 64, 32, 16], kernel_sizes=[convgru_kernel] * 5, down_scaling=2)
        self.rnn1 = UnidirectionalNet(grunet1, rep_size, kernel_size, seq_len)
        self.rnn2 = UnidirectionalNet(grunet2, rep_size, kernel_size, seq_len)
        del grunet1
        del grunet2

        #
        # Part 3: Combine predictions of both directions
        self.grid_identity = torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float)
        self.grid_identity = self.grid_identity.view(-1, 3, 4).expand(batch_size, 3, 4).cuda()
        self.grid_identity = nn.functional.affine_grid(self.grid_identity, (batch_size, out_size, 1, 128, 128))

        self.soften = nn.AvgPool3d((9, 9, 1), (1, 1, 1), padding=(4, 4, 0))


    def forward(self, core, penu, clinical, factor):
        factor = factor.unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5)  # one additional dim for later when squeeze

        core_rep = self.common_core(core)
        penu_rep = self.common_penu(penu)

        offset1 = self.rnn1(core, penu, core_rep, penu_rep, clinical)
        offset2 = self.rnn2(core, penu, core_rep, penu_rep, clinical)

        offset = [factor[:, i]*offset1[i] + (1-factor[:, i])*offset2[self.len - i - 1] for i in range(self.len)]

        del offset1
        del offset2

        output_by_core = []
        output_by_penu = []
        output_factors = []

        for i in range(self.len):
            fc = factor[:, i]
            zero = torch.zeros(fc.size(), requires_grad=False).cuda()
            ones = torch.ones(fc.size(), requires_grad=False).cuda()
            output_factors.append(torch.where(fc < 0.5, zero, ones))

            offset[i] = self.soften(offset[i])
            pred_by_core = nn.functional.grid_sample(core, self.grid_identity + offset[i][:, :, :, :, :3])
            pred_by_penu = nn.functional.grid_sample(penu, self.grid_identity + offset[i][:, :, :, :, 3:])
            output_by_core.append(pred_by_core)
            output_by_penu.append(pred_by_penu)

        return torch.cat(output_by_core, dim=1), torch.cat(output_by_penu, dim=1)  #, torch.cat(marker, dim=1),  torch.cat(output_factors, dim=1)
