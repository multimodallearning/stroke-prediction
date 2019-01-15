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

        # Additional "normal" convolution as in vanilla Unet
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
                GRUnetBlock(h_sizes[1] + h_sizes[1], h_sizes[3], k_sizes[3], output_size=h_sizes[0])]
                #GRUnetBlock(h_sizes[0] + h_sizes[0], h_sizes[4], k_sizes[4], output_size=out_size)]

    def __init__(self, clinical_size, hidden_sizes, kernel_sizes, batch_size=4):
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


    def forward(self, input_rep, hidden=None):
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

        for i in range(self.N_BLOCKS // 2 - 1):
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

        return upd_hidden, F.interpolate(self.grid_offset(output), scale_factor=(1, 4, 4)).permute(0, 2, 3, 4, 1)


class GRUnetBidirectionalSequence(nn.Module):
    def grid_identity_def(self, bs, os, depth=1, length=128):
        result = torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float).view(-1, 3, 4).cuda()
        return nn.functional.affine_grid(result.expand(bs, 3, 4), (bs, os, depth, length, length))

    def __init__(self, grunet1, grunet2, rep_size, kernel_size, seq_len):
        super().__init__()
        self.rnn1 = grunet1
        self.rnn2 = grunet2
        self.len = seq_len
        assert seq_len > 0

        #
        # Part 1: Separate core / penumbra representations
        self.core_rep1 = GRUnetBlock(1, rep_size // 2, kernel_size)
        self.penu_rep1 = GRUnetBlock(1, rep_size // 2, kernel_size)
        self.core_rep2 = GRUnetBlock(1, rep_size // 2, kernel_size)
        self.penu_rep2 = GRUnetBlock(1, rep_size // 2, kernel_size)

        #
        # Part 2: non-lin deformation Unet
        batch_size = 4
        out_size = 6
        self.grid_identity = self.grid_identity_def(batch_size, out_size)

        self.soft = nn.AvgPool3d((9, 9, 1), (1, 1, 1), padding=(4, 4, 0))

        self.thresh = nn.Threshold(0.5, 0)

        #
        # Part 3: Learn progression marker
        ksize = (1,3,3)
        psize = (0,1,1)
        dsize = (1,2,2)
        dim_in_img = 16
        dim_hidden = 32
        self.img2vec = nn.Sequential(
            nn.InstanceNorm3d(dim_in_img),  # 128, 28
            nn.Conv3d(16, 16, kernel_size=ksize, padding=psize),  # 128, 28
            nn.ReLU(),
            nn.MaxPool3d(4, 4),  # 32, 7
            nn.InstanceNorm3d(16),
            nn.Conv3d(16, 20, kernel_size=ksize, padding=psize),  # 32, 7
            nn.ReLU(),
            nn.MaxPool3d(dsize,dsize),  # 16, 7
            nn.InstanceNorm3d(20),
            nn.Conv3d(20, 25, kernel_size=ksize),  # 14, 7
            nn.ReLU(),
            nn.MaxPool3d(dsize, dsize),  # 7, 7
            nn.InstanceNorm3d(25),
            nn.Conv3d(25, 30, kernel_size=1),  # 7, 7
            nn.ReLU(),
            nn.MaxPool3d(7),  # 1, 1
        )
        self.vec2vec1 = nn.GRUCell(dim_hidden, dim_hidden, bias=True)
        self.vec2vec2 = nn.GRUCell(dim_hidden, dim_hidden, bias=True)
        self.vec2scalar = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden // 2),
            nn.ReLU(True),
            nn.Linear(dim_hidden // 2, dim_hidden // 4),
            nn.ReLU(True),
            nn.Linear(dim_hidden // 4, 1),
            nn.Sigmoid()
        )
        self.vec2scalar[-2].weight.data.normal_(mean=0, std=0.001)  # for Sigmoid()=0.5 init
        self.vec2scalar[-2].bias.data.normal_(mean=0, std=0.1)  # for Sigmoid()=0.5 init

    def forward(self, core, penu, clinical, factor):
        factor = factor.unsqueeze(2).unsqueeze(3).unsqueeze(4)

        marker = []
        offset = []

        hidden = None
        hidden_core = None
        hidden_penu = None
        hidden_vec1 = torch.zeros((4, 32)).cuda()
        hidden_vec2 = torch.zeros((4, 32)).cuda()
        for i in range(self.len):
            hidden_core, core_rep = self.core_rep1(core, hidden_core)
            hidden_penu, penu_rep = self.penu_rep1(penu, hidden_penu)
            input_img = torch.cat((core_rep, penu_rep), dim=1)

            hidden, last_output = self.rnn1(F.interpolate(input_img, scale_factor=(1, 0.5, 0.5)), hidden)
            offset.append(factor[:, i].unsqueeze(1) * last_output)

            input_vec = torch.cat((self.img2vec(input_img), clinical), dim=1).squeeze()
            hidden_vec1 = self.vec2vec1(input_vec, hidden_vec1)
            hidden_vec2 = self.vec2vec2(hidden_vec1, hidden_vec2)
            scalar = self.vec2scalar(hidden_vec2).unsqueeze(2).unsqueeze(3).unsqueeze(4)
            marker.append(factor[:, i].unsqueeze(1) * scalar)

        hidden = None
        hidden_core = None
        hidden_penu = None
        hidden_vec1 = torch.zeros((4, 32)).cuda()
        hidden_vec2 = torch.zeros((4, 32)).cuda()
        for i in range(self.len - 1, -1, -1):
            hidden_core, core_rep = self.core_rep2(core, hidden_core)
            hidden_penu, penu_rep = self.penu_rep2(penu, hidden_penu)
            input_img = torch.cat((core_rep, penu_rep), dim=1)

            hidden, last_output = self.rnn2(F.interpolate(input_img, scale_factor=(1, 0.5, 0.5)), hidden)
            offset[i] += (1-factor[:, i]).unsqueeze(1) * last_output

            input_vec = torch.cat((self.img2vec(input_img), clinical), dim=1).squeeze()
            hidden_vec1 = self.vec2vec1(input_vec, hidden_vec1)
            hidden_vec2 = self.vec2vec2(hidden_vec1, hidden_vec2)
            scalar = self.vec2scalar(hidden_vec2).unsqueeze(2).unsqueeze(3).unsqueeze(4)
            marker[i] += (1-factor[:, i].unsqueeze(1)) * scalar

        output_by_core = []
        output_by_penu = []
        #output_factors = []
        for i in range(self.len):
            #fc = factor[:, i].unsqueeze(1)
            #zero = torch.zeros(fc.size(), requires_grad=False).cuda()
            #ones = torch.ones(fc.size(), requires_grad=False).cuda()
            #output_factors.append(torch.where(fc < 0.5, zero, ones))

            offset[i] = self.soft(offset[i])
            pred_by_core = nn.functional.grid_sample(core, self.grid_identity + offset[i][:, :, :, :, :3])
            pred_by_penu = nn.functional.grid_sample(penu, self.grid_identity + offset[i][:, :, :, :, 3:])
            output_by_core.append(pred_by_core)
            output_by_penu.append(pred_by_penu)

        return torch.cat(output_by_core, dim=1), torch.cat(output_by_penu, dim=1), torch.cat(marker, dim=1)  #torch.cat(output_factors, dim=1)
