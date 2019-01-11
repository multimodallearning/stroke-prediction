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

        #
        # Part 1: Separate core / penumbra representations
        self.core_rep = GRUnetBlock(1, self.hidden_sizes[0] // 2, self.kernel_sizes[0])
        self.penu_rep = GRUnetBlock(1, self.hidden_sizes[0] // 2, self.kernel_sizes[0])

        #
        # Part 2: non-lin deformation Unet
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


    def forward(self, core, penu, clinical, hidden=None):
        outputs = [None] * (self.N_BLOCKS // 2 + 1)
        indices = [None] * (self.N_BLOCKS // 2)
        upd_hidden = [None] * self.N_BLOCKS
        if hidden is None:
            hidden = [None] * (self.N_BLOCKS + 2)

        hidden_core = hidden[0]
        hidden_penu = hidden[1]
        hidden = hidden[2:]

        h_c, core_rep = self.core_rep(core, hidden_core)
        h_p, penu_rep = self.penu_rep(penu, hidden_penu)
        input_ = F.interpolate(torch.cat((core_rep, penu_rep), dim=1), scale_factor=(1, 0.5, 0.5))
        del core_rep
        del penu_rep

        #
        # Non-lin deform

        for i in range(self.N_BLOCKS // 2):
            upd_block_hidden, output = self.blocks[i](input_, hidden[i])
            upd_hidden[i] = upd_block_hidden
            outputs[i] = output

            input_, indices_ = self.pool(output)
            indices[i] = indices_
        del indices_

        upd_block_hidden, output = self.blocks[self.N_BLOCKS // 2](input_, hidden[self.N_BLOCKS // 2])
        upd_hidden[self.N_BLOCKS // 2] = upd_block_hidden
        outputs[self.N_BLOCKS // 2] = output

        for i in range(self.N_BLOCKS // 2 - 1):
            unpool = self.unpool(output, indices[self.N_BLOCKS // 2 - (i + 1)])
            skip = outputs[self.N_BLOCKS // 2 - (i + 1)]
            input_ = torch.cat((unpool, skip), dim=1)

            j = self.N_BLOCKS // 2 + (i + 1)
            upd_block_hidden, output = self.blocks[j](input_, hidden[j])
            upd_hidden[j] = upd_block_hidden

        del upd_block_hidden
        del outputs
        del input_
        del unpool
        del skip

        _out = F.interpolate(self.grid_offset(output), scale_factor=(1, 4, 4)).permute(0, 2, 3, 4, 1)

        return [h_c, h_p] + upd_hidden, _out


class GRUnetBidirectionalSequence(nn.Module):
    def grid_identity_def(self, bs, os, depth=1, length=128):
        result = torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float).view(-1, 3, 4).cuda()
        return nn.functional.affine_grid(result.expand(bs, 3, 4), (bs, os, depth, length, length))

    def __init__(self, grunet1, grunet2, seq_len):
        super().__init__()
        self.rnn1 = grunet1  # bi-directional
        self.rnn2 = grunet2  # bi-directional
        self.len = seq_len
        assert seq_len > 0

        batch_size = 4
        out_size = 6
        self.grid_identity = self.grid_identity_def(batch_size, out_size)

        self.soft = nn.AvgPool3d((9, 9, 1), (1, 1, 1), padding=(4, 4, 0))

        self.thresh = nn.Threshold(0.5, 0)

    def forward(self, core, penu, clinical, t_core):
        length = [self.len - t_core[i] for i in range(len(t_core))]
        t_half = [t_core[b]+length[b]//2 for b in range(len(t_core))]
        factor = torch.tensor([[1] * self.len] * len(t_core), dtype=torch.float).cuda()

        for b in range(len(t_core)):
            factor[b, :t_core[b]] = 1
            factor[b, t_core[b]:t_half[b]] = torch.tensor([1 - i/length[b] for i in range(length[b]//2)], dtype=torch.float).cuda()
            factor[b, t_half[b]:] = torch.tensor([(length[b]//2)/length[b] - i/length[b] for i in range(length[b] - length[b]//2)], dtype=torch.float).cuda()

        factor = factor.unsqueeze(2).unsqueeze(3).unsqueeze(4)

        offset = []
        hidden = None
        for i in range(self.len):
            hidden, last_output = self.rnn1(core, penu, clinical, hidden)
            offset.append(factor[:, i].unsqueeze(1) * last_output)

        hidden = None
        for i in range(self.len - 1, -1, -1):
            hidden, last_output = self.rnn2(core, penu, clinical, hidden)
            offset[i] += (1-factor[:, i].unsqueeze(1)) * last_output

        output_by_core = []
        output_by_penu = []
        output_factors = []
        for i in range(self.len):
            offset[i] = self.soft(offset[i])
            fc = factor[:, i].unsqueeze(1)
            zero = torch.zeros(fc.size(), requires_grad=False).cuda()
            ones = torch.ones(fc.size(), requires_grad=False).cuda()
            output_factors.append(torch.where(fc < 0.5, zero, ones))
            pred_by_core = nn.functional.grid_sample(core, self.grid_identity + offset[i][:, :, :, :, :3])
            pred_by_penu = nn.functional.grid_sample(penu, self.grid_identity + offset[i][:, :, :, :, 3:])
            output_by_core.append(pred_by_core)
            output_by_penu.append(pred_by_penu)

        return torch.cat(output_by_core, dim=1), torch.cat(output_by_penu, dim=1), torch.cat(output_factors, dim=1)
