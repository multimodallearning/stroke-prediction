"""
Based on:
https://github.com/jacobkimmel/pytorch_convgru
"""

import torch
import torch.nn as nn


class GRUnetBlock(nn.Module):
    """
    Generate a convolutional GRU cell
    """

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
    def __init__(self, clinical_size, hidden_sizes, kernel_sizes, output_size=1):
        '''
        Generates a RecurrentUnet.
        Preserves spatial dimensions across cells, only altering depth.

        Parameters
        ----------
        clinical_size : integer. depth dimension of clinical input tensors.
        hidden_sizes : integer or list. depth dimensions of hidden state.
            if integer, the same hidden size is used for all cells.
        kernel_sizes : integer or list. sizes of Conv3d gate kernels.
            if integer, the same kernel size is used for all cells.
        n_layers : integer. number of chained `ConvGRUCell`.
        '''
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

        upsample = (28, 128, 128)
        if type(kernel_sizes[0]) == tuple:
            upsample = (1, 128, 128)

        self.core_rep = GRUnetBlock(1, self.hidden_sizes[0] // 2, self.kernel_sizes[0])
        self.penumbra_rep = GRUnetBlock(1, self.hidden_sizes[0] // 2, self.kernel_sizes[0])
        self.clinical_rep = nn.Sequential(
            nn.Conv3d(clinical_size, clinical_size * 2, 1),
            nn.ReLU(),
            nn.Conv3d(clinical_size * 2, 1, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=upsample)
        )

        self.blocks = [GRUnetBlock(hidden_sizes[0] + 1, self.hidden_sizes[0], self.kernel_sizes[0]),
                       GRUnetBlock(self.hidden_sizes[0], self.hidden_sizes[1], self.kernel_sizes[1]),
                       GRUnetBlock(self.hidden_sizes[1], self.hidden_sizes[2], self.kernel_sizes[2], output_size=self.hidden_sizes[1]),
                       GRUnetBlock(self.hidden_sizes[1] + self.hidden_sizes[1], self.hidden_sizes[3], self.kernel_sizes[3], output_size=self.hidden_sizes[0]),
                       GRUnetBlock(self.hidden_sizes[0] + self.hidden_sizes[0], self.hidden_sizes[4], self.kernel_sizes[4], output_size=output_size)]

        for i in range(len(self.blocks)):
            setattr(self, 'GRUnetBlock' + str(i).zfill(2), self.blocks[i])

        pool = 2
        if type(kernel_sizes[0]) == tuple:
            pool = (1, 2, 2)

        self.pool = nn.MaxPool3d(pool, pool, return_indices=True)
        self.unpool = nn.MaxUnpool3d(pool, pool)

        self.output = nn.Sigmoid()

    def forward(self, core, penumbra, clinical, hidden=None):
        '''
        Parameters
        ----------
        x : 4D input tensor. (batch, channels, height, width).
        hidden : list of 4D hidden state representations. (batch, channels, height, width).

        Returns
        -------
        upd_hidden : 5D hidden representation. (layer, batch, channels, height, width).
        '''

        outputs = [None] * self.N_BLOCKS
        indices = [None] * (self.N_BLOCKS // 2)
        upd_hidden = [None] * self.N_BLOCKS
        if hidden is None:
            hidden = [None] * (self.N_BLOCKS + 2)

        hidden_core = hidden[0]
        hidden_penu = hidden[1]
        hidden = hidden[2:]

        h_c, core = self.core_rep(core, hidden_core)
        h_p, penumbra = self.penumbra_rep(penumbra, hidden_penu)
        clinical = self.clinical_rep(clinical)

        input_ = torch.cat((core, penumbra, clinical), dim=1)
        del core
        del penumbra
        del clinical

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

        for i in range(self.N_BLOCKS // 2):
            unpool = self.unpool(output, indices[self.N_BLOCKS // 2 - (i + 1)])
            skip = outputs[self.N_BLOCKS // 2 - (i + 1)]
            input_ = torch.cat((unpool, skip), dim=1)

            j = self.N_BLOCKS // 2 + (i + 1)
            upd_block_hidden, output = self.blocks[j](input_, hidden[j])
            upd_hidden[j] = upd_block_hidden
            outputs[j] = output

        del upd_block_hidden
        del outputs
        del input_
        del unpool
        del skip

        return [h_c, h_p] + upd_hidden, self.output(output)


class GRUnetSequence(nn.Module):
    def __init__(self, grunet, seq_len):
        super().__init__()
        self.rnn = grunet
        self.len = seq_len
        assert seq_len > 0

    def forward(self, core, penumbra, clinical):
        hidden = None
        output = []
        for i in range(self.len):
            hidden, last_output = self.rnn(core, penumbra, clinical, hidden)
            output.append(last_output)
        return torch.cat(output, dim=1)