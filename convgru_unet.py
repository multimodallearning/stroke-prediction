"""
https://github.com/jacobkimmel/pytorch_convgru
"""

import torch
import torch.nn as nn


class ConvGRUCell_Unet(nn.Module):
    """
    Generate a convolutional GRU cell
    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        if (isinstance(kernel_size, tuple) or isinstance(kernel_size, list)) and len(kernel_size) == 3:
            padding = (kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2)
        else:
            padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate = nn.Conv3d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv3d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = nn.Conv3d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)

        nn.init.orthogonal_(self.reset_gate.weight)
        nn.init.orthogonal_(self.update_gate.weight)
        nn.init.orthogonal_(self.out_gate.weight)
        nn.init.constant_(self.reset_gate.bias, 0.)
        nn.init.constant_(self.update_gate.bias, 0.)
        nn.init.constant_(self.out_gate.bias, 0.)

    def forward(self, input_, prev_state):
        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if torch.cuda.is_available():
                prev_state = torch.zeros(state_size).cuda()
            else:
                prev_state = torch.zeros(state_size)

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = torch.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state


class ConvGRU_Unet(nn.Module):
    def __init__(self, input_size, hidden_sizes, kernel_sizes, n_layers, shared_unet):
        '''
        Generates a multi-layer convolutional GRU.
        Preserves spatial dimensions across cells, only altering depth.

        Parameters
        ----------
        input_size : integer. depth dimension of input tensors.
        hidden_sizes : integer or list. depth dimensions of hidden state.
            if integer, the same hidden size is used for all cells.
        kernel_sizes : integer or list. sizes of Conv3d gate kernels.
            if integer, the same kernel size is used for all cells.
        n_layers : integer. number of chained `ConvGRUCell`.
        '''

        super(ConvGRU_Unet, self).__init__()

        self.input_size = input_size

        if type(hidden_sizes) != list:
            self.hidden_sizes = [hidden_sizes]*n_layers
        else:
            assert len(hidden_sizes) == n_layers, '`hidden_sizes` must have the same length as n_layers'
            self.hidden_sizes = hidden_sizes
        if type(kernel_sizes) != list:
            self.kernel_sizes = [kernel_sizes]*n_layers
        else:
            assert len(kernel_sizes) == n_layers, '`kernel_sizes` must have the same length as n_layers'
            self.kernel_sizes = kernel_sizes

        self.n_layers = n_layers

        cells = []
        for i in range(self.n_layers):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_sizes[i-1]

            cell = ConvGRUCell_Unet(input_dim, self.hidden_sizes[i], self.kernel_sizes[i])
            name = 'ConvGRUCell_' + str(i).zfill(2)

            setattr(self, name, cell)
            cells.append(getattr(self, name))

        self.cells = cells

        self.unet = shared_unet

    def forward(self, x, hidden=None):
        '''
        Parameters
        ----------
        x : 4D input tensor. (batch, channels, height, width).
        hidden : list of 4D hidden state representations. (batch, channels, height, width).

        Returns
        -------
        upd_hidden : 5D hidden representation. (layer, batch, channels, height, width).
        '''
        if hidden is None:
            hidden = [None] * self.n_layers
        else:
            inc = 0
            hidden_list = []
            for size in self.hidden_sizes:
                hidden_list.append(hidden[:, inc:inc+size, :, :, :])
                inc += size
            hidden = hidden_list

        input_ = self.unet(x)

        upd_hidden = []

        for layer_idx in range(self.n_layers):
            cell = self.cells[layer_idx]
            cell_hidden = hidden[layer_idx]

            # pass through layer
            upd_cell_hidden = cell(input_, cell_hidden)
            upd_hidden.append(upd_cell_hidden)
            # update input_ to the last updated hidden layer for next pass
            input_ = upd_cell_hidden

        # retain tensors in list to allow different hidden sizes
        return torch.cat(upd_hidden, dim=1)


########################################################################################################################
'''
from torch.autograd import Function
from torch.autograd.function import once_differentiable


class AffineGridGenerator(Function):
    @staticmethod
    def forward(ctx, theta, size):
        assert type(size) == torch.Size
        ctx.size = size
        ctx.is_cuda = theta.is_cuda

        if len(size) == 5:
            N, C, D, H, W = size
            base_grid = theta.new(N, D, H, W, 4)

            base_grid[:, :, :, :, 0] = (torch.linspace(-1, 1, W) if W > 1 else torch.Tensor([-1]))
            base_grid[:, :, :, :, 1] = (torch.linspace(-1, 1, H) if H > 1 else torch.Tensor([-1]))\
                .unsqueeze(-1)
            base_grid[:, :, :, :, 2] = (torch.linspace(-1, 1, D) if D > 1 else torch.Tensor([-1]))\
                .unsqueeze(-1).unsqueeze(-1)
            base_grid[:, :, :, :, 3] = 1

            grid = torch.bmm(base_grid.view(N, D * H * W, 4), theta.transpose(1, 2))
            grid = grid.view(N, D, H, W, 3)

        elif len(size) == 4:
            N, C, H, W = size
            base_grid = theta.new(N, H, W, 3)

            base_grid[:, :, :, 0] = (torch.linspace(-1, 1, W) if W > 1 else torch.Tensor([-1]))
            base_grid[:, :, :, 1] = (torch.linspace(-1, 1, H) if H > 1 else torch.Tensor([-1]))\
                .unsqueeze(-1)
            base_grid[:, :, :, 2] = 1

            grid = torch.bmm(base_grid.view(N, H * W, 3), theta.transpose(1, 2))
            grid = grid.view(N, H, W, 2)
        else:
            raise RuntimeError("AffineGridGenerator needs 4d (spatial) or 5d (volumetric) inputs.")

        ctx.base_grid = base_grid

        return grid


@staticmethod
@once_differentiable
def backward(ctx, grad_grid):
    assert ctx.is_cuda == grad_grid.is_cuda
    base_grid = ctx.base_grid

    if len(ctx.size) == 5:
        N, C, D, H, W = ctx.size
        assert grad_grid.size() == torch.Size([N, D, H, W, 3])
        grad_theta = torch.bmm(
            base_grid.view(N, D * H * W, 4).transpose(1, 2),
            grad_grid.view(N, D * H * W, 3))
    elif len(ctx.size) == 4:
        N, C, H, W = ctx.size
        assert grad_grid.size() == torch.Size([N, H, W, 2])
        grad_theta = torch.bmm(
            base_grid.view(N, H * W, 3).transpose(1, 2),
            grad_grid.view(N, H * W, 2))
    else:
        assert False

    grad_theta = grad_theta.transpose(1, 2)

    return grad_theta, None
'''


class DeformGRU_Unet(nn.Module):
    def __init__(self, input_size, hidden_sizes, kernel_sizes, n_layers, shared_unet):
        '''
        Generates a multi-layer convolutional GRU.
        Preserves spatial dimensions across cells, only altering depth.

        Parameters
        ----------
        input_size : integer. depth dimension of input tensors.
        hidden_sizes : integer or list. depth dimensions of hidden state.
            if integer, the same hidden size is used for all cells.
        kernel_sizes : integer or list. sizes of Conv3d gate kernels.
            if integer, the same kernel size is used for all cells.
        n_layers : integer. number of chained `ConvGRUCell`.
        '''

        super(DeformGRU_Unet, self).__init__()

        self.input_size = input_size

        if type(hidden_sizes) != list:
            self.hidden_sizes = [hidden_sizes]*n_layers
        else:
            assert len(hidden_sizes) == n_layers, '`hidden_sizes` must have the same length as n_layers'
            self.hidden_sizes = hidden_sizes
        if type(kernel_sizes) != list:
            self.kernel_sizes = [kernel_sizes]*n_layers
        else:
            assert len(kernel_sizes) == n_layers, '`kernel_sizes` must have the same length as n_layers'
            self.kernel_sizes = kernel_sizes

        self.n_layers = n_layers

        cells = []
        for i in range(self.n_layers):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_sizes[i-1]

            cell = ConvGRUCell_Unet(input_dim, self.hidden_sizes[i], self.kernel_sizes[i])
            name = 'ConvGRUCell_' + str(i).zfill(2)

            setattr(self, name, cell)
            cells.append(getattr(self, name))

        self.cells = cells

        self.unet = shared_unet

        #### Deform ####

        ksize = (1, 3, 3)
        psize = (0, 1, 1)
        dsize = (1, 2, 2)

        self.input_channels_mixin = nn.Sequential(
            nn.Conv3d(self.hidden_sizes[0], self.hidden_sizes[0], kernel_size=ksize, padding=psize),
            nn.ReLU(),
            nn.Conv3d(self.hidden_sizes[0], 1, kernel_size=1, padding=0),
            nn.ReLU(),
        )

        self.params_from_hidden = nn.Sequential(
            nn.Conv3d(self.hidden_sizes[-1], self.hidden_sizes[-1], kernel_size=(1, 7, 7)),
            nn.MaxPool3d(dsize, stride=dsize),
            nn.ReLU(True),
            nn.Conv3d(self.hidden_sizes[-1], self.hidden_sizes[-1] // 2, kernel_size=(1, 5, 5)),
            nn.MaxPool3d(dsize, stride=dsize),
            nn.ReLU(True),
            nn.Conv3d(self.hidden_sizes[-1] // 2, self.hidden_sizes[-1] // 2, kernel_size=(1, 5, 5)),
            nn.MaxPool3d(dsize, stride=dsize),
            nn.ReLU(True),
            nn.Conv3d(self.hidden_sizes[-1] // 2, 3, kernel_size=(1, 3, 3)),
            nn.MaxPool3d(dsize, stride=dsize),
            nn.ReLU(True)
        )

        self.fc_params = nn.Sequential(
            nn.Linear(3 * 5 * 5, 32),
            nn.ReLU(True),
            nn.Linear(32, 4 * 3)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_params[2].weight.data.zero_()
        self.fc_params[2].bias.data.copy_(torch.tensor([0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.5, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x, hidden):
        hidden_result = self.params_from_hidden(hidden)
        hidden_result = hidden_result.view(-1, 3 * 5 * 5)
        theta = self.fc_params(hidden_result)
        theta = theta.view(-1, 3, 4)

        grid = nn.functional.affine_grid(theta, x.size())
        '''
        identity = torch.zeros((batchsize, 3, 4), dtype=torch.float)
        identity[:, 0, 0] = 1.0
        identity[:, 1, 1] = 1.0
        identity[:, 2, 2] = 1.0
        if use_gpu:
            identity = identity.cuda()

        grid = AffineGridGenerator.apply(identity, size)
        '''
        print(theta)
        x = nn.functional.grid_sample(x, grid)

        return x

    def forward(self, x, hidden=None):
        '''
        Parameters
        ----------
        x : 4D input tensor. (batch, channels, height, width).
        hidden : list of 4D hidden state representations. (batch, channels, height, width).

        Returns
        -------
        upd_hidden : 5D hidden representation. (layer, batch, channels, height, width).
        '''
        if hidden is None:
            hidden = [None] * self.n_layers
        else:
            inc = 0
            hidden_list = []
            for size in self.hidden_sizes:
                hidden_list.append(hidden[:, inc:inc+size, :, :, :])
                inc += size
            hidden = hidden_list

        input_ = self.unet(x)

        upd_hidden = []

        for layer_idx in range(self.n_layers):
            cell = self.cells[layer_idx]
            cell_hidden = hidden[layer_idx]

            # pass through layer
            upd_cell_hidden = cell(input_, cell_hidden)
            upd_hidden.append(upd_cell_hidden)
            # update input_ to the last updated hidden layer for next pass
            input_ = upd_cell_hidden

        #### Deform ####

        sampler_input = self.input_channels_mixin(input_)
        result = self.stn(sampler_input, input_)

        # retain tensors in list to allow different hidden sizes
        return torch.cat(upd_hidden, dim=1), result
