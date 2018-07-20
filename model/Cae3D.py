import torch
import torch.nn as nn
from common.CaeDto import CaeDto


class CaeBase(nn.Module):

    def __init__(self, size_input_xy=128, size_input_z=28, channels=[1, 16, 32, 64, 128, 1024, 128, 1], n_ch_global=2,
                 leakage=0.01, inner_xy=12, inner_z=3):
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
        self.leakage = leakage

    def freeze(self, freeze=False):
        requires_grad = not freeze
        for param in self.parameters():
            param.requires_grad = requires_grad


class Enc3D(CaeBase):
    def __init__(self, size_input_xy, size_input_z, channels, n_ch_global, leakage, datatype=None):
        super().__init__(size_input_xy, size_input_z, channels, n_ch_global, leakage, inner_xy=10, inner_z=3)

        self.datatype = datatype

        self.encoder = nn.Sequential(
            nn.BatchNorm3d(self.n_input),
            nn.Conv3d(self.n_input, self.n_ch_origin, 3, stride=1, padding=(1, 0, 0)),
            nn.LeakyReLU(self.leakage, True),
            nn.BatchNorm3d(self.n_ch_origin),
            nn.Conv3d(self.n_ch_origin, self.n_ch_origin, 3, stride=1, padding=(1, 0, 0)),
            nn.LeakyReLU(self.leakage, True),

            nn.BatchNorm3d(self.n_ch_origin),
            nn.Conv3d(self.n_ch_origin, self.n_ch_down2x, 3, stride=2, padding=1),
            nn.LeakyReLU(self.leakage, True),

            nn.BatchNorm3d(self.n_ch_down2x),
            nn.Conv3d(self.n_ch_down2x, self.n_ch_down2x, 3, stride=1, padding=(1, 0, 0)),
            nn.LeakyReLU(self.leakage, True),
            nn.BatchNorm3d(self.n_ch_down2x),
            nn.Conv3d(self.n_ch_down2x, self.n_ch_down2x, 3, stride=1, padding=(1, 0, 0)),
            nn.LeakyReLU(self.leakage, True),

            nn.BatchNorm3d(self.n_ch_down2x),
            nn.Conv3d(self.n_ch_down2x, self.n_ch_down4x, 3, stride=2, padding=1),
            nn.LeakyReLU(self.leakage, True),

            nn.BatchNorm3d(self.n_ch_down4x),
            nn.Conv3d(self.n_ch_down4x, self.n_ch_down4x, 3, stride=1, padding=(1, 0, 0)),
            nn.LeakyReLU(self.leakage, True),
            nn.BatchNorm3d(self.n_ch_down4x),
            nn.Conv3d(self.n_ch_down4x, self.n_ch_down4x, 3, stride=1, padding=(1, 0, 0)),
            nn.LeakyReLU(self.leakage, True),

            nn.BatchNorm3d(self.n_ch_down4x),
            nn.Conv3d(self.n_ch_down4x, self.n_ch_down8x, 3, stride=2, padding=0),
            nn.LeakyReLU(self.leakage, True),

            nn.BatchNorm3d(self.n_ch_down8x),
            nn.Conv3d(self.n_ch_down8x, self.n_ch_fc, 3, stride=1, padding=0),
            nn.LeakyReLU(self.leakage, True),
        )

        self.convex_combine = nn.Sequential(
            nn.Linear(self.n_ch_global, 1),
            nn.LeakyReLU(self.leakage, True)
        )

    def _interpolate(self, latent_core, latent_penu, step):
        core_to_penumbra = latent_penu - latent_core
        results = []
        for batch_sample in range(step.size()[0]):
            results.append(
                (latent_core[batch_sample, :, :, :, :] + step[batch_sample, :, :, :, :] *
                 core_to_penumbra[batch_sample, :, :, :, :]).unsqueeze(0)
            )
        return torch.cat(results, dim=0)

    def _forward_single(self, input_image):
        if input_image is None:
            return None
        return self.encoder(input_image)

    def _get_step(self, dto: CaeDto):
        step = dto.given_variables.time_to_treatment
        return step

    def _forward_shape(self, dto: CaeDto, step):
        dto.latents.gtruth.core = self._forward_single(dto.given_variables.gtruth.core)
        dto.latents.gtruth.penu = self._forward_single(dto.given_variables.gtruth.penu)
        dto.latents.gtruth.lesion = self._forward_single(dto.given_variables.gtruth.lesion)
        dto.latents.gtruth.interpolation = self._interpolate(dto.latents.gtruth.core,
                                                             dto.latents.gtruth.penu,
                                                             step)
        return dto

    def _forward_inputs(self, dto: CaeDto, step):
        dto.latents.inputs.core = self._forward_single(dto.given_variables.inputs.core)
        dto.latents.inputs.penu = self._forward_single(dto.given_variables.inputs.penu)
        dto.latents.inputs.interpolation = self._interpolate(dto.latents.inputs.core,
                                                             dto.latents.inputs.penu,
                                                             step)
        return dto

    def forward(self, dto: CaeDto):
        step = self._get_step(dto)
        return self._forward_shape(dto, step)
        # return self._forward_inputs(dto, step)


class Dec3D(CaeBase):
    def __init__(self, size_input_xy, size_input_z, channels, n_ch_global, leakage):
        super().__init__(size_input_xy, size_input_z, channels, n_ch_global, leakage, inner_xy=10, inner_z=3)

        self.decoder = nn.Sequential(
            nn.BatchNorm3d(self.n_ch_fc),
            nn.ConvTranspose3d(self.n_ch_fc, self.n_ch_down8x, 3, stride=1, padding=0, output_padding=0),
            nn.LeakyReLU(leakage, True),

            nn.BatchNorm3d(self.n_ch_down8x),
            nn.ConvTranspose3d(self.n_ch_down8x, self.n_ch_down4x, 3, stride=2, padding=0, output_padding=0),
            nn.LeakyReLU(leakage, True),

            nn.BatchNorm3d(self.n_ch_down4x),
            nn.Conv3d(self.n_ch_down4x, self.n_ch_down4x, 3, stride=1, padding=(1, 2, 2)),
            nn.LeakyReLU(leakage, True),
            nn.BatchNorm3d(self.n_ch_down4x),
            nn.Conv3d(self.n_ch_down4x, self.n_ch_down2x, 3, stride=1, padding=(1, 2, 2)),
            nn.LeakyReLU(leakage, True),

            nn.BatchNorm3d(self.n_ch_down2x),
            nn.ConvTranspose3d(self.n_ch_down2x, self.n_ch_down2x, 2, stride=2, padding=0, output_padding=0),
            nn.LeakyReLU(leakage, True),

            nn.BatchNorm3d(self.n_ch_down2x),
            nn.Conv3d(self.n_ch_down2x, self.n_ch_down2x, 3, stride=1, padding=(1, 2, 2)),
            nn.LeakyReLU(leakage, True),
            nn.BatchNorm3d(self.n_ch_down2x),
            nn.Conv3d(self.n_ch_down2x, self.n_ch_origin, 3, stride=1, padding=(1, 2, 2)),
            nn.LeakyReLU(leakage, True),

            nn.BatchNorm3d(self.n_ch_origin),
            nn.ConvTranspose3d(self.n_ch_origin, self.n_ch_origin, 2, stride=2, padding=0, output_padding=0),
            nn.LeakyReLU(leakage, True),

            nn.BatchNorm3d(self.n_ch_origin),
            nn.Conv3d(self.n_ch_origin, self.n_ch_origin, 3, stride=1, padding=(1, 2, 2)),
            nn.LeakyReLU(leakage, True),
            nn.BatchNorm3d(self.n_ch_origin),
            nn.Conv3d(self.n_ch_origin, self.n_ch_origin, 3, stride=1, padding=(1, 2, 2)),
            nn.LeakyReLU(leakage, True),

            nn.BatchNorm3d(self.n_ch_origin),
            nn.Conv3d(self.n_ch_origin, self.n_ch_origin, 1, stride=1, padding=0),
            nn.LeakyReLU(leakage, True),
            nn.BatchNorm3d(self.n_ch_origin),
            nn.Conv3d(self.n_ch_origin, self.n_classes, 1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def _forward_single(self, input_latent):
        if input_latent is None:
            return None
        return self.decoder(input_latent)

    def _forward_gtruth(self, dto: CaeDto):
        dto.reconstructions.gtruth.core = self._forward_single(dto.latents.gtruth.core)
        dto.reconstructions.gtruth.penu = self._forward_single(dto.latents.gtruth.penu)
        dto.reconstructions.gtruth.lesion = self._forward_single(dto.latents.gtruth.lesion)
        dto.reconstructions.gtruth.interpolation = self._forward_single(dto.latents.gtruth.interpolation)
        dto.reconstructions.inputs.core = self._forward_single(dto.latents.inputs.core)
        dto.reconstructions.inputs.penu = self._forward_single(dto.latents.inputs.penu)
        dto.reconstructions.inputs.interpolation = self._forward_single(dto.latents.inputs.interpolation)
        return dto

    def forward(self, dto: CaeDto):
        return self._forward_gtruth(dto)


class Cae3D(nn.Module):
    def __init__(self, enc, dec):
        super().__init__()
        self.enc = enc
        self.dec = dec

    def forward(self, dto: CaeDto):
        dto = self.enc(dto)
        dto = self.dec(dto)
        return dto
