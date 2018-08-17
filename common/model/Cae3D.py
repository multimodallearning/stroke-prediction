import torch
import torch.nn as nn
import common.data as data
from common.dto.CaeDto import CaeDto
import common.dto.CaeDto as CaeDtoUtil


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
            nn.ELU(self.alpha, True),
            nn.BatchNorm3d(self.n_ch_origin),
            nn.Conv3d(self.n_ch_origin, self.n_ch_origin, 3, stride=1, padding=(1, 0, 0)),
            nn.ELU(self.alpha, True),

            nn.BatchNorm3d(self.n_ch_origin),
            nn.Conv3d(self.n_ch_origin, self.n_ch_down2x, 3, stride=2, padding=1),
            nn.ELU(self.alpha, True),

            nn.BatchNorm3d(self.n_ch_down2x),
            nn.Conv3d(self.n_ch_down2x, self.n_ch_down2x, 3, stride=1, padding=(1, 0, 0)),
            nn.ELU(self.alpha, True),
            nn.BatchNorm3d(self.n_ch_down2x),
            nn.Conv3d(self.n_ch_down2x, self.n_ch_down2x, 3, stride=1, padding=(1, 0, 0)),
            nn.ELU(self.alpha, True),

            nn.BatchNorm3d(self.n_ch_down2x),
            nn.Conv3d(self.n_ch_down2x, self.n_ch_down4x, 3, stride=2, padding=1),
            nn.ELU(self.alpha, True),

            nn.BatchNorm3d(self.n_ch_down4x),
            nn.Conv3d(self.n_ch_down4x, self.n_ch_down4x, 3, stride=1, padding=(1, 0, 0)),
            nn.ELU(self.alpha, True),
            nn.BatchNorm3d(self.n_ch_down4x),
            nn.Conv3d(self.n_ch_down4x, self.n_ch_down4x, 3, stride=1, padding=(1, 0, 0)),
            nn.ELU(self.alpha, True),

            nn.BatchNorm3d(self.n_ch_down4x),
            nn.Conv3d(self.n_ch_down4x, self.n_ch_down8x, 3, stride=2, padding=0),
            nn.ELU(self.alpha, True),

            nn.BatchNorm3d(self.n_ch_down8x),
            nn.Conv3d(self.n_ch_down8x, self.n_ch_fc, 3, stride=1, padding=0),
            nn.ELU(self.alpha, True),
        )

        self.convex_combine = nn.Sequential(
            nn.Linear(self.n_ch_global, 1),
            nn.ELU(self.alpha, True)
        )

    def _interpolate(self, latent_core, latent_penu, step):
        assert step is not None, 'Step must be given for interpolation!'
        if latent_core is None or latent_penu is None:
            return None
        core_to_penumbra = latent_penu - latent_core
        results = []
        for batch_sample in range(step.size()[0]):
            results.append(
                (latent_core[batch_sample, :, :, :, :] +
                 step[batch_sample, :, :, :, :] * core_to_penumbra[batch_sample, :, :, :, :]).unsqueeze(0)
            )
        return torch.cat(results, dim=0)

    def _forward_single(self, input_image):
        if input_image is None:
            return None
        return self.encoder(input_image)

    def _get_step(self, dto: CaeDto):
        step = dto.given_variables.time_to_treatment
        return step

    def forward(self, dto: CaeDto):
        step = self._get_step(dto)
        if dto.mode == CaeDtoUtil.MODE_GTRUTH or dto.mode == CaeDtoUtil.MODE_DEFAULT:
            assert dto.latents.gtruth._is_empty()  # Don't accidentally overwrite other results by code mistakes
            dto.latents.gtruth.core = self._forward_single(dto.given_variables.gtruth.core)
            dto.latents.gtruth.penu = self._forward_single(dto.given_variables.gtruth.penu)
            dto.latents.gtruth.lesion = self._forward_single(dto.given_variables.gtruth.lesion)
            dto.latents.gtruth.interpolation = self._interpolate(dto.latents.gtruth.core,
                                                                 dto.latents.gtruth.penu,
                                                                 step)
        if dto.mode == CaeDtoUtil.MODE_INPUTS or dto.mode == CaeDtoUtil.MODE_DEFAULT:
            assert dto.latents.inputs._is_empty()  # Don't accidentally overwrite other results by code mistakes
            dto.latents.inputs.core = self._forward_single(dto.given_variables.inputs.core)
            dto.latents.inputs.penu = self._forward_single(dto.given_variables.inputs.penu)
            dto.latents.inputs.interpolation = self._interpolate(dto.latents.inputs.core,
                                                                 dto.latents.inputs.penu,
                                                                 step)
        return dto


class Enc3DCtp(Enc3D):
    def __init__(self, size_input_xy, size_input_z, channels, n_ch_global, alpha, padding):
        Enc3D.__init__(self, size_input_xy, size_input_z, channels, n_ch_global, alpha)
        assert channels[0] > 2, 'At least 3 channels required to process input'
        self._padding = padding

    def forward(self, dto: CaeDto):
        step = self._get_step(dto)
        cbv = dto.given_variables.inputs.core[:, :, self._padding[0]:-self._padding[0],
                                                    self._padding[1]:-self._padding[1],
                                                    self._padding[2]:-self._padding[2]]
        ttd = dto.given_variables.inputs.penu[:, :, self._padding[0]:-self._padding[0],
                                                    self._padding[1]:-self._padding[1],
                                                    self._padding[2]:-self._padding[2]]
        if dto.mode == CaeDtoUtil.MODE_GTRUTH or dto.mode == CaeDtoUtil.MODE_DEFAULT:
            cat_core = torch.cat((dto.given_variables.gtruth.core, cbv, ttd), dim=data.DIM_CHANNEL_TORCH3D_5)
            cat_penu = torch.cat((dto.given_variables.gtruth.penu, cbv, ttd), dim=data.DIM_CHANNEL_TORCH3D_5)
            cat_lesion = torch.cat((dto.given_variables.gtruth.lesion, cbv, ttd), dim=data.DIM_CHANNEL_TORCH3D_5)
            dto.latents.gtruth.core = self._forward_single(cat_core)
            dto.latents.gtruth.penu = self._forward_single(cat_penu)
            dto.latents.gtruth.lesion = self._forward_single(cat_lesion)
            dto.latents.gtruth.interpolation = self._interpolate(dto.latents.gtruth.core,
                                                                 dto.latents.gtruth.penu,
                                                                 step)
        return dto


class Dec3D(CaeBase):
    def __init__(self, size_input_xy, size_input_z, channels, n_ch_global, alpha):
        super().__init__(size_input_xy, size_input_z, channels, n_ch_global, alpha, inner_xy=10, inner_z=3)

        self.decoder = nn.Sequential(
            nn.BatchNorm3d(self.n_ch_fc),
            nn.ConvTranspose3d(self.n_ch_fc, self.n_ch_down8x, 3, stride=1, padding=0, output_padding=0),
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

    def _forward_single(self, input_latent):
        if input_latent is None:
            return None
        return self.decoder(input_latent)

    def forward(self, dto: CaeDto):
        if dto.mode == CaeDtoUtil.MODE_GTRUTH or dto.mode == CaeDtoUtil.MODE_DEFAULT:
            assert dto.reconstructions.gtruth._is_empty()  # Don't accidentally overwrite other results by code mistakes
            dto.reconstructions.gtruth.core = self._forward_single(dto.latents.gtruth.core)
            dto.reconstructions.gtruth.penu = self._forward_single(dto.latents.gtruth.penu)
            dto.reconstructions.gtruth.lesion = self._forward_single(dto.latents.gtruth.lesion)
            dto.reconstructions.gtruth.interpolation = self._forward_single(dto.latents.gtruth.interpolation)
        if dto.mode == CaeDtoUtil.MODE_INPUTS or dto.mode == CaeDtoUtil.MODE_DEFAULT:
            assert dto.reconstructions.inputs._is_empty()  # Don't accidentally overwrite other results by code mistakes
            dto.reconstructions.inputs.core = self._forward_single(dto.latents.inputs.core)
            dto.reconstructions.inputs.penu = self._forward_single(dto.latents.inputs.penu)
            dto.reconstructions.inputs.interpolation = self._forward_single(dto.latents.inputs.interpolation)
        return dto


class Cae3D(nn.Module):
    def __init__(self, enc: Enc3D, dec: Dec3D):
        super().__init__()
        self.enc = enc
        self.dec = dec

    def forward(self, dto: CaeDto):
        dto = self.enc(dto)
        dto = self.dec(dto)
        return dto

    def freeze(self, freeze: bool):
        self.enc.freeze(freeze)
        self.dec.freeze(freeze)


class Cae3DCtp(Cae3D):
    def __init__(self, enc: Enc3DCtp, dec: Dec3D):
        Cae3D.__init__(self, enc, dec)