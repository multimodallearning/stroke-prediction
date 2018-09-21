import torch
import common.data as data
from common.dto.CaeDto import CaeDto
import common.dto.CaeDto as CaeDtoUtil
from common.model.CaeElemInt3D import CaeBase as Caebase
from common.model.CaeElemInt3D import Enc3D as Enc3Dbase


MC = 4
GLOBAL = 5


class CaeBase(Caebase):

    def __init__(self, size_input_xy=128, size_input_z=28, channels=[MC, 16, 32, 64, 128, 1024, 128, MC],
                 n_ch_global=GLOBAL, inner_xy=12, inner_z=3):
        super().__init__(size_input_xy, size_input_z, channels, n_ch_global, inner_xy, inner_z)


class Enc3D(Enc3Dbase):
    def forward(self, dto: CaeDto):
        mc = torch.cat((dto.given_variables.gtruth.penu,
                        dto.given_variables.inputs.cbv,
                        dto.given_variables.inputs.ttd), dim=data.DIM_CHANNEL_TORCH3D_5)

        if dto.flag == CaeDtoUtil.FLAG_GTRUTH or dto.flag == CaeDtoUtil.FLAG_DEFAULT:
            assert dto.latents.gtruth._is_empty()  # Don't accidentally overwrite other results by code mistakes
            dto.latents.gtruth.lesion, _ = self._forward_single(torch.cat((dto.given_variables.gtruth.lesion, mc),
                                                                          dim=data.DIM_CHANNEL_TORCH3D_5))
            dto.latents.gtruth.core, linear_core = self._forward_single(torch.cat((dto.given_variables.gtruth.core, mc),
                                                                          dim=data.DIM_CHANNEL_TORCH3D_5))
            dto.latents.gtruth.penu, linear_penu = self._forward_single(torch.cat((dto.given_variables.gtruth.penu, mc),
                                                                          dim=data.DIM_CHANNEL_TORCH3D_5))
            step = self._get_step(dto, linear_core, linear_penu)
            dto.latents.gtruth.interpolation = self._interpolate(dto.latents.gtruth.core,
                                                                 dto.latents.gtruth.penu,
                                                                 step)
        return dto