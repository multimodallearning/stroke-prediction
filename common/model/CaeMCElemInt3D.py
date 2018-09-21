import torch
import torch.nn as nn
import common.data as data
from common.dto.CaeDto import CaeDto
import common.dto.CaeDto as CaeDtoUtil

from common.model.CaeElemInt3D import Enc3D as Enc3Dbase
from common.model.CaeElemInt3D import Dec3D as Dec3Dbase


class Enc3D(Enc3Dbase):
    def forward(self, dto: CaeDto):
        torch.cat()

        if dto.flag == CaeDtoUtil.FLAG_GTRUTH or dto.flag == CaeDtoUtil.FLAG_DEFAULT:
            assert dto.latents.gtruth._is_empty()  # Don't accidentally overwrite other results by code mistakes
            dto.latents.gtruth.lesion, _ = self._forward_single(dto.given_variables.gtruth.lesion)
            dto.latents.gtruth.core, linear_core = self._forward_single(dto.given_variables.gtruth.core)
            dto.latents.gtruth.penu, linear_penu = self._forward_single(dto.given_variables.gtruth.penu)
            step = self._get_step(dto, linear_core, linear_penu)
            dto.latents.gtruth.interpolation = self._interpolate(dto.latents.gtruth.core,
                                                                 dto.latents.gtruth.penu,
                                                                 step)
        return dto


class Dec3D(Dec3Dbase):
    def forward(self, dto: CaeDto):
        torch.cat()

        if dto.flag == CaeDtoUtil.FLAG_GTRUTH or dto.flag == CaeDtoUtil.FLAG_DEFAULT:
            assert dto.reconstructions.gtruth._is_empty()  # Don't accidentally overwrite other results by code mistakes
            dto.reconstructions.gtruth.core = self._forward_single(dto.latents.gtruth.core)
            dto.reconstructions.gtruth.penu = self._forward_single(dto.latents.gtruth.penu)
            dto.reconstructions.gtruth.lesion = self._forward_single(dto.latents.gtruth.lesion)
            dto.reconstructions.gtruth.interpolation = self._forward_single(dto.latents.gtruth.interpolation)
        if dto.flag == CaeDtoUtil.FLAG_INPUTS or dto.flag == CaeDtoUtil.FLAG_DEFAULT:
            assert dto.reconstructions.inputs._is_empty()  # Don't accidentally overwrite other results by code mistakes
            dto.reconstructions.inputs.core = self._forward_single(dto.latents.inputs.core)
            dto.reconstructions.inputs.penu = self._forward_single(dto.latents.inputs.penu)
            dto.reconstructions.inputs.interpolation = self._forward_single(dto.latents.inputs.interpolation)
        return dto
