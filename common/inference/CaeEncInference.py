from common.model.Cae3D import Cae3D, Enc3D
from common.inference.CaeInference import CaeInference
from common.dto.CaeDto import CaeDto
import common.dto.CaeDto as CaeDtoUtil
from torch.autograd import Variable
from common import data


class CaeEncInference(CaeInference):
    """Common inference for training and testing,
    i.e. feed-forward of CAE and the previous Encoder
    """
    def __init__(self, model:Cae3D, new_enc:Enc3D, path_model, path_outputs_base='/tmp/',
                 normalization_hours_penumbra = 10):
        CaeInference.__init__(self, model, path_model, path_outputs_base, normalization_hours_penumbra)
        self._new_enc = new_enc

    def infer(self, dto: CaeDto):
        pass

    def init_unet_segm_variables(self, batch: dict, dto: CaeDto):
        unet_core = Variable(batch[data.KEY_IMAGES][:, 0, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5))
        unet_penu = Variable(batch[data.KEY_IMAGES][:, 1, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5))
        if self.is_cuda:
            unet_core = unet_core.cuda()
            unet_penu = unet_penu.cuda()
        dto.given_variables.inputs.core = unet_core
        dto.given_variables.inputs.penu = unet_penu
        return dto

    def inference_step(self, batch: dict, step=None):
        dto = self.init_clinical_variables(batch, step)

        dto.mode = CaeDtoUtil.MODE_INPUTS
        dto = self.init_unet_segm_variables(batch, dto)
        dto = self._new_enc(dto)

        dto.mode = CaeDtoUtil.MODE_GTRUTH
        dto = self.init_gtruth_segm_variables(batch, dto)
        dto = self._model(dto)

        return dto
