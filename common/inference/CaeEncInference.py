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
    def __init__(self, model:Cae3D, new_enc:Enc3D, normalization_hours_penumbra = 10):
        CaeInference.__init__(self, model, normalization_hours_penumbra, init_ctp=False, init_inputs=True)
        self._new_enc = new_enc

    def infer(self, dto: CaeDto):
        pass

    def inference_step(self, batch: dict, step=None):
        ''' super()
        def inference_step(self, batch: dict, step):
            dto = self.init_dto(batch, step)
            return self.infer(dto)
        '''

        dto = self._init_clinical_variables(batch, step)

        dto.mode = CaeDtoUtil.FLAG_INPUTS
        dto = self.init_unet_segm_variables(batch, dto)
        dto = self._new_enc(dto)
        dto = self.model.dec(dto)

        dto.mode = CaeDtoUtil.FLAG_GTRUTH
        dto = self._init_gtruth_segm_variables(batch, dto)
        dto = self.model(dto)

        return dto
