from common.model.Cae3D import Cae3D
from common.inference.Inference import Inference
from torch.autograd import Variable
import common.dto.CaeDto as CaeDtoUtil
import torch
from common import data
from common.dto.CaeDto import CaeDto


class CaeInference(Inference):
    """Common inference for training and testing,
    i.e. feed-forward of CAE
    """
    def __init__(self, model:Cae3D, normalization_hours_penumbra = 10):
        Inference.__init__(self, model)
        self._normalization_hours_penumbra = normalization_hours_penumbra

    def _get_normalization(self, batch):
        to_to_ta = batch[data.KEY_GLOBAL][:, 0, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5).type(torch.FloatTensor)
        normalization = torch.ones(to_to_ta.size()[0], 1).type(torch.FloatTensor) * \
                        self._normalization_hours_penumbra - to_to_ta.squeeze().unsqueeze(data.DIM_CHANNEL_TORCH3D_5)
        return normalization

    def get_time_to_treatment(self, batch, global_variables, step):
        normalization = self._get_normalization(batch)
        if step is None:
            ta_to_tr = batch[data.KEY_GLOBAL][:, 1, :, :, :].squeeze().unsqueeze(data.DIM_CHANNEL_TORCH3D_5)
            time_to_treatment = Variable(ta_to_tr.type(torch.FloatTensor) / normalization)
        else:
            time_to_treatment = Variable((step * torch.ones(global_variables.size()[0], 1)) / normalization)
        return time_to_treatment.unsqueeze(2).unsqueeze(3).unsqueeze(4)

    def init_clinical_variables(self, batch: dict, step):
        globals_incl_time = Variable(batch[data.KEY_GLOBAL].type(torch.FloatTensor))
        type_core = Variable(torch.zeros(globals_incl_time.size()[0], 1, 1, 1, 1))
        type_penumbra = Variable(torch.ones(globals_incl_time.size()[0], 1, 1, 1, 1))
        time_to_treatment = self.get_time_to_treatment(batch, globals_incl_time, step)

        if self.is_cuda:
            if time_to_treatment is not None:
                time_to_treatment = time_to_treatment.cuda()
            globals_incl_time = globals_incl_time.cuda()
            type_core = type_core.cuda()
            type_penumbra = type_penumbra.cuda()

        return CaeDtoUtil.init_dto(globals_incl_time, time_to_treatment,
                                   type_core, type_penumbra, None, None, None, None, None, None, None)

    def init_gtruth_segm_variables(self, batch: dict, dto: CaeDto):
        core_gt = Variable(batch[data.KEY_LABELS][:, 0, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5))
        penu_gt = Variable(batch[data.KEY_LABELS][:, 1, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5))
        lesion_gt = Variable(batch[data.KEY_LABELS][:, 2, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5))
        if self.is_cuda:
            core_gt = core_gt.cuda()
            penu_gt = penu_gt.cuda()
            lesion_gt = lesion_gt.cuda()
        dto.given_variables.gtruth.core = core_gt
        dto.given_variables.gtruth.penu = penu_gt
        dto.given_variables.gtruth.lesion = lesion_gt
        return dto

    def infer(self, dto: CaeDto):
        return self._model(dto)

    def inference_step(self, batch: dict, step=None):
        dto = self.init_clinical_variables(batch, step)
        dto.mode = CaeDtoUtil.FLAG_GTRUTH
        dto = self.init_gtruth_segm_variables(batch, dto)
        return self.infer(dto)
