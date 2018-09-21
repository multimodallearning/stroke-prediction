from learner.CaeReconstructionStepLearner import CaeReconstructionStepLearner
import common.dto.MetricMeasuresDto as MetricMeasuresDtoInit
from common.dto.CaeDto import CaeDto
import common.dto.CaeDto as CaeDtoUtil
from torch.autograd import Variable
from common import data, metrics, util
import matplotlib.pyplot as plt


class CaeMultiChannelReconstructionStepLearner(CaeReconstructionStepLearner):
    """ A Learner to learn best interpolation steps for the
    reconstruction shape space. Uses CaeDto data transfer objects.
    """
    FN_VIS_BASE = '_cae1MCstep_'
    FNB_MARKS = '_cae1MCstep'

    def init_perfusion_variables(self, batch: dict, dto: CaeDto):
        cbv = Variable(batch[data.KEY_IMAGES][:, 0, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5))
        ttd = Variable(batch[data.KEY_IMAGES][:, 1, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5))
        if self.is_cuda:
            cbv = cbv.cuda()
            ttd = ttd.cuda()
        dto.given_variables.inputs.cbv = cbv
        dto.given_variables.inputs.ttd = ttd
        return dto

    def inference_step(self, batch: dict, step=None):
        dto = self.init_clinical_variables(batch, step)
        dto = self.init_perfusion_variables(batch, dto)
        dto.mode = CaeDtoUtil.FLAG_GTRUTH
        dto = self.init_gtruth_segm_variables(batch, dto)
        return self.infer(dto)

    def batch_metrics_step(self, dto: CaeDto, epoch):
        batch_metrics = MetricMeasuresDtoInit.init_dto()
        lesion_result = dto.reconstructions.gtruth.interpolation[:, 0, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5)
        batch_metrics.lesion = metrics.binary_measures_torch(lesion_result,
                                                             dto.given_variables.gtruth.lesion, self.is_cuda)
        core_result = dto.reconstructions.gtruth.core[:, 0, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5)
        batch_metrics.core = metrics.binary_measures_torch(core_result,
                                                           dto.given_variables.gtruth.core, self.is_cuda)
        penu_result = dto.reconstructions.gtruth.penu[:, 0, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5)
        batch_metrics.penu = metrics.binary_measures_torch(penu_result,
                                                           dto.given_variables.gtruth.penu, self.is_cuda)
        return batch_metrics
