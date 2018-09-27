from learner.CaeReconstructionStepLearner import CaeReconstructionStepLearner
from common.inference.CaeMcInference import CaeMcInference
import common.dto.MetricMeasuresDto as MetricMeasuresDtoInit
from common.dto.CaeDto import CaeDto
from common import data, metrics


class CaeMcReconstructionStepLearner(CaeReconstructionStepLearner, CaeMcInference):
    """ A Learner to learn best interpolation steps for the
    reconstruction shape space. Uses CaeDto data transfer objects.
    """
    FN_VIS_BASE = '_cae1MCstep_'
    FNB_MARKS = '_cae1MCstep'

    def __init__(self, dataloader_training, dataloader_validation, cae_model, optimizer, scheduler, n_epochs,
                 path_previous_base, path_outputs_base, criterion, normalization_hours_penumbra=10, init_inputs=False):
        CaeReconstructionStepLearner.__init__(self, dataloader_training, dataloader_validation, cae_model, optimizer,
                                              scheduler, n_epochs, path_previous_base, path_outputs_base, criterion,
                                              normalization_hours_penumbra)
        CaeMcInference.__init__(self, cae_model, normalization_hours_penumbra=normalization_hours_penumbra,
                                init_inputs=init_inputs)

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
