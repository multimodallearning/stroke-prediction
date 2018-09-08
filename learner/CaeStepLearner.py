from learner.CaeReconstructionLearner import CaeReconstructionLearner
from common.dto.CaeDto import CaeDto
from torch.autograd import Variable
import torch


class CaeStepLearner(CaeReconstructionLearner):
    """ A Learner to learn best interpolation steps for the
    reconstruction shape space. Uses CaeDto data transfer objects.
    """
    FN_VIS_BASE = '_cae1step_'
    FNB_MARKS = '_cae1step'
    N_EPOCHS_ADAPT_BETA1 = 4

    def loss_step(self, dto: CaeDto, epoch):
        loss = 0.0
        divd = 2
        diff_penu_fuct = dto.reconstructions.gtruth.penu - dto.reconstructions.gtruth.interpolation
        loss += 1 * torch.mean(torch.abs(diff_penu_fuct) - diff_penu_fuct)
        loss += 1 * self._criterion(dto.reconstructions.gtruth.interpolation, dto.given_variables.gtruth.lesion)
        return loss / divd

    def get_time_to_treatment(self, batch, global_variables, step):
        normalization = self._get_normalization(batch)
        if step is None:
            time_to_treatment = None
        else:
            time_to_treatment = Variable((step * torch.ones(global_variables.size()[0], 1)) / normalization).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        return time_to_treatment