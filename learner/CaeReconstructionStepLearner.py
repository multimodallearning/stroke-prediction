from learner.CaeReconstructionLearner import CaeReconstructionLearner
from torch.autograd import Variable
import torch


class CaeReconstructionStepLearner(CaeReconstructionLearner):
    """ A Learner to learn best interpolation steps for the
    reconstruction shape space. Uses CaeDto data transfer objects.
    """
    FN_VIS_BASE = '_cae1step_'
    FNB_MARKS = '_cae1step'

    def get_time_to_treatment(self, batch, global_variables, step):
        normalization = self._get_normalization(batch)
        if step is None:
            time_to_treatment = None
        else:
            time_to_treatment = Variable((step * torch.ones(global_variables.size()[0], 1)) / normalization).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        return time_to_treatment