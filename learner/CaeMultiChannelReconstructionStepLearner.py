from learner.CaeReconstructionStepLearner import CaeReconstructionStepLearner
from common.dto.CaeDto import CaeDto
from torch.autograd import Variable
import torch


class CaeMultiChannelReconstructionStepLearner(CaeReconstructionStepLearner):
    """ A Learner to learn best interpolation steps for the
    reconstruction shape space. Uses CaeDto data transfer objects.
    """
    FN_VIS_BASE = '_cae1MCstep_'
    FNB_MARKS = '_cae1MCstep'

