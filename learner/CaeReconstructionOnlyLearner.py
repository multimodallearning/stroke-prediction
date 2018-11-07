from learner.CaeReconstructionLearner import CaeReconstructionLearner
from common.dto.CaeDto import CaeDto


class CaeReconstructionOnlyLearner(CaeReconstructionLearner):
    """ A Learner to learn best interpolation steps for the
    reconstruction shape space. Uses CaeDto data transfer objects.
    """
    FN_VIS_BASE = '_cae1recon_'
    FNB_MARKS = '_cae1recon'

    def loss_step(self, dto: CaeDto, epoch):
        return self._criterion(dto.reconstructions.gtruth.core, dto.given_variables.gtruth.core) +\
               self._criterion(dto.reconstructions.gtruth.penu, dto.given_variables.gtruth.penu) +\
               self._criterion(dto.reconstructions.gtruth.lesion, dto.given_variables.gtruth.lesion)
