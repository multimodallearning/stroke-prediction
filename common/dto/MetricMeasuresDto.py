from common.dto.Dto import Dto


class BinaryMeasuresDto(Dto):
    """ DTO for the metric measures on binary images.
    """
    def __init__(self, dc=None, hd=None, assd=None):
        super().__init__()
        self.dc = dc
        self.hd = hd
        self.assd = assd


class MetricMeasuresDto(Dto):
    """ DTO for the metric measures.
    """
    def __init__(self, loss, core:BinaryMeasuresDto, penu, lesion):
        super().__init__()
        self.loss = loss
        self.core = core
        self.penu = penu
        self.lesion = lesion


def init_dto(loss=None, core_dc=None):
    """
    Inits a MetricMeasuresDto with the given variables.
    :return: MetricMeasuresDto
    """

    core = BinaryMeasuresDto(core_dc,)

    return MetricMeasuresDto(loss, core, penu, lesion)
