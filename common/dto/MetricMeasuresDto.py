from common.dto.Dto import Dto


class BinaryMeasuresDto(Dto):
    """ DTO for the metric measures on binary images.
    """
    def __init__(self, dc, hd, assd):
        super().__init__()
        self.dc = dc
        self.hd = hd
        self.assd = assd


class MetricMeasuresDto(Dto):
    """ DTO for all evaluation metric measures.
    """
    def __init__(self, loss, core:BinaryMeasuresDto, penu:BinaryMeasuresDto, lesion:BinaryMeasuresDto):
        super().__init__()
        self.loss = loss
        self.core = core
        self.penu = penu
        self.lesion = lesion


def init_dto(loss=None, core_dc=None, core_hd=None, core_assd=None,
             penu_dc=None, penu_hd=None, penu_assd=None,
             lesion_dc=None, lesion_hd=None, lesion_assd=None):
    """
    Inits a MetricMeasuresDto with the evaluation measures.
    :return: MetricMeasuresDto
    """

    core = BinaryMeasuresDto(core_dc, core_hd, core_assd)
    penu = BinaryMeasuresDto(penu_dc, penu_hd, penu_assd)
    lesion = BinaryMeasuresDto(lesion_dc, lesion_hd, lesion_assd)

    return MetricMeasuresDto(loss, core, penu, lesion)
