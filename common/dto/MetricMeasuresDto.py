import numpy
from common.dto.Dto import Dto


class MeasuresDto(Dto):
    def add(self, other):
        if isinstance(other, type(self)):
            for attr, value in other:
                if self.__dict__[attr] is None:
                    self.__dict__[attr] = value
                elif isinstance(value, MeasuresDto):
                    self.__dict__[attr].add(value)
                else:
                    self.__dict__[attr] += value
        else:
            raise Exception('A' + str(type(self)) + 'must be added')

    def div(self, divisor):
        for attr, value in self.__dict__.items():
            if value is not None and value != numpy.Inf:
                if isinstance(value, MeasuresDto):
                    self.__dict__[attr].div(divisor)
                else:
                    self.__dict__[attr] = value / divisor


class BinaryMeasuresDto(MeasuresDto):
    """ DTO for the metric measures on binary images.
    """
    def __init__(self, dc, hd, assd, sensitivity, specificity):
        super().__init__()
        self.dc = dc
        self.hd = hd
        self.assd = assd
        self.sensitivity = sensitivity
        self.specificity = specificity


class MetricMeasuresDto(MeasuresDto):
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
             lesion_dc=None, lesion_hd=None, lesion_assd=None,
             lesion_sensitivity=None, lesion_specificity=None):
    """
    Inits a MetricMeasuresDto with the evaluation measures.
    :return: MetricMeasuresDto
    """

    core = BinaryMeasuresDto(core_dc, core_hd, core_assd, None, None)
    penu = BinaryMeasuresDto(penu_dc, penu_hd, penu_assd, None, None)
    lesion = BinaryMeasuresDto(lesion_dc, lesion_hd, lesion_assd, lesion_sensitivity, lesion_specificity)

    return MetricMeasuresDto(loss, core, penu, lesion)
