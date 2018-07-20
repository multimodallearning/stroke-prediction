from common.Dto import Dto


class UnetDto(Dto):
    """ DTO for Unet usage.
    """
    def __init__(self, given_variables: Dto, outputs: Dto):
        super().__init__()
        self.given_variables = given_variables
        self.outputs = outputs


def init_unet_dto(input_modalities, gtruth_core=None, gtruth_penumbra=None, gtruth_lesion=None):
    """
    Inits a Unet_Dto with the given variables.
    :param input_modalities:    CTP input modalities
    :param gtruth_core:         manual segmentation mask for core
    :param gtruth_penumbra:     manual segmentation mask for penumbra
    :param gtruth_lesion:       manual segmentation mask for follow-up lesion
    :return: Unet_Dto
    """

    given_variables = Dto(input_modalities=input_modalities, core=gtruth_core, penu=gtruth_penumbra,
                          lesion=gtruth_lesion)

    outputs = Dto(core=None, penu=None, lesion=None)

    return UnetDto(given_variables=given_variables, outputs=outputs)
