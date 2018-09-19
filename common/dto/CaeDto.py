from common.dto.Dto import Dto

FLAG_DEFAULT = 'default'
FLAG_GTRUTH = 'gtruth'
FLAG_INPUTS = 'inputs'

class CaeDto(Dto):
    """ DTO for CAE usage.
    """
    def __init__(self, given_variables: Dto, latents: Dto, reconstructions: Dto):
        super().__init__()
        self.given_variables = given_variables
        self.reconstructions = latents
        self.latents = reconstructions
        self.flag = FLAG_DEFAULT


def init_dto(global_variables, time_to_treatment, type_core, type_penumbra, inputs_core, inputs_penu,
             gtruth_core, gtruth_penumbra, gtruth_lesion, inputs_cbv, inputs_ttd):
    """
    Inits a CaeDto with the given variables.
    :param global_variables:    global clinical scalar variables, such as age et cetera
    :param time_to_treatment:   global clinical scalar variable time_to_treatment
    :param type_core:           aux value to represent core
    :param type_penumbra:       aux value to represent penumbra
    :param inputs_core:         input data for core segmentation from Unet
    :param inputs_penu:         input data for penu segmentation from Unet
    :param inputs_cbv:          input data CTP CBV image
    :param inputs_ttd:          input data CTP TTD image
    :param gtruth_core:         manual segmentation mask for core
    :param gtruth_penumbra:     manual segmentation mask for penumbra
    :param gtruth_lesion:       manual segmentation mask for follow-up lesion
    :return: CaeDto
    """

    given_variables = Dto(globals=global_variables,
                          time_to_treatment=time_to_treatment,
                          scalar_types=Dto(core=type_core, penu=type_penumbra),
                          inputs=Dto(core=inputs_core, penu=inputs_penu, cbv=inputs_cbv, ttd=inputs_ttd),
                          gtruth=Dto(core=gtruth_core, penu=gtruth_penumbra, lesion=gtruth_lesion))

    latents = Dto(inputs=Dto(core=None, penu=None, interpolation=None),
                  gtruth=Dto(core=None, penu=None, interpolation=None, lesion=None))

    reconstructions = Dto(inputs=Dto(core=None, penu=None, interpolation=None),
                          gtruth=Dto(core=None, penu=None, interpolation=None, lesion=None))

    return CaeDto(given_variables=given_variables, latents=latents, reconstructions=reconstructions)
