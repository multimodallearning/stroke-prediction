from common.inference.CaeInference import CaeInference


class CaeMcInference(CaeInference):
    """Common inference for training and testing,
    i.e. feed-forward of CAE with multi-channel input
    """

    def __init__(self, model:Cae3D, normalization_hours_penumbra = 10, init_inputs=False):
        CaeInference.__init__(self, model, normalization_hours_penumbra, init_ctp=True, init_inputs=init_inputs)