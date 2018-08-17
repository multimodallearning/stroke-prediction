from abc import abstractmethod


class Inference():
    """Base class for all classes that require model inference.
    """
    IMSHOW_VMAX_CBV = 12
    IMSHOW_VMAX_TTD = 40
    FN_VIS_BASE = '_visual_'
    INFERENCE_INITALIZED = False

    @abstractmethod
    def __init__(self, model):
        if not self.INFERENCE_INITALIZED:
            self._model = model
            self.INFERENCE_INITALIZED = True

    @abstractmethod
    def inference_step(self, batch: dict):
        pass

    @property
    def is_cuda(self) -> bool:
        return next(self._model.parameters()).is_cuda
