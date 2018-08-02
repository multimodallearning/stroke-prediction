from abc import abstractmethod


class Inference():
    """Base class for all classes that require model inference.
    """

    @abstractmethod
    def __init__(self, model, path_model, path_outputs_base):
        self._model = model
        self._path_model = path_model
        self._path_outputs_base = path_outputs_base

    @abstractmethod
    def inference_step(self, batch: dict):
        pass

    @property
    def is_cuda(self) -> bool:
        return next(self._model.parameters()).is_cuda
