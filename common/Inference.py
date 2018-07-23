from abc import abstractmethod


class Inference():
    """Base class for all classes that require model inference.
    """
    def __init__(self, model, path_model, path_outputs_base='/tmp/'):
        self._model = model
        self._path_model = path_model
        self._path_outputs_base = path_outputs_base

    @abstractmethod
    def inference_step(self, batch):
        pass