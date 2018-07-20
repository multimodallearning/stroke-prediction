from common.Dto import Dto
from abc import abstractmethod
import torch


class Tester:
    """Base class with a standard routine for
    a testing procedure. The single steps can
    be overridden by subclasses to specify the
    procedures required for a specific test run.
    """

    def __init__(self, dataloader, model, path_model, path_outputs_base='/tmp/', metrics={}):
        super().__init__()
        assert dataloader.batch_size == 1, 'Metrics need to be computed on batch size = 1'
        self._dataloader = dataloader
        self._path_outputs_base = path_outputs_base
        self._metrics = metrics
        self._model = model
        self._model.load_state_dict(torch.load(self._path_model))
        for p in self._model.parameters():
            p.requires_grad = False

    @abstractmethod
    def inference_step(self, batch):
        pass

    def infer_batch(self, batch, metrics):
        dto = self.inference_step(batch)
        metrics = self.metrics_step(dto, metrics)
        return metrics

    def metrics_step(self, dto: Dto, metrics):
        return metrics

    def print_inference(self, phase):
        pass

    def run_inference(self):
        self._model.eval()

        for batch in self._dataloader:
            self._metrics = self.infer_batch(batch, self._metrics)

        self.print_inference(self._metrics)

        del batch
