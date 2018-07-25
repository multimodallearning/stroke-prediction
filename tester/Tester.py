from common.dto.Dto import Dto
from common.inference.Inference import Inference
import torch


class Tester(Inference):
    """Base class with a standard routine for
    a testing procedure. The single steps can
    be overridden by subclasses to specify the
    procedures required for a specific test run.
    """

    def __init__(self, dataloader, model, path_model, path_outputs_base='/tmp/', metrics={}, cuda=True):
        Inference.__init__(self, model, path_model, path_outputs_base, cuda)
        self._dataloader = dataloader
        self._path_outputs_base = path_outputs_base
        self._metrics = metrics
        self._model = model
        self._model.load_state_dict(torch.load(path_model))
        for p in self._model.parameters():
            p.requires_grad = False

    def infer_batch(self, batch, metrics):
        dto = self.inference_step(batch)
        metrics = self.metrics_step(dto, metrics)
        self.save_inference(dto, batch)
        return metrics

    def metrics_step(self, dto: Dto, metrics):
        return metrics

    def save_inference(self, dto: Dto, batch):
        pass

    def print_inference(self, batch, metrics):
        pass

    def run_inference(self):
        self._model.eval()

        for batch in self._dataloader:
            self._metrics = self.infer_batch(batch, self._metrics)
            self.print_inference(batch, self._metrics)

        del batch
