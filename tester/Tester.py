from common.dto.Dto import Dto
from common.inference.Inference import Inference
from common.dto.MetricMeasuresDto import MetricMeasuresDto
import common.dto.MetricMeasuresDto as MetricMeasuresDtoInit
from torch.utils.data import DataLoader
import torch


class Tester(Inference):
    """Base class with a standard routine for
    a testing procedure. The single steps can
    be overridden by subclasses to specify the
    procedures required for a specific test run.
    """

    def __init__(self, dataloader: DataLoader, path_model: str, path_outputs_base: str='/tmp/'):
        Inference.__init__(self, torch.load(path_model))
        assert dataloader.batch_size == 1, "You must ensure a batch size of 1 for correct case metric measures."
        self._dataloader = dataloader
        self._path_outputs_base = path_outputs_base
        self.model.freeze(True)
        self.model.eval()

    def infer_batch(self, batch: dict):
        dto = self.inference_step(batch)
        batch_metrics = self.batch_metrics_step(dto)
        self.save_inference(dto, batch)
        return batch_metrics, dto

    def batch_metrics_step(self, dto: Dto):
        return MetricMeasuresDtoInit.init_dto()

    def _fn(self, case_id, type, suffix):
        return self._path_outputs_base + '_' + str(case_id) + str(type) + str(suffix) + '.nii.gz'

    def save_inference(self, dto: Dto, batch: dict):
        pass

    def print_inference(self, batch: dict, metrics: MetricMeasuresDto, dto: Dto = None):
        pass

    def run_inference(self):
        for batch in self._dataloader:
            batch_metrics, dto = self.infer_batch(batch)
            self.print_inference(batch, batch_metrics, dto)
