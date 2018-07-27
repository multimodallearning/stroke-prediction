from abc import abstractmethod
from common.dto.Dto import Dto
from common.inference.Inference import Inference
from common.dto.MetricMeasuresDto import MetricMeasuresDto
import common.dto.MetricMeasuresDto as MetricMeasuresDtoInit
import torch
import numpy


class Learner(Inference):
    """Base class with a standard routine for
    a training procedure. The single steps can
    be overridden by subclasses to specify the
    procedures required for a specific training.
    """
    IMSHOW_VMAX_CBV = 12
    IMSHOW_VMAX_TTD = 40
    FN_VIS_BASE = '_samples_visualization_'

    def __init__(self, dataloader_training, dataloader_validation, model, path_model, optimizer, n_epochs,
                 path_outputs_base='/tmp/', cuda=True):
        Inference.__init__(self, model, path_model, path_outputs_base, cuda)
        self._dataloader_training = dataloader_training
        self._dataloader_validation = dataloader_validation
        self._optimizer = optimizer
        self._n_epochs = n_epochs
        self._metrics = {'training': None, 'validate': None}

    @abstractmethod
    def loss_step(self, dto: Dto, epoch):
        pass

    def train_batch(self, batch: dict, epoch) -> MetricMeasuresDto:
        dto = self.inference_step(batch)
        loss = self.loss_step(dto, epoch)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        batch_metrics = self.batch_metrics_step(dto, epoch)
        batch_metrics.loss = loss.squeeze().cpu().data.numpy()[0]

        del loss
        del dto

        return batch_metrics

    def validate_batch(self, batch: dict, epoch) -> MetricMeasuresDto:
        dto = self.inference_step(batch)
        loss = self.loss_step(dto, epoch)

        batch_metrics = self.batch_metrics_step(dto, epoch)
        batch_metrics.loss = loss.squeeze().cpu().data.numpy()[0]

        del loss
        del dto

        return batch_metrics

    def batch_metrics_step(self, dto: Dto, epoch) -> MetricMeasuresDto:
        return MetricMeasuresDtoInit.init_dto()

    def print_epoch(self, epoch, phase, epoch_metrics: MetricMeasuresDto):
        pass

    def plot_epoch(self, epoch):
        pass

    def visualize_epoch(self, epoch):
        pass

    def adapt_lr(self, epoch):
        pass

    def _loop_tru(self):
        for key in sorted(self.__dict__.keys()):
            txt = '[ ]'
            val = self.__dict__[key]
            if val is not None:
                txt = '[x]'
            result += indent + txt + ' ' + key + '\n'
            if isinstance(val, Dto):
                result += val.__repr__(indent=(indent + '    '))
        return result


    def run_training(self):
        minloss = numpy.Inf

        for epoch in range(self._n_epochs):
            self.adapt_lr(epoch)

            # ---------------------------- (1) TRAINING ---------------------------- #

            self._model.train()

            epoch_metrics = MetricMeasuresDtoInit.init_dto()
            for batch in self._dataloader_training:
                batch_metrics = self.train_batch(batch, epoch)
                epoch_metrics.add(batch_metrics)
            epoch_metrics.normalize()

            self.print_epoch(epoch, 'training', epoch_metrics)
            del epoch_metrics
            del batch

            # ---------------------------- (2) VALIDATE ---------------------------- #

            self._model.eval()

            epoch_metrics = MetricMeasuresDtoInit.init_dto()
            for batch in self._dataloader_validation:
                epoch_metrics.append(self.validate_batch(batch, epoch))
                epoch_metrics.add(batch_metrics)
            epoch_metrics.normalize()

            self.print_epoch(epoch, 'validate', epoch_metrics)
            del epoch_metrics
            del batch

            # ------------ (3) SAVE MODEL / VISUALIZE (if new optimum) ------------ #

            if self._metrics['validate']['loss'] and self._metrics['validate']['loss'][-1] < minloss:
                minloss = self._metrics['validate']['loss'][-1]
                torch.save(self._model.state_dict(), self._path_model)
                self.visualize_epoch(epoch)

            # ----------------------------- (4) PLOT ----------------------------- #

            self.plot_epoch(epoch)


