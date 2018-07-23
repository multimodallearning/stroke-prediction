from common.Dto import Dto
from abc import abstractmethod
import torch
import numpy


class Learner:
    """Base class with a standard routine for
    a training procedure. The single steps can
    be overridden by subclasses to specify the
    procedures required for a specific training.
    """
    IMSHOW_VMAX_CBV = 12
    IMSHOW_VMAX_TTD = 40
    FN_VIS_BASE = '_samples_visualization_'

    def __init__(self, dataloader_training, dataloader_validation, model, path_model, optimizer, n_epochs,
                 path_outputs_base='/tmp/', metrics={'training': {'loss': []}, 'validate': {'loss': []}}, cuda=True):
        super().__init__()
        self._dataloader_training = dataloader_training
        self._dataloader_validation = dataloader_validation
        self._model = model
        self._path_model = path_model
        self._optimizer = optimizer
        self._n_epochs = n_epochs
        self._path_outputs_base = path_outputs_base
        self._metrics = metrics
        self._cuda = cuda

    @abstractmethod
    def inference_step(self, batch, epoch):
        pass

    @abstractmethod
    def loss_step(self, dto: Dto, epoch):
        pass

    def train_batch(self, batch, epoch, running_epoch_metrics):
        dto = self.inference_step(batch, epoch)
        loss = self.loss_step(dto, epoch)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        running_epoch_metrics['loss'].append(loss.squeeze().cpu().data.numpy()[0])
        running_epoch_metrics = self.metrics_step(dto, epoch, running_epoch_metrics)

        del loss
        del dto

        return running_epoch_metrics

    def validate_batch(self, batch, epoch, running_epoch_metrics):
        dto = self.inference_step(batch, epoch)
        loss = self.loss_step(dto, epoch)

        running_epoch_metrics['loss'].append(loss.squeeze().cpu().data.numpy()[0])
        running_epoch_metrics = self.metrics_step(dto, epoch, running_epoch_metrics)

        del loss
        del dto

        return running_epoch_metrics

    def metrics_step(self, dto: Dto, epoch, running_epoch_metrics):
        return running_epoch_metrics

    def print_epoch(self, epoch, phase, epoch_metrics):
        pass

    def plot_epoch(self, epoch):
        pass

    def visualize_epoch(self, epoch):
        pass

    def adapt_lr(self, epoch):
        pass

    def run_training(self):
        minloss = numpy.Inf

        for epoch in range(self._n_epochs):
            self.adapt_lr(epoch)

            # ---------------------------- (1) TRAINING ---------------------------- #

            running_epoch_metrics = {}
            for key in self._metrics['training'].keys():
                 running_epoch_metrics[key] = []

            self._model.train()

            for batch in self._dataloader_training:
                running_epoch_metrics = self.train_batch(batch, epoch, running_epoch_metrics)

            for metric in running_epoch_metrics.keys():
                self._metrics['training'][metric].append(numpy.mean(running_epoch_metrics[metric]))

            self.print_epoch(epoch, 'training', self._metrics['training'])
            del running_epoch_metrics
            del batch

            # ---------------------------- (2) VALIDATE ---------------------------- #

            running_epoch_metrics = {}
            for key in self._metrics['validate'].keys():
                running_epoch_metrics[key] = []

            self._model.eval()

            for batch in self._dataloader_validation:
                running_epoch_metrics = self.validate_batch(batch, epoch, running_epoch_metrics)

            for metric in running_epoch_metrics.keys():
                self._metrics['validate'][metric].append(numpy.mean(running_epoch_metrics[metric]))

            self.print_epoch(epoch, 'validate', self._metrics['validate'])
            del running_epoch_metrics
            del batch

            # ------------ (3) SAVE MODEL / VISUALIZE (if new optimum) ------------ #

            if self._metrics['validate']['loss'] and self._metrics['validate']['loss'][-1] < minloss:
                minloss = self._metrics['validate']['loss'][-1]
                torch.save(self._model.state_dict(), self._path_model)
                self.visualize_epoch(epoch)

            # ----------------------------- (4) PLOT ----------------------------- #

            self.plot_epoch(epoch)


