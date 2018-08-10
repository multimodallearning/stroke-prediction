from abc import abstractmethod
from torch.utils.data import DataLoader
from common.dto.Dto import Dto
from common.inference.Inference import Inference
from common.dto.MetricMeasuresDto import MetricMeasuresDto
import common.dto.MetricMeasuresDto as MetricMeasuresDtoInit
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn import Module
import matplotlib.pyplot as plt
import torch
import numpy
import jsonpickle


class Learner(Inference):
    """Base class with a standard routine for
    a training procedure. The single steps can
    be overridden by subclasses to specify the
    procedures required for a specific training.
    """
    EXT_MODEL = '.model'
    EXT_OPTIM = '.optim'
    EXT_TRAIN = '.json'

    def __init__(self, dataloader_training: DataLoader, dataloader_validation: DataLoader, model: Module,
                 path_model: str, optimizer: Optimizer, scheduler: _LRScheduler, n_epochs: int,
                 path_training_metrics: str=None, path_outputs_base: str='/tmp/'):
        Inference.__init__(self, model, path_model, path_outputs_base)
        self._path_optim = self._path_model.replace(self.EXT_MODEL, self.EXT_OPTIM)
        self._path_train = self._path_model.replace(self.EXT_MODEL, self.EXT_TRAIN)
        assert dataloader_training.batch_size > 1, 'For normalization layers batch_size > 1 is required.'
        self._dataloader_training = dataloader_training
        self._dataloader_validation = dataloader_validation
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._n_epochs = n_epochs
        if path_training_metrics is None:
            self._metric_dtos = {'training': [], 'validate': []}
        else:
            if self.is_cuda:  # restore model weights from previous training
                self._model = torch.load(path_model).cuda()
            else:
                self._model = torch.load(path_model)
            self.load_training(path_training_metrics)  # restore training curves from previous training
            print('Continue training from files:', path_training_metrics, path_model, self._path_optim)
        assert len(self._metric_dtos['training']) == len(self._metric_dtos['validate']), 'Incomplete training data!'

    @abstractmethod
    def loss_step(self, dto: Dto, epoch):
        pass

    def get_start_epoch(self):
        return 0

    def get_start_min_loss(self):
        return numpy.Inf

    def load_training(self, path):
        print('Loading:', self._path_train, ',', self._optimizer)
        self._optimizer.load_state_dict(torch.load(self._path_optim))
        with open(path, 'r') as fp:
            self._metric_dtos = jsonpickle.decode(fp.read())

    def save_training(self):
        torch.save(self._optimizer.state_dict(), self._path_optim)
        with open(self._path_train, 'w') as fp:
            fp.write(jsonpickle.encode(self._metric_dtos))

    def save_model(self, suffix=''):
        torch.save(self._model.cpu(), self._path_model.replace(self.EXT_MODEL, suffix + self.EXT_MODEL))
        self._model.cuda()

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

    def plot_epoch(self, plotter, epochs):
        pass

    def visualize_epoch(self, epoch):
        pass

    def adapt_lr(self, epoch):
        if self._scheduler is not None:
            self._scheduler.step()

    def adapt_betas(self, epoch):
        pass

    def run_training(self):
        min_loss = self.get_start_min_loss()

        for epoch in range(self.get_start_epoch(), self._n_epochs):
            self.adapt_lr(epoch)
            self.adapt_betas(epoch)

            # ---------------------------- (1) TRAINING ---------------------------- #

            self._model.train()

            epoch_metrics = MetricMeasuresDtoInit.init_dto()
            for batch in self._dataloader_training:
                epoch_metrics.add(self.train_batch(batch, epoch))
            epoch_metrics.div(len(self._dataloader_training))

            self.print_epoch(epoch, 'training', epoch_metrics)
            self._metric_dtos['training'].append(epoch_metrics)
            del epoch_metrics
            del batch

            # ---------------------------- (2) VALIDATE ---------------------------- #

            self._model.eval()

            epoch_metrics = MetricMeasuresDtoInit.init_dto()
            for batch in self._dataloader_validation:
                epoch_metrics.add(self.validate_batch(batch, epoch))
            epoch_metrics.div(len(self._dataloader_validation))

            self.print_epoch(epoch, 'validate', epoch_metrics)
            self._metric_dtos['validate'].append(epoch_metrics)
            del epoch_metrics
            del batch

            # ------------ (3) SAVE MODEL / VISUALIZE (if new optimum) ------------ #

            if self._metric_dtos['validate'] and self._metric_dtos['validate'][-1].loss < min_loss:
                min_loss = self._metric_dtos['validate'][-1].loss
                self.save_model()
                self.save_training()  # allows to continue if training has been interrupted
                print('(New optimum: Training saved)', end=' ')
                self.visualize_epoch(epoch)

            # ----------------- (4) PLOT / SAVE EVALUATION METRICS ---------------- #

            if epoch > 0:
                fig, plot = plt.subplots()
                self.plot_epoch(plot, range(1, epoch + 2))
                fig.savefig(self._path_outputs_base + self.FN_VIS_BASE + 'plots.png', bbox_inches='tight', dpi=300)
                del plot
                del fig

        # ------------ (5) SAVE FINAL MODEL / VISUALIZE ------------- #

        self.save_model('_final')
        self.visualize_epoch(epoch)
