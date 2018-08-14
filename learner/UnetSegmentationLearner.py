from common.inference.UnetInference import UnetInference
import common.dto.MetricMeasuresDto as MetricMeasuresDtoInit
from learner.Learner import Learner
from common.dto.UnetDto import UnetDto
import matplotlib.pyplot as plt
from common import data, metrics, util
import numpy


class UnetSegmentationLearner(Learner, UnetInference):
    """ A Learner to train a Unet on shape segmentations.
    """
    FN_VIS_BASE = '_unet_'

    def __init__(self, dataloader_training, dataloader_validation, unet_model, path_unet_model, optimizer, scheduler,
                 n_epochs, path_training_metrics, path_outputs_base, criterion):
        Learner.__init__(self, dataloader_training, dataloader_validation, unet_model, path_unet_model, optimizer,
                         scheduler, n_epochs, path_training_metrics=path_training_metrics,
                         path_outputs_base=path_outputs_base)
        self._criterion = criterion  # main loss criterion

    def loss_step(self, dto: UnetDto, epoch):
        loss = 0.0
        divd = 2

        loss += 1 * self._criterion(dto.outputs.core, dto.given_variables.core)
        loss += 1 * self._criterion(dto.outputs.penu, dto.given_variables.penu)

        return loss / divd

    def batch_metrics_step(self, dto: UnetDto, epoch):
        batch_metrics = MetricMeasuresDtoInit.init_dto()
        batch_metrics.core = metrics.binary_measures_torch(dto.outputs.core,
                                                           dto.given_variables.core, self.is_cuda)
        batch_metrics.penu = metrics.binary_measures_torch(dto.outputs.penu,
                                                           dto.given_variables.penu, self.is_cuda)
        return batch_metrics

    def get_start_epoch(self):
        if self._metric_dtos['training']:
            return len([dto.loss for dto in self._metric_dtos['training']])
        return 0

    def get_start_min_loss(self):
        if self._metric_dtos['validate']:
            return min([dto.loss for dto in self._metric_dtos['validate']])
        return numpy.Inf

    def print_epoch(self, epoch, phase, epoch_metrics):
        output = '\nEpoch {}/{} {} loss: {:.3} - DC Core:{:.3}, DC Penumbra:{:.3}'
        print(output.format(epoch + 1, self._n_epochs, phase,
                            epoch_metrics.loss,
                            epoch_metrics.core.dc,
                            epoch_metrics.penu.dc), end=' ')

    def plot_epoch(self, plot, epochs):
        plot.plot(epochs, [dto.loss for dto in self._metric_dtos['training']], 'r-')
        plot.plot(epochs, [dto.loss for dto in self._metric_dtos['validate']], 'g-')
        plot.plot(epochs, [dto.core.dc for dto in self._metric_dtos['validate']], 'c+')
        plot.plot(epochs, [dto.penu.dc for dto in self._metric_dtos['validate']], 'm+')
        plot.set_ylabel('L Train.(red)/Val.(green) | Dice Val. Core(c), Penu(m)')

    def visualize_epoch(self, epoch):
        visual_samples, visual_times = util.get_vis_samples(self._dataloader_training, self._dataloader_validation)

        pad = [20, 20, 20]

        f, axarr = plt.subplots(len(visual_samples), 6)
        inc = 0
        for sample in visual_samples:
            dto = self.inference_step(sample)
            zslice = 34
            axarr[inc, 0].imshow(sample[data.KEY_IMAGES].numpy()[0, 0, zslice, pad[1]:-pad[1], pad[2]:-pad[2]],
                                 vmin=0, vmax=self.IMSHOW_VMAX_CBV, cmap='jet')
            axarr[inc, 1].imshow(dto.given_variables.core.cpu().data.numpy()[0, 0, 14, :, :],
                                 vmin=0, vmax=1, cmap='gray')
            axarr[inc, 2].imshow(dto.outputs.core.cpu().data.numpy()[0, 0, 14, :, :],
                                 vmin=0, vmax=1, cmap='gray')
            axarr[inc, 3].imshow(dto.outputs.penu.cpu().data.numpy()[0, 0, 14, :, :],
                                 vmin=0, vmax=1, cmap='gray')
            axarr[inc, 4].imshow(dto.given_variables.penu.cpu().data.numpy()[0, 0, 14, :, :],
                                 vmin=0, vmax=1, cmap='gray')
            axarr[inc, 5].imshow(sample[data.KEY_IMAGES].numpy()[0, 1, zslice, pad[1]:-pad[1], pad[2]:-pad[2]],
                                 vmin=0, vmax=self.IMSHOW_VMAX_TTD, cmap='jet')

            del sample

            titles = ['CBV', 'Core GT', 'p(Core)', 'p(Penu.)', 'Penu. GT', 'TTD']

            for ax, title in zip(axarr[inc], titles):
                ax.set_title(title)

            inc += 1

        for ax in axarr.flatten():
            ax.title.set_fontsize(3)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

        f.subplots_adjust(hspace=0.05)
        f.savefig(self._path_outputs_base + self.FN_VIS_BASE + str(epoch + 1) + '.png', bbox_inches='tight', dpi=300)

        del f
        del axarr