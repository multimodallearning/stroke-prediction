from learner.Learner import Learner
from common.dto.CaeDto import CaeDto
from common.inference.CaeInference import CaeInference
import common.dto.MetricMeasuresDto as MetricMeasuresDtoInit
import matplotlib.pyplot as plt
import torch
from common import data, util, metrics
import numpy


class CaeReconstructionLearner(Learner, CaeInference):
    """ A Learner to train a CAE on the reconstruction of
    shape segmentations. Uses CaeDto data transfer objects.
    """
    FN_VIS_BASE = '_cae_'
    N_EPOCHS_ADAPT_BETA1 = 5

    def __init__(self, dataloader_training, dataloader_validation, cae_model, path_cae_model, optimizer, scheduler,
                 n_epochs, path_training_metrics, path_outputs_base, criterion, normalization_hours_penumbra=10):
        Learner.__init__(self, dataloader_training, dataloader_validation, cae_model, path_cae_model, optimizer,
                         scheduler, n_epochs, path_training_metrics=path_training_metrics,
                         path_outputs_base=path_outputs_base)
        CaeInference.__init__(self, cae_model, path_cae_model, path_outputs_base, normalization_hours_penumbra)
                              # TODO: This needs some refactoring (double initialization of model, path etc)
        self._path_model = path_cae_model
        self._criterion = criterion  # main loss criterion

    def adapt_betas(self, epoch):
        betas = self._optimizer.defaults['betas']
        if epoch < self.N_EPOCHS_ADAPT_BETA1:
            betas = list(betas)
            betas[0] -= 0.1 * (self.N_EPOCHS_ADAPT_BETA1 - epoch)
            betas = tuple(betas)
            for param_group in self._optimizer.param_groups:
                param_group['betas'] = betas
            print('Momentum betas have been set to:', param_group['betas'], end=' ')
        elif epoch == self.N_EPOCHS_ADAPT_BETA1:
            for param_group in self._optimizer.param_groups:
                param_group['betas'] = betas
            print('Momentum betas have been set to:', param_group['betas'], end=' ')

    def get_start_epoch(self):
        if self._metric_dtos['training']:
            return len([dto.loss for dto in self._metric_dtos['training']])
        return 0

    def get_start_min_loss(self):
        if self._metric_dtos['validate']:
            return min([dto.loss for dto in self._metric_dtos['validate']])
        return numpy.Inf

    def loss_step(self, dto: CaeDto, epoch):
        loss = 0.0
        divd = 5

        diff_penu_fuct = dto.reconstructions.gtruth.penu - dto.reconstructions.gtruth.interpolation
        diff_penu_core = dto.reconstructions.gtruth.penu - dto.reconstructions.gtruth.core
        loss += 1 * torch.mean(torch.abs(diff_penu_fuct) - diff_penu_fuct)
        loss += 1 * torch.mean(torch.abs(diff_penu_core) - diff_penu_core)

        loss += 1 * self._criterion(dto.reconstructions.gtruth.core, dto.given_variables.gtruth.core)
        loss += 1 * self._criterion(dto.reconstructions.gtruth.penu, dto.given_variables.gtruth.penu)

        loss += 1 * torch.mean(torch.abs(dto.latents.gtruth.interpolation - dto.latents.gtruth.lesion))

        return loss / divd

    def batch_metrics_step(self, dto: CaeDto, epoch):
        batch_metrics = MetricMeasuresDtoInit.init_dto()
        batch_metrics.lesion = metrics.binary_measures_torch(dto.reconstructions.gtruth.interpolation,
                                                             dto.given_variables.gtruth.lesion, self.is_cuda)
        batch_metrics.core = metrics.binary_measures_torch(dto.reconstructions.gtruth.core,
                                                           dto.given_variables.gtruth.core, self.is_cuda)
        batch_metrics.penu = metrics.binary_measures_torch(dto.reconstructions.gtruth.penu,
                                                           dto.given_variables.gtruth.penu, self.is_cuda)
        return batch_metrics

    def print_epoch(self, epoch, phase, epoch_metrics):
        output = '\nEpoch {}/{} {} loss: {:.3} - DC:{:.3}, HD:{:.3}, ASSD:{:.3}, DC core:{:.3}, DC penu.:{:.3}'
        print(output.format(epoch + 1, self._n_epochs, phase,
                            epoch_metrics.loss,
                            epoch_metrics.lesion.dc,
                            epoch_metrics.lesion.hd,
                            epoch_metrics.lesion.assd,
                            epoch_metrics.core.dc,
                            epoch_metrics.penu.dc), end=' ')

    def plot_epoch(self, plot, epochs):
        plot.plot(epochs, [dto.loss for dto in self._metric_dtos['training']], 'r-')
        plot.plot(epochs, [dto.loss for dto in self._metric_dtos['validate']], 'g-')
        plot.plot(epochs, [dto.lesion.dc for dto in self._metric_dtos['validate']], 'k-')
        plot.plot(epochs, [dto.core.dc for dto in self._metric_dtos['validate']], 'c+')
        plot.plot(epochs, [dto.penu.dc for dto in self._metric_dtos['validate']], 'm+')
        plot.set_ylabel('L Train.(red)/Val.(green) | Dice Val. Lesion(b), Core(c), Penu(m)')
        plot.set_ylim(0, 1)
        ax2 = plot.twinx()
        ax2.plot(epochs, [dto.lesion.assd for dto in self._metric_dtos['validate']], 'b-')
        ax2.set_ylabel('Validation ASSD (blue)', color='b')
        ax2.tick_params('y', colors='b')

    def visualize_epoch(self, epoch):
        visual_samples, visual_times = util.get_vis_samples(self._dataloader_training, self._dataloader_validation)

        pad = [20, 20, 20]

        f, axarr = plt.subplots(len(visual_samples), 15)
        inc = 0
        for sample, time in zip(visual_samples, visual_times):

            col = 3
            for step in [None, -10, -1, 0, 1, 2, 3, 4, 5, 20]:
                dto = self.inference_step(sample, step)
                axarr[inc, col].imshow(dto.reconstructions.gtruth.interpolation.cpu().data.numpy()[0, 0, 14, :, :],
                                       vmin=0, vmax=1, cmap='gray')
                if col == 3:
                    col += 1
                col += 1

            zslice = 34
            axarr[inc, 0].imshow(sample[data.KEY_IMAGES].numpy()[0, 0, zslice, pad[1]:-pad[1], pad[2]:-pad[2]],
                                 vmin=0, vmax=self.IMSHOW_VMAX_CBV, cmap='jet')
            axarr[inc, 1].imshow(sample[data.KEY_IMAGES].numpy()[0, 1, zslice, pad[1]:-pad[1], pad[2]:-pad[2]],
                                 vmin=0, vmax=self.IMSHOW_VMAX_TTD, cmap='jet')
            axarr[inc, 2].imshow(dto.given_variables.gtruth.lesion.cpu().data.numpy()[0, 0, 14, :, :],
                                 vmin=0, vmax=1, cmap='gray')
            axarr[inc, 4].imshow(dto.given_variables.gtruth.core.cpu().data.numpy()[0, 0, 14, :, :],
                                 vmin=0, vmax=1, cmap='gray')
            axarr[inc, 14].imshow(dto.given_variables.gtruth.penu.cpu().data.numpy()[0, 0, 14, :, :],
                                  vmin=0, vmax=1, cmap='gray')

            del sample
            del dto

            titles = ['CBV', 'TTD', 'Lesion', 'p(' +
                      ('{:03.1f}'.format(float(time)))
                      + 'h)', 'Core', 'p(-10h)', 'p(-1h)', 'p(0h)', 'p(1h)', 'p(2h)', 'p(3h)', 'p(4h)', 'p(5h)',
                      'p(20h)',
                      'Penumbra']

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
