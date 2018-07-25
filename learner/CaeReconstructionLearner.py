from learner.Learner import Learner
from common.dto.CaeDto import CaeDto
from common.inference.CaeInference import CaeInference
import matplotlib.pyplot as plt
import torch
from common import data, util


class CaeReconstructionLearner(Learner, CaeInference):
    """ A Learner to train a CAE on the reconstruction of
    shape segmentations. Uses Cae_Dto data transfer objects.
    """
    FN_VIS_BASE = '_cae_'

    def __init__(self, dataloader_training, dataloader_validation, cae_model, path_cae_model, optimizer, n_epochs,
                 path_outputs_base, criterion, normalization_hours_penumbra=10, epoch_interpolant_constraint=1,
                 every_x_epoch_half_lr=100, cuda=True):
        Learner.__init__(self, dataloader_training, dataloader_validation, cae_model, path_cae_model, optimizer,
                         n_epochs, path_outputs_base=path_outputs_base,
                         metrics={'training': {'loss': [], 'dc': [], 'hd': [], 'assd': []},
                                  'validate': {'loss': [], 'dc': [], 'hd': [], 'assd': [], 'dc_core': [], 'dc_penu': []}
                                 }, cuda=cuda)
        CaeInference.__init__(self, cae_model, path_cae_model, path_outputs_base, normalization_hours_penumbra,
                              cuda=cuda)  # TODO: This needs some refactoring (double initialization of model, path etc)
        self._path_model = path_cae_model
        self._criterion = criterion  # main loss criterion
        self._epoch_interpolant_constraint = epoch_interpolant_constraint  # start at epoch to increase weight for the
                                                                           # loss keeping interpolation close to lesion
                                                                           # in latent space
        self._every_x_epoch_half_lr = every_x_epoch_half_lr  # every x-th epoch half the learning rate

    def validation_step(self, batch, epoch):
        pass

    def loss_step(self, dto: CaeDto, epoch):
        loss = 0.0
        divd = 4

        diff_penu_fuct = dto.reconstructions.gtruth.penu - dto.reconstructions.gtruth.interpolation
        diff_penu_core = dto.reconstructions.gtruth.penu - dto.reconstructions.gtruth.core
        loss += 1 * torch.mean(torch.abs(diff_penu_fuct) - diff_penu_fuct)
        loss += 1 * torch.mean(torch.abs(diff_penu_core) - diff_penu_core)

        loss += 1 * self._criterion(dto.reconstructions.gtruth.core, dto.given_variables.gtruth.core)
        loss += 1 * self._criterion(dto.reconstructions.gtruth.penu, dto.given_variables.gtruth.penu)

        if self._epoch_interpolant_constraint < epoch < self._epoch_interpolant_constraint + 25:
            weight = 0.04 * (epoch - self._epoch_interpolant_constraint)
            loss += weight * torch.mean(torch.abs(dto.latents.gtruth.interpolation - dto.latents.gtruth.lesion))
            divd += weight

        return loss / divd

    def metrics_step(self, dto: CaeDto, epoch, running_epoch_metrics):
        dc, hd, assd = util.compute_binary_measure_numpy(dto.reconstructions.gtruth.interpolation.cpu().data.numpy(),
                                                         dto.given_variables.gtruth.lesion.cpu().data.numpy())
        dc_core, _, _ = util.compute_binary_measure_numpy(dto.reconstructions.gtruth.core.cpu().data.numpy(),
                                                          dto.given_variables.gtruth.core.cpu().data.numpy())
        dc_penu, _, _ = util.compute_binary_measure_numpy(dto.reconstructions.gtruth.penu.cpu().data.numpy(),
                                                          dto.given_variables.gtruth.penu.cpu().data.numpy())
        for metric in running_epoch_metrics.keys():
            if metric == 'dc':
                running_epoch_metrics[metric].append(dc)
            elif metric == 'hd':
                running_epoch_metrics[metric].append(hd)
            elif metric == 'assd':
                running_epoch_metrics[metric].append(assd)
            elif metric == 'dc_core':
                running_epoch_metrics[metric].append(dc_core)
            elif metric == 'dc_penu':
                running_epoch_metrics[metric].append(dc_penu)
        return running_epoch_metrics

    def adapt_lr(self, epoch):
        if epoch % self._every_x_epoch_half_lr == self._every_x_epoch_half_lr - 1:
            for param_group in self._optimizer.param_groups:
                param_group['lr'] *= 0.5

    def print_epoch(self, epoch, phase, epoch_metrics):
        output = 'Epoch {}/{} {} loss: {:.3} - DC:{:.3}, HD:{:.3}, ASSD:{:.3}'
        if phase == 'validate':
            output += ', DC core:{:.3}, DC penu.:{:.3}'
            print(output.format(epoch + 1, self._n_epochs, phase,
                                epoch_metrics['loss'][-1],
                                epoch_metrics['dc'][-1],
                                epoch_metrics['hd'][-1],
                                epoch_metrics['assd'][-1],
                                epoch_metrics['dc_core'][-1],
                                epoch_metrics['dc_penu'][-1])
                 )
        elif phase == 'training':
            print(output.format(epoch + 1, self._n_epochs, phase,
                                epoch_metrics['loss'][-1],
                                epoch_metrics['dc'][-1],
                                epoch_metrics['hd'][-1],
                                epoch_metrics['assd'][-1])
                 )
        else:
            print('Given learning phase did not match in order to print correctly!')

    def plot_epoch(self, epoch):
        if epoch > 0:
            fig, ax1 = plt.subplots()
            t = range(1, epoch + 2)
            ax1.plot(t, self._metrics['training']['loss'], 'r-')
            ax1.plot(t, self._metrics['validate']['loss'], 'g-')
            ax1.plot(t, self._metrics['validate']['dc'], 'k-')
            ax1.plot(t, self._metrics['validate']['dc_core'], 'c+')
            ax1.plot(t, self._metrics['validate']['dc_penu'], 'm+')
            ax1.set_ylabel('L Train.(red)/Val.(green) | Dice Val. Lesion(b), Core(c), Penu(m)')
            ax2 = ax1.twinx()
            ax2.plot(t, self._metrics['validate']['assd'], 'b-')
            ax2.set_ylabel('Validation ASSD (blue)', color='b')
            ax2.tick_params('y', colors='b')
            fig.savefig(self._path_outputs_base + 'cae_losses.png', bbox_inches='tight', dpi=300)
            del fig
            del ax1
            del ax2

    def visualize_epoch(self, epoch):
        print('  > new validation loss optimum <  (model saved)')
        visual_samples, visual_times = util.get_vis_samples(self._dataloader_training, self._dataloader_validation)

        pad = [20, 20, 20]

        f, axarr = plt.subplots(len(visual_samples), 15)
        inc = 0
        for sample, time in zip(visual_samples, visual_times):

            col = 3
            for step in [None, -1, 0, 1, 2, 3, 4, 5, 10, 20]:
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
                      + 'h)', 'Core', 'p(-1h)', 'p(0h)', 'p(1h)', 'p(2h)', 'p(3h)', 'p(4h)', 'p(5h)', 'p(10h)',
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