from common.UnetInference import UnetInference
from learner.Learner import Learner
from common.UnetDto import UnetDto
import matplotlib.pyplot as plt
import util
import data


class UnetSegmentationLearner(Learner, UnetInference):
    """ A Learner to train a Unet on shape segmentations.
    """
    FN_VIS_BASE = '_unet_'

    def __init__(self, dataloader_training, dataloader_validation, unet_model, path_unet_model, optimizer, n_epochs,
                 path_outputs_base, criterion, every_x_epoch_half_lr=100, cuda=True):
        super().__init__(dataloader_training, dataloader_validation, unet_model, path_unet_model,
                         optimizer, n_epochs, path_outputs_base=path_outputs_base,
                         metrics={'training': {'loss': [], 'dc_core': [], 'dc_penu': []},
                                  'validate': {'loss': [], 'dc_core': [], 'dc_penu': []}
                                  }, cuda=cuda)
        self._path_model = path_unet_model
        self._criterion = criterion  # main loss criterion
        self._every_x_epoch_half_lr = every_x_epoch_half_lr  # every x-th epoch half the learning rate

    def validation_step(self, batch, epoch):
        pass

    def loss_step(self, dto: UnetDto, epoch):
        loss = 0.0
        divd = 2

        loss += 1 * self._criterion(dto.outputs.core, dto.given_variables.core)
        loss += 1 * self._criterion(dto.outputs.penu, dto.given_variables.penu)

        return loss / divd

    def metrics_step(self, dto: UnetDto, epoch, running_epoch_metrics):
        dc_core, _, _ = util.compute_binary_measure_numpy(dto.outputs.core.cpu().data.numpy(),
                                                          dto.given_variables.core.cpu().data.numpy())
        dc_penu, _, _ = util.compute_binary_measure_numpy(dto.outputs.penu.cpu().data.numpy(),
                                                          dto.given_variables.penu.cpu().data.numpy())
        for metric in running_epoch_metrics.keys():
            if metric == 'dc_core':
                running_epoch_metrics[metric].append(dc_core)
            elif metric == 'dc_penu':
                running_epoch_metrics[metric].append(dc_penu)
        return running_epoch_metrics

    def adapt_lr(self, epoch):
        if epoch % self._every_x_epoch_half_lr == self._every_x_epoch_half_lr - 1:
            for param_group in self._optimizer.param_groups:
                param_group['lr'] *= 0.5

    def print_epoch(self, epoch, phase, epoch_metrics):
        output = 'Epoch {}/{} {} loss: {:.3} - DC Core:{:.3}, DC Penumbra:{:.3}'
        print(output.format(epoch + 1, self._n_epochs, phase,
                            epoch_metrics['loss'][-1],
                            epoch_metrics['dc_core'][-1],
                            epoch_metrics['dc_penu'][-1]))

    def plot_epoch(self, epoch):
        if epoch > 0:
            fig, ax1 = plt.subplots()
            t = range(1, epoch + 2)
            ax1.plot(t, self._metrics['training']['loss'], 'r-')
            ax1.plot(t, self._metrics['validate']['loss'], 'g-')
            ax1.plot(t, self._metrics['validate']['dc_core'], 'c+')
            ax1.plot(t, self._metrics['validate']['dc_penu'], 'm+')
            ax1.set_ylabel('L Train.(red)/Val.(green) | Dice Val. Core(c), Penu(m)')
            fig.savefig(self._path_outputs_base + '_losses.png', bbox_inches='tight', dpi=300)
            del fig
            del ax1

    def visualize_epoch(self, epoch):
        print('  > new validation loss optimum <  (model saved)')
        visual_samples, visual_times = util.get_vis_samples(self._dataloader_training, self._dataloader_validation)

        pad = [20, 20, 20]

        f, axarr = plt.subplots(len(visual_samples), 6)
        inc = 0
        for sample in visual_samples:
            dto = self.inference_step(sample, epoch)
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