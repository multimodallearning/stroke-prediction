import common.UnetDto as UnetDtoInit
from torch.autograd import Variable
from tester.Tester import Tester
from common.Dto import Dto
import nibabel as nib
import numpy as np
import data
import util


class UnetSegmentationTester(Tester):
    def __init__(self, dataloader, model, path_model, path_outputs_base):
        super().__init__(dataloader, model, path_model, path_outputs_base=path_outputs_base,
                         metrics={'dc_core': [], 'dc_penu': []})
        assert dataloader.batch_size == 1, "Dataloader must process one case per batch for correct metrics measures"

    def inference_step(self, batch):
        input_modalities = Variable(batch[data.KEY_IMAGES])
        core_gt = Variable(batch[data.KEY_LABELS][:, 0, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5))
        penu_gt = Variable(batch[data.KEY_LABELS][:, 1, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5))

        if self._cuda:
            input_modalities = input_modalities.cuda()
            core_gt = core_gt.cuda()
            penu_gt = penu_gt.cuda()

        dto = UnetDtoInit.init_unet_dto(input_modalities, core_gt, penu_gt)

        return self._model(dto)

    def metrics_step(self, dto: Dto, metrics):
        dc_core, _, _ = util.compute_binary_measure_numpy(dto.outputs.core.cpu().data.numpy(),
                                                          dto.given_variables.core.cpu().data.numpy())
        dc_penu, _, _ = util.compute_binary_measure_numpy(dto.outputs.penu.cpu().data.numpy(),
                                                          dto.given_variables.penu.cpu().data.numpy())
        for metric in metrics.keys():
            if metric == 'dc_core':
                metrics[metric].append(dc_core)
            elif metric == 'dc_penu':
                metrics[metric].append(dc_penu)
        return metrics

    def save_inference(self, dto: Dto, batch):
        case_id = int(batch[data.KEY_CASE_ID])
        # Output results on which metrics have been computed
        nifph = nib.load('/share/data_zoe1/lucas/Linda_Segmentations/' + str(case_id) + '/train' + str(case_id) +
                         '_CBVmap_reg1_downsampled.nii.gz').affine
        nib.save(nib.Nifti1Image(np.transpose(dto.outputs.core.cpu().data.numpy(), (4, 3, 2, 1, 0)), nifph),
                 self._path_outputs_base + '_' + str(case_id) + '_core.nii.gz')
        nib.save(nib.Nifti1Image(np.transpose(dto.outputs.penu.cpu().data.numpy(), (4, 3, 2, 1, 0)), nifph),
                 self._path_outputs_base + '_' + str(case_id) + '_penu.nii.gz')

    def print_inference(self, batch, metrics):
        output = 'Case Id {}:\t DC Core:{:.3},\tDC Penumbra:{:.3}'
        print(output.format(int(batch[data.KEY_CASE_ID]),
                            metrics['dc_core'][-1],
                            metrics['dc_penu'][-1]))