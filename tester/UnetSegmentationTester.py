from tester.Tester import Tester
from common.inference.UnetInference import UnetInference
from common.dto.UnetDto import UnetDto
from common.model.Unet3D import Unet3D
import nibabel as nib
import numpy as np
from common import data, util


class UnetSegmentationTester(Tester, UnetInference):
    def __init__(self, dataloader, model:Unet3D, path_model, path_outputs_base):
        Tester.__init__(self, dataloader, model, path_model, path_outputs_base=path_outputs_base,
                        metrics={'dc_core': [], 'dc_penu': []})

    def metrics_step(self, dto: UnetDto, metrics):
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

    def save_inference(self, dto: UnetDto, batch):
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