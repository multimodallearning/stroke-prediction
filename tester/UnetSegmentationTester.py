from tester.Tester import Tester
from common.inference.UnetInference import UnetInference
from common.dto.UnetDto import UnetDto
from common.dto.MetricMeasuresDto import MetricMeasuresDto
import common.dto.MetricMeasuresDto as MetricMeasuresDtoInit
from common.model.Unet3D import Unet3D
from common import data, metrics
import nibabel as nib
import numpy as np


class UnetSegmentationTester(Tester, UnetInference):
    def __init__(self, dataloader, model: Unet3D, path_model, path_outputs_base='/tmp/'):
        Tester.__init__(self, dataloader, model, path_model, path_outputs_base=path_outputs_base)

    def batch_metrics_step(self, dto: UnetDto):
        batch_metrics = MetricMeasuresDtoInit.init_dto()
        batch_metrics.core = metrics.measures_on_binary_numpy(dto.outputs.core.cpu().data.numpy(),
                                                              dto.given_variables.core.cpu().data.numpy())
        batch_metrics.penu = metrics.measures_on_binary_numpy(dto.outputs.penu.cpu().data.numpy(),
                                                              dto.given_variables.penu.cpu().data.numpy())
        return batch_metrics

    def save_inference(self, dto: UnetDto, batch: dict):
        case_id = int(batch[data.KEY_CASE_ID])
        # Output the results on which metrics have been computed
        nifph = nib.load('/share/data_zoe1/lucas/Linda_Segmentations/' + str(case_id) + '/train' + str(case_id) +
                         '_CBVmap_reg1_downsampled.nii.gz').affine
        nib.save(nib.Nifti1Image(np.transpose(dto.outputs.core.cpu().data.numpy(), (4, 3, 2, 1, 0)), nifph),
                 self._path_outputs_base + '_' + str(case_id) + '_core.nii.gz')
        nib.save(nib.Nifti1Image(np.transpose(dto.outputs.penu.cpu().data.numpy(), (4, 3, 2, 1, 0)), nifph),
                 self._path_outputs_base + '_' + str(case_id) + '_penu.nii.gz')

    def print_inference(self, batch: dict, batch_metrics: MetricMeasuresDto, dto: UnetDto):
        output = 'Case Id {}:\t DC Core:{:.3},\tDC Penumbra:{:.3}'
        print(output.format(int(batch[data.KEY_CASE_ID]),
                            batch_metrics.core.dc,
                            batch_metrics.penu.dc))
