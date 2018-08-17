from tester.Tester import Tester
from common.inference.UnetInference import UnetInference
from common.dto.UnetDto import UnetDto
from common.dto.MetricMeasuresDto import MetricMeasuresDto
import common.dto.MetricMeasuresDto as MetricMeasuresDtoInit
from common import data, metrics
import nibabel as nib
import numpy as np
import scipy.ndimage.interpolation as ndi


class UnetSegmentationTester(Tester, UnetInference):
    def __init__(self, dataloader, path_model, path_outputs_base='/tmp/', padding=None):
        Tester.__init__(self, dataloader, path_model, path_outputs_base=path_outputs_base)
        self._pad = padding

    def batch_metrics_step(self, dto: UnetDto):
        batch_metrics = MetricMeasuresDtoInit.init_dto()
        batch_metrics.core = metrics.binary_measures_torch(dto.outputs.core,
                                                           dto.given_variables.core, self.is_cuda)
        batch_metrics.penu = metrics.binary_measures_torch(dto.outputs.penu,
                                                           dto.given_variables.penu, self.is_cuda)
        return batch_metrics

    def _transpose_unpad_zoom(self, image):
        image = np.transpose(image, (4, 3, 2, 1, 0))
        if self._pad is not None:
            image = image[self._pad[0]:-self._pad[0], self._pad[1]:-self._pad[1], self._pad[2]:-self._pad[2], 0, 0]
        return ndi.zoom(image, zoom=(2, 2, 1))

    def save_inference(self, dto: UnetDto, batch: dict, suffix=''):
        case_id = int(batch[data.KEY_CASE_ID])
        # Output the results on which metrics have been computed
        nifph = nib.load('/share/data_zoe1/lucas/Linda_Segmentations/' + str(case_id) + '/train' + str(case_id) +
                         '_TTDmap_reg1_downsampled.nii.gz').affine
        core = self._transpose_unpad_zoom(dto.outputs.core.cpu().data.numpy())
        nib.save(nib.Nifti1Image(core, nifph), self._fn(case_id, '_core', suffix))
        penu = self._transpose_unpad_zoom(dto.outputs.penu.cpu().data.numpy())
        nib.save(nib.Nifti1Image(penu, nifph), self._fn(case_id, '_penu', suffix))

    def print_inference(self, batch: dict, batch_metrics: MetricMeasuresDto, dto: UnetDto):
        output = 'Case Id {}:\t DC Core:{:.3},\tDC Penumbra:{:.3}'
        print(output.format(int(batch[data.KEY_CASE_ID]),
                            batch_metrics.core.dc,
                            batch_metrics.penu.dc))
