from common.inference.CaeInference import CaeInference
from common.dto.CaeDto import CaeDto
from common.dto.MetricMeasuresDto import MetricMeasuresDto
import common.dto.MetricMeasuresDto as MetricMeasuresDtoInit
from common import metrics, data
from tester.Tester import Tester
import scipy.ndimage.interpolation as ndi
import nibabel as nib
import numpy as np


class CaeReconstructionTester(Tester, CaeInference):
    def __init__(self, dataloader, path_model, path_outputs_base='/tmp/', normalization_hours_penumbra=10):
        Tester.__init__(self, dataloader, path_model, path_outputs_base=path_outputs_base)
        CaeInference.__init__(self, self._model, path_model, path_outputs_base, normalization_hours_penumbra)
        # TODO: This needs some refactoring (double initialization of model, path etc)

    def batch_metrics_step(self, dto: CaeDto):
        batch_metrics = MetricMeasuresDtoInit.init_dto()
        batch_metrics.lesion = metrics.binary_measures_torch(dto.reconstructions.gtruth.interpolation,
                                                             dto.given_variables.gtruth.lesion, self.is_cuda)
        batch_metrics.core = metrics.binary_measures_torch(dto.reconstructions.gtruth.core,
                                                           dto.given_variables.gtruth.core, self.is_cuda)
        batch_metrics.penu = metrics.binary_measures_torch(dto.reconstructions.gtruth.penu,
                                                           dto.given_variables.gtruth.penu, self.is_cuda)
        return batch_metrics

    def save_inference(self, dto: CaeDto, batch: dict, suffix=''):
        case_id = int(batch[data.KEY_CASE_ID])
        # Output results on which metrics have been computed
        nifph = nib.load('/share/data_zoe1/lucas/Linda_Segmentations/' + str(case_id) + '/train' + str(case_id) +
                         '_CBVmap_reg1_downsampled.nii.gz').affine
        image = np.transpose(dto.reconstructions.gtruth.core.cpu().data.numpy(), (4, 3, 2, 1, 0))[:, :, :, 0, 0]
        nib.save(nib.Nifti1Image(ndi.zoom(image, zoom=(2, 2, 1)), nifph), self._fn(case_id, '_core', suffix))

        nifph = nib.load('/share/data_zoe1/lucas/Linda_Segmentations/' + str(case_id) + '/train' + str(case_id) +
                         '_FUCT_MAP_T_Samplespace_reg1_downsampled.nii.gz').affine
        image = np.transpose(dto.reconstructions.gtruth.interpolation.cpu().data.numpy(), (4, 3, 2, 1, 0))[:, :, :, 0, 0]
        nib.save(nib.Nifti1Image(ndi.zoom(image, zoom=(2, 2, 1)), nifph), self._fn(case_id, '_pred', suffix))

        nifph = nib.load('/share/data_zoe1/lucas/Linda_Segmentations/' + str(case_id) + '/train' + str(case_id) +
                         '_TTDmap_reg1_downsampled.nii.gz').affine
        image = np.transpose(dto.reconstructions.gtruth.penu.cpu().data.numpy(), (4, 3, 2, 1, 0))[:, :, :, 0, 0]
        nib.save(nib.Nifti1Image(ndi.zoom(image, zoom=(2, 2, 1)), nifph), self._fn(case_id, '_penu', suffix))

    def print_inference(self, batch: dict, batch_metrics: MetricMeasuresDto, dto: CaeDto, note=''):
        output = 'Case Id={}\ttA-tO={:.3f}\ttR-tA={:.3f}\tnormalized_time_to_treatment={:.3f}\t-->\
                  \tDC={:.3f}\tHD={:.3f}\tASSD={:.3f}\tDC Core={:.3f}\tDC Penumbra={:.3f}\t\
                  Precision={:.3}\tRecall/Sensitivity={:.3}\tSpecificity={:.3}\tDistToCornerPRC={:.3}\t{}'
        print(output.format(int(batch[data.KEY_CASE_ID]),
                            float(batch[data.KEY_GLOBAL][:, 0, :, :, :]),
                            float(batch[data.KEY_GLOBAL][:, 1, :, :, :]),
                            float(dto.given_variables.time_to_treatment),
                            batch_metrics.lesion.dc,
                            batch_metrics.lesion.hd,
                            batch_metrics.lesion.assd,
                            batch_metrics.core.dc,
                            batch_metrics.penu.dc,
                            batch_metrics.lesion.precision,
                            batch_metrics.lesion.sensitivity,
                            batch_metrics.lesion.specificity,
                            batch_metrics.lesion.prc_euclidean_distance,
                            note))