from common.model.Cae3D import Cae3D
from common.inference.CaeInference import CaeInference
from common.dto.CaeDto import CaeDto
from common.dto.MetricMeasuresDto import MetricMeasuresDto
import common.dto.MetricMeasuresDto as MetricMeasuresDtoInit
from tester.Tester import Tester
import nibabel as nib
import numpy as np
from common import metrics, data


class CaeReconstructionTester(Tester, CaeInference):
    def __init__(self, dataloader, model: Cae3D, path_model, path_outputs_base='/tmp/', normalization_hours_penumbra=0):
        Tester.__init__(self, dataloader, model, path_model, path_outputs_base=path_outputs_base)
        CaeInference.__init__(self, model, path_model, path_outputs_base, normalization_hours_penumbra,
                              cuda=True)  # TODO: This needs some refactoring (double initialization of model, path etc)
                                          # TODO: Check if cuda required for the Inference modules

    def batch_metrics_step(self, dto: CaeDto):
        batch_metrics = MetricMeasuresDtoInit.init_dto()
        batch_metrics.lesion = metrics.measures_on_binary_numpy(dto.reconstructions.gtruth.interpolation.cpu().data.numpy(),
                                                                dto.given_variables.gtruth.lesion.cpu().data.numpy())
        batch_metrics.core = metrics.measures_on_binary_numpy(dto.reconstructions.gtruth.core.cpu().data.numpy(),
                                                              dto.given_variables.gtruth.core.cpu().data.numpy())
        batch_metrics.penu = metrics.measures_on_binary_numpy(dto.reconstructions.gtruth.penu.cpu().data.numpy(),
                                                              dto.given_variables.gtruth.penu.cpu().data.numpy())
        return batch_metrics

    def save_inference(self, dto: CaeDto, batch: dict):
        case_id = int(batch[data.KEY_CASE_ID])
        # Output results on which metrics have been computed
        nifph = nib.load('/share/data_zoe1/lucas/Linda_Segmentations/' + str(case_id) + '/train' + str(case_id) +
                         '_CBVmap_reg1_downsampled.nii.gz').affine
        nib.save(
            nib.Nifti1Image(np.transpose(dto.reconstructions.gtruth.core.cpu().data.numpy(), (4, 3, 2, 1, 0)), nifph),
            self._path_outputs_base + '_' + str(case_id) + '_core.nii.gz'
        )
        nib.save(
            nib.Nifti1Image(np.transpose(dto.reconstructions.gtruth.interpolation.cpu().data.numpy(), (4, 3, 2, 1, 0)), nifph),
            self._path_outputs_base + '_' + str(case_id) + '_pred.nii.gz'
        )
        nib.save(
            nib.Nifti1Image(np.transpose(dto.reconstructions.gtruth.penu.cpu().data.numpy(), (4, 3, 2, 1, 0)), nifph),
            self._path_outputs_base + '_' + str(case_id) + '_penu.nii.gz'
        )

    def print_inference(self, batch: dict, batch_metrics: MetricMeasuresDto):
        output = 'Case Id {}:\tDC:{:.3},\tHD:{:.3},\tASSD:{:.3},\tDC Core:{:.3},\tDC Penumbra:{:.3}'
        print(output.format(int(batch[data.KEY_CASE_ID]),
                            batch_metrics.lesion.dc,
                            batch_metrics.lesion.hd,
                            batch_metrics.lesion.assd,
                            batch_metrics.core.dc,
                            batch_metrics.penu.dc))