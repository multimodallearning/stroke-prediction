from common.model.Cae3D import Cae3D
from common.dto.CaeDto import CaeDto
from common.inference.CaeInference import CaeInference
from tester.Tester import Tester
import nibabel as nib
import numpy as np
from common import metrics, data


class CaeReconstructionTester(Tester, CaeInference):
    def __init__(self, dataloader, model:Cae3D, path_model_TESTER, path_outputs_base):
        Tester.__init__(self, dataloader, model, path_model_TESTER, path_outputs_base=path_outputs_base,
                        metrics={'dc_core': [], 'dc_penu': [], 'dc': [], 'hd': [], 'assd': []})

    def metrics_step(self, dto: CaeDto, metric_measures):
        metric_lesion = metrics.compute_binary_measure_numpy(dto.reconstructions.gtruth.interpolation.cpu().data.numpy(),
                                                             dto.given_variables.gtruth.lesion.cpu().data.numpy())
        metric_core, _, _ = metrics.compute_binary_measure_numpy(dto.reconstructions.gtruth.core.cpu().data.numpy(),
                                                                 dto.given_variables.gtruth.core.cpu().data.numpy())
        metric_penu, _, _ = metrics.compute_binary_measure_numpy(dto.reconstructions.gtruth.penu.cpu().data.numpy(),
                                                                 dto.given_variables.gtruth.penu.cpu().data.numpy())
        for mm in metric_measures.keys():
            if mm == 'dc_core':
                metric_measures[mm].append(metric_core['dc'])
            elif mm == 'dc_penu':
                metric_measures[mm].append(metric_penu['dc'])
            else:
                metric_measures[mm].append(metric_lesion[mm])

        return metric_measures

    def save_inference(self, dto: CaeDto, batch):
        case_id = int(batch[data.KEY_CASE_ID])
        # Output results on which metrics have been computed
        nifph = nib.load('/share/data_zoe1/lucas/Linda_Segmentations/' + str(case_id) + '/train' + str(case_id) +
                         '_CBVmap_reg1_downsampled.nii.gz').affine
        nib.save(nib.Nifti1Image(np.transpose(dto.reconstructions.gtruth.core.cpu().data.numpy(), (4, 3, 2, 1, 0)), nifph),
                 self._path_outputs_base + '_' + str(case_id) + '_core.nii.gz')
        nib.save(nib.Nifti1Image(np.transpose(dto.reconstructions.gtruth.interpolation.cpu().data.numpy(), (4, 3, 2, 1, 0)), nifph),
            self._path_outputs_base + '_' + str(case_id) + '_pred.nii.gz')
        nib.save(nib.Nifti1Image(np.transpose(dto.reconstructions.gtruth.penu.cpu().data.numpy(), (4, 3, 2, 1, 0)), nifph),
                 self._path_outputs_base + '_' + str(case_id) + '_penu.nii.gz')

    def print_inference(self, batch, metrics):
        output = 'Case Id {}:\tDC:{:.3},\tHD:{:.3},\tASSD:{:.3},\tDC Core:{:.3},\tDC Penumbra:{:.3}'
        print(output.format(int(batch[data.KEY_CASE_ID]),
                            metrics['dc'][-1],
                            metrics['hd'][-1],
                            metrics['assd'][-1],
                            metrics['dc_core'][-1],
                            metrics['dc_penu'][-1]))