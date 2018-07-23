from common.CaeDto import CaeDto
from common.CaeInference import CaeInference
from tester.Tester import Tester
import nibabel as nib
import numpy as np
import data
import util


class CaeReconstructionTester(Tester, CaeInference):
    def __init__(self, dataloader, model, path_model, path_outputs_base):
        Tester.__init__(dataloader, model, path_model, path_outputs_base=path_outputs_base,
                        metrics={'dc_core': [], 'dc_penu': [], 'dc': [], 'hd': [], 'assd': []})

    def metrics_step(self, dto: CaeDto, metrics):
        dc, hd, assd = util.compute_binary_measure_numpy(dto.reconstructions.gtruth.interpolation.cpu().data.numpy(),
                                                         dto.given_variables.gtruth.lesion.cpu().data.numpy())
        dc_core, _, _ = util.compute_binary_measure_numpy(dto.reconstructions.gtruth.core.cpu().data.numpy(),
                                                          dto.given_variables.gtruth.core.cpu().data.numpy())
        dc_penu, _, _ = util.compute_binary_measure_numpy(dto.reconstructions.gtruth.penu.cpu().data.numpy(),
                                                          dto.given_variables.gtruth.penu.cpu().data.numpy())
        for metric in metrics.keys():
            if metric == 'dc':
                metrics[metric].append(dc)
            elif metric == 'hd':
                metrics[metric].append(hd)
            elif metric == 'assd':
                metrics[metric].append(assd)
            elif metric == 'dc_core':
                metrics[metric].append(dc_core)
            elif metric == 'dc_penu':
                metrics[metric].append(dc_penu)
        return metrics

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