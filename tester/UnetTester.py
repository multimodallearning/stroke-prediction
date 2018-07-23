import common.UnetDto as UnetDtoInit
from torch.autograd import Variable
from tester.Tester import Tester
from common.Dto import Dto
import data
import util


class UnetTester(Tester):
    def __init__(self):
        super().__init__(metrics={'loss': [], 'dc_core': [], 'dc_penu': []})

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

    def print_inference(self, phase, metrics):
        output = '{} loss: {:.3} - DC Core:{:.3}, DC Penumbra:{:.3}'
        print(output.format(phase,
                            metrics['loss'][-1],
                            metrics['dc_core'][-1],
                            metrics['dc_penu'][-1]))