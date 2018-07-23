from common.Inference import Inference
from torch.autograd import Variable
import common.UnetDto as UnetDtoInit
import data


class UnetInference(Inference):
    """Common inference for training and testing,
    i.e. feed-forward of Unet
    """
    def inference_step(self, batch, epoch):
        input_modalities = Variable(batch[data.KEY_IMAGES])
        core_gt = Variable(batch[data.KEY_LABELS][:, 0, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5))
        penu_gt = Variable(batch[data.KEY_LABELS][:, 1, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5))

        if self._cuda:
            input_modalities = input_modalities.cuda()
            core_gt = core_gt.cuda()
            penu_gt = penu_gt.cuda()

        dto = UnetDtoInit.init_unet_dto(input_modalities, core_gt, penu_gt)

        return self._model(dto)