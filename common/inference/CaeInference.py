from common.model.Cae3D import Cae3D
from common.inference.Inference import Inference
from torch.autograd import Variable
import common.dto.CaeDto as CaeDtoInit
import torch
from common import data


class CaeInference(Inference):
    """Common inference for training and testing,
    i.e. feed-forward of CAE
    """
    def __init__(self, model:Cae3D, path_model, path_outputs_base='/tmp/', normalization_hours_penumbra = 10,
                 cuda=True):
        Inference.__init__(self, model, path_model, path_outputs_base, cuda)
        self._normalization_hours_penumbra = normalization_hours_penumbra

    def _get_normalized_time(self, batch):
        to_to_ta = batch[data.KEY_GLOBAL][:, 0, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5).type(torch.FloatTensor)
        normalization = torch.ones(to_to_ta.size()[0], 1).type(torch.FloatTensor) * \
                        self._normalization_hours_penumbra - to_to_ta.squeeze().unsqueeze(data.DIM_CHANNEL_TORCH3D_5)
        return to_to_ta, normalization

    def inference_step(self, batch, step=None):
        to_to_ta, normalization = self._get_normalized_time(batch)

        globals_no_times = batch[data.KEY_GLOBAL][:, 2:, :, :, :].type(torch.FloatTensor)
        globals_incl_time = Variable(torch.cat((to_to_ta, globals_no_times), dim=data.DIM_CHANNEL_TORCH3D_5))
        type_core = Variable(torch.zeros(globals_incl_time.size()[0], 1, 1, 1, 1))
        type_penumbra = Variable(torch.ones(globals_incl_time.size()[0], 1, 1, 1, 1))
        core_gt = Variable(batch[data.KEY_LABELS][:, 0, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5))

        if step is None:
            ta_to_tr = batch[data.KEY_GLOBAL][:, 1, :, :, :].squeeze().unsqueeze(data.DIM_CHANNEL_TORCH3D_5)
            time_to_treatment = Variable(ta_to_tr.type(torch.FloatTensor) / normalization)
        else:
            time_to_treatment = Variable((step * torch.ones(core_gt.size()[0], 1)) / normalization)

        del to_to_ta
        del normalization
        del globals_no_times

        cbv = Variable(batch[data.KEY_IMAGES][:, 0, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5))
        ttd = Variable(batch[data.KEY_IMAGES][:, 1, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5))
        penu_gt = Variable(batch[data.KEY_LABELS][:, 1, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5))
        lesion_gt = Variable(batch[data.KEY_LABELS][:, 2, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5))

        if self._cuda:
            globals_incl_time = globals_incl_time.cuda()
            time_to_treatment = time_to_treatment.cuda()
            type_core = type_core.cuda()
            type_penumbra = type_penumbra.cuda()
            cbv = cbv.cuda()
            ttd = ttd.cuda()
            core_gt = core_gt.cuda()
            penu_gt = penu_gt.cuda()
            lesion_gt = lesion_gt.cuda()

        dto = CaeDtoInit.init_dto(globals_incl_time,
                                  time_to_treatment.unsqueeze(2).unsqueeze(3).unsqueeze(4),
                                  type_core,
                                  type_penumbra,
                                  cbv,
                                  ttd,
                                  core_gt,
                                  penu_gt,
                                  lesion_gt
                                  )

        return self._model(dto)