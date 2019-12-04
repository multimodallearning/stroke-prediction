import datetime
import common.dto.CaeDto as CaeDtoUtil
from common import data, util, metrics
import torch
from torch.autograd import Variable
from matplotlib import pyplot as plt
from sklearn.metrics.classification import f1_score as f1
import nibabel as nib
import numpy as np
import scipy.ndimage.interpolation as ndi


class OracleModule(torch.nn.Module):
    def __init__(self, batch_size, model):
        super().__init__()
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False
        self.step = torch.nn.Parameter(torch.rand(batch_size, 1, 1, 1, 1))

    def forward(self, dto):
        dto.given_variables.time_to_treatment = torch.nn.functional.relu(self.step)
        dto = self.model(dto)
        return dto


def test():
    args = util.get_args_shape_testing()

    assert len(args.fold) == len(args.path), 'You must provide as many --fold arguments as caepath model arguments\
                                                in the exact same order!'

    # Params / Config
    modalities = ['_CBV_reg1_downsampled', '_TTD_reg1_downsampled']
    labels = ['_CBVmap_subset_reg1_downsampled_mirrored', '_TTDmap_subset_reg1_downsampled_mirrored',
              '_FUCT-CBV-MAP_MAX_subset_reg1_downsampled_mirrored']
    criterion = metrics.BatchDiceLoss([1.0])
    n_opt_epochs = 600
    pad = args.padding
    pad_value = 0
    z_slice = 13

    # Data
    transform = [data.ResamplePlaneXY(args.xyresample),
                 data.PadImages(pad[0], pad[1], pad[2], pad_value=pad_value),
                 data.ToTensor()]

    # Fold-wise evaluation according to fold indices and fold model for all folds and model path provided as arguments:
    for i, path in enumerate(args.path):
        print('Model ' + path + ' of fold ' + str(i+1) + '/' + str(len(args.fold)) + ' with indices: ' + str(args.fold[i]))
        ds_test = data.get_testdata(modalities=modalities, labels=labels, transform=transform, indices=args.fold[i])
        print('Size test set:', len(ds_test.sampler.indices), '| # batches:', len(ds_test))
        # Single case evaluation for all cases in fold

        model = torch.load(path).cuda()
        type_core = Variable(torch.zeros(ds_test.batch_size, 1, 1, 1, 1)).cuda()
        type_penumbra = Variable(torch.ones(ds_test.batch_size, 1, 1, 1, 1)).cuda()

        for batch in ds_test:
            oracle = OracleModule(ds_test.batch_size, model).cuda()
            opt = torch.optim.Adam([oracle.step], 1e-1)

            for epoch in range(n_opt_epochs):
                core_gt = Variable(batch[data.KEY_LABELS][:, 0, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5)).cuda()
                penu_gt = Variable(batch[data.KEY_LABELS][:, 1, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5)).cuda()
                lesion_gt = Variable(batch[data.KEY_LABELS][:, 2, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5)).cuda()
                dto = CaeDtoUtil.init_dto(None, None, type_core, type_penumbra, None, None, core_gt, penu_gt, lesion_gt)
                dto.flag = CaeDtoUtil.FLAG_GTRUTH

                dto = oracle(dto)

                loss = criterion(dto.reconstructions.gtruth.interpolation, dto.given_variables.gtruth.lesion)

                loss.backward()
                opt.step()
                opt.zero_grad()

                if epoch % 100 == 0:
                    print('{:1.4f}'.format(float(loss)), end=' -> ')

            case_id = int(batch[data.KEY_CASE_ID])
            print('ID', case_id, 'F1:', f1(lesion_gt.data.cpu().numpy().flatten() > 0.5, dto.reconstructions.gtruth.interpolation.data.cpu().numpy().flatten() > 0.5),
                  '(time:', float(dto.given_variables.time_to_treatment), ')')

            nifph = nib.load('/share/data_zoe1/lucas/Linda_Segmentations/' + str(case_id) + '/train' + str(case_id) + '_FUCT-CBV-MAP_MAX_subset_reg1_downsampled_mirrored.nii.gz').affine
            image = np.transpose(dto.reconstructions.gtruth.interpolation.cpu().data.numpy(), (4, 3, 2, 1, 0))[:, :, :, 0, 0]
            nib.save(nib.Nifti1Image(ndi.zoom(image, zoom=(2, 2, 1)), nifph), '/data_zoe1/lucas/Linda_Segmentations/tmp/train' + str(case_id) + '_FUCT_CBVFUmax_timeOracle_CAE1.nii.gz')


if __name__ == '__main__':
    print(datetime.datetime.now())
    test()
    print(datetime.datetime.now())
