import datetime
import common.dto.CaeDto as CaeDtoUtil
from common import data, util, metrics
import torch
from torch.autograd import Variable
from matplotlib import pyplot as plt
from sklearn.metrics.classification import f1_score as f1


class OracleModule(torch.nn.Module):
    def __init__(self, batch_size, model):
        super().__init__()
        self.decoder = model
        for param in self.decoder.parameters():
            param.requires_grad = False
        self.zcode = torch.nn.Parameter(torch.rand(batch_size, 800, 1, 10, 10))

    def forward(self, dto):
        dto.latents.gtruth.lesion = self.zcode
        return self.decoder(dto)


def test():
    args = util.get_args_shape_testing()

    assert len(args.fold) == len(args.path), 'You must provide as many --fold arguments as caepath model arguments\
                                                in the exact same order!'

    # Params / Config
    modalities = ['_CBV_reg1_downsampled', '_TTD_reg1_downsampled']
    labels = ['_CBVmap_subset_reg1_downsampled_mirrored', '_TTDmap_subset_reg1_downsampled_mirrored',
              '_FUCT_MAP_T_Samplespace_subset_reg1_downsampled_mirrored']
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

        for batch in ds_test:
            oracle = OracleModule(ds_test.batch_size, model.dec).cuda()
            opt = torch.optim.Adam([oracle.zcode], 1e-1)

            for epoch in range(n_opt_epochs):
                lesion_gt = Variable(batch[data.KEY_LABELS][:, 2, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5)).cuda()
                dto = CaeDtoUtil.init_dto(None, None, None, None, None, None, None, None, lesion_gt)
                dto.flag = CaeDtoUtil.FLAG_GTRUTH

                dto = oracle(dto)

                loss = criterion(dto.reconstructions.gtruth.lesion, dto.given_variables.gtruth.lesion)

                loss.backward()
                opt.step()
                opt.zero_grad()

                if epoch % 100 == 0:
                    print('{:1.4f}'.format(float(loss)), end=' -> ')

                '''
                if epoch == 0:
                    dto_init = dto
                elif epoch == 4:
                    dto0 = dto
                elif epoch == 24:
                    dto1 = dto
                elif epoch == 124:
                    dto2 = dto
                '''

            print('ID', int(batch[data.KEY_CASE_ID]), 'F1:', f1(lesion_gt.data.cpu().numpy().flatten() > 0.5,
                                                                  dto.reconstructions.gtruth.lesion.data.cpu().numpy().flatten() > 0.5))

            '''
            plt.figure()
            plt.subplot(161)
            plt.imshow(lesion_gt[0, 0, z_slice].data.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            plt.subplot(162)
            plt.imshow(dto_init.reconstructions.gtruth.lesion[0, 0, z_slice].data.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            plt.subplot(163)
            plt.imshow(dto0.reconstructions.gtruth.lesion[0, 0, z_slice].data.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            plt.subplot(164)
            plt.imshow(dto1.reconstructions.gtruth.lesion[0, 0, z_slice].data.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            plt.subplot(165)
            plt.imshow(dto2.reconstructions.gtruth.lesion[0, 0, z_slice].data.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            plt.subplot(166)
            plt.imshow(dto.reconstructions.gtruth.lesion[0, 0, z_slice].data.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            plt.show()
            '''
            

if __name__ == '__main__':
    print(datetime.datetime.now())
    test()
    print(datetime.datetime.now())
