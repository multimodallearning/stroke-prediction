import torch
import torch.nn as nn
from common import data, metrics
from GRUnet_2 import BidirectionalSequence, tensor2index
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import argparse
import datetime


class Criterion(nn.Module):
    def __init__(self, weights):
        super(Criterion, self).__init__()
        self.dc = metrics.BatchDiceLoss([1.0])  # weighted inversely by each volume proportion
        assert len(weights) == 6
        self.weights = [i/100 for i in weights]

    def compute_2nd_order_derivative(self, x):
        a = torch.Tensor([[[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                          [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                          [[1, 0, -1], [2, 0, -2], [1, 0, -1]]])
        a = a.view((1, 1, 3, 3, 3))
        G_x = nn.functional.conv3d(x, a)

        b = torch.Tensor([[[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                          [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                          [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]])
        b = b.view((1, 1, 3, 3, 3))
        G_y = nn.functional.conv3d(x, b)

        b = torch.Tensor([[[1, 2, 1], [1, 2, 1], [1, 2, 1]],
                          [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                          [[-1, -2, -1], [-1, -2, -1], [-1, -2, -1]]])
        b = b.view((1, 1, 3, 3, 3))
        G_z = nn.functional.conv3d(x, b)

        return torch.sqrt(torch.pow(G_x, 2) + torch.pow(G_y, 2) + torch.pow(G_z, 2))

    def forward(self, pr_core, gt_core,  pr_lesion, gt_lesion, pr_penu, gt_penu, output, out_c, out_p, mid_c, mid_p,
                grid_c, grid_p):
        loss = self.weights[0] * self.dc(pr_core, gt_core)
        loss += self.weights[2] * self.dc(pr_penu, gt_penu)
        loss += self.weights[1] * self.dc(pr_lesion, gt_lesion)
        loss += self.weights[3] * self.dc(out_c, out_p)
        loss += self.weights[4] * self.dc(mid_c, mid_p)

        for i in range(output.size()[1]-1):
            diff = output[:, i+1] - output[:, i]
            loss += self.weights[5] * torch.mean(torch.abs(diff) - diff)  # monotone

        _ = self.compute_2nd_order_derivative(grid_c) + self.compute_2nd_order_derivative(grid_p)

        return loss


def get_factors(batch, batchsize, sequence_length, sequence_thresholds):
    factor = torch.tensor([[0] * sequence_length] * batchsize, dtype=torch.float).cuda()
    t_core = batch[data.KEY_GLOBAL][:, 0]
    idx_core = tensor2index(t_core, sequence_thresholds)  # int(batch[data.KEY_GLOBAL][b, 0, :, :, :])
    for b in range(batchsize):
        factor[b, :int(idx_core[b])] = 1
        factor[b, int(idx_core[b]):] = torch.tensor(
            [1 - (sequence_thresholds[i] - float(t_core[b])) / (sequence_thresholds[-1] - float(t_core[b])) for i in
             range(int(idx_core[b]), sequence_length)], dtype=torch.float).cuda()
        factor[b, -1] = 0

    output_factors = []
    for i in range(sequence_length):
        fc = factor[:, i]
        zero = torch.zeros(fc.size(), requires_grad=False).cuda()
        ones = torch.ones(fc.size(), requires_grad=False).cuda()
        output_factors.append(torch.where(fc < 0.5, zero, ones).unsqueeze(1))

    return t_core, idx_core, factor, torch.cat(output_factors, dim=1).unsqueeze(2).unsqueeze(3).unsqueeze(4)


def combine_prediction(out_c, out_p, factor, output_factors, arg_combine):
    if arg_combine == 'split':
        pr = output_factors * out_c + (1 - output_factors) * out_p
    elif arg_combine == 'linear':
        pr = factor.unsqueeze(2).unsqueeze(3).unsqueeze(4) * out_c + (1 - factor).unsqueeze(2).unsqueeze(3).unsqueeze(
            4) * out_p
    else:
        pr = 0.5 * out_c + 0.5 * out_p
    return pr


def get_results(batch, batchsize, pr, out_c, out_p, lesion_pos, sequence_thresholds, t_core, idx_core):
    pr_core = []
    pr_lesion = []
    pr_penu = []
    pr_out_c = []
    pr_out_p = []
    pr_mid_c = []
    pr_mid_p = []

    if lesion_pos is not None:
        raise Exception('Cannot use lesion_pos gradient back prop with non-uniform index sampling!')
        index_lesion = lesion_pos * torch.tensor([list(range(sequence_length))] * batchsize).float().cuda()
        index_lesion = torch.sum(index_lesion, dim=1) / torch.sum(lesion_pos, dim=1)
        floor = torch.floor(index_lesion)
        ceil = torch.ceil(index_lesion)
        alpha = (index_lesion - floor)
        for b in range(batchsize):
            pr_lesion.append(alpha[b] * torch.index_select(pr[b], 0, floor[b].long()) + (1 - alpha[b]) * torch.index_select(pr[b], 0, ceil[b].long()))
            pr_out_c.append(alpha[b] * torch.index_select(out_c[b], 0, floor[b].long()) + (1 - alpha[b]) * torch.index_select(out_c[b], 0, ceil[b].long()))
            pr_out_p.append(alpha[b] * torch.index_select(out_p[b], 0, floor[b].long()) + (1 - alpha[b]) * torch.index_select(out_p[b], 0, ceil[b].long()))
    else:
        idx_lesion = tensor2index(t_core + batch[data.KEY_GLOBAL][:, 1], sequence_thresholds)
        idx_middle = tensor2index(t_core + (sequence_thresholds[-1] - t_core) / 2, sequence_thresholds)
        for b in range(batchsize):
            pr_lesion.append(pr[b, int(idx_lesion[b])])
            pr_out_c.append(out_c[b, int(idx_lesion[b])])
            pr_out_p.append(out_p[b, int(idx_lesion[b])])
            pr_mid_c.append(out_c[b, int(idx_middle[b])])
            pr_mid_p.append(out_p[b, int(idx_middle[b])])

    for b in range(batchsize):
        pr_core.append(pr[b, int(idx_core[b])])  # int(batch[data.KEY_GLOBAL][b, 0, :, :, :])
        pr_penu.append(pr[b, -1])

    return torch.stack(pr_lesion, dim=0).unsqueeze(1),\
           torch.stack(pr_core, dim=0).unsqueeze(1), torch.stack(pr_penu, dim=0).unsqueeze(1),\
           torch.stack(pr_out_c, dim=0).unsqueeze(1), torch.stack(pr_out_p, dim=0).unsqueeze(1),\
           torch.stack(pr_mid_c, dim=0).unsqueeze(1), torch.stack(pr_mid_p, dim=0).unsqueeze(1),\
           idx_lesion, idx_middle


def get_title(prefix, row, idx, batch, seq_thr, lesion_pos=None):
    if lesion_pos is not None:
        lesion_pos = int(lesion_pos[row])
    else:
        lesion_pos = 0
    suffix = ''
    if idx == int(batch[data.KEY_GLOBAL][row, 0, :, :, :]):
        suffix += ' [C]'
    if idx == int(batch[data.KEY_GLOBAL][row, 0, :, :, :]) + int(batch[data.KEY_GLOBAL][row, 1, :, :, :]):
        suffix += ' [L]'
    if idx == len(seq_thr)-1:
        suffix += ' [P]'
    return '{}{}'.format(str(seq_thr[idx]), suffix)


def visualise_batch(axarr, batch, gt, pr, grid_default, grid_c, grid_p, idx_lesion, n_visual_samples, sequence_length,
                    sequence_thresholds, init_offset, factor = 3):
    for row in range(n_visual_samples):
        titles = []
        core = gt[row, 0]
        com = np.round(ndi.center_of_mass(core)).astype(np.int)
        axarr[factor * row + init_offset, 0].imshow(core[com[0], :, :], vmin=0, vmax=1, cmap='gray')
        titles.append('CORE')
        axarr[factor * row + init_offset, 1].imshow(gt[row, 1, com[0], :, :], vmin=0, vmax=1, cmap='gray')
        titles.append('FUCT')
        axarr[factor * row + init_offset, 2].imshow(gt[row, 2, com[0], :, :], vmin=0, vmax=1, cmap='gray')
        titles.append('PENU')
        for i in range(sequence_length):
            axarr[factor * row + init_offset, i + 3].imshow(pr[row, i, com[0], :, :], vmin=0, vmax=1, cmap='gray')
            titles.append(get_title('Pr', row, i, batch, sequence_thresholds, idx_lesion))
        for ax, title in zip(axarr[factor * row], titles):
            ax.set_title(title, verticalalignment='top')
        titles = []

        for offset, grid in zip([1, 2], [grid_c, grid_p]):
            axarr[factor * row + offset + init_offset, 0].imshow(grid_default[row, 0, com[0], :, :], vmin=0, vmax=1, cmap='gray')
            titles.append('')
            axarr[factor * row + offset + init_offset, 1].imshow(grid_default[row, 0, com[0], :, :], vmin=0, vmax=1, cmap='gray')
            titles.append('')
            axarr[factor * row + offset + init_offset, 2].imshow(grid_default[row, 0, com[0], :, :], vmin=0, vmax=1, cmap='gray')
            titles.append('')
            for i in range(sequence_length):
                axarr[factor * row + offset + init_offset, i + 3].imshow(grid[row, i, com[0], :, :], vmin=0, vmax=1, cmap='gray')
                titles.append('')
            for ax, title in zip(axarr[factor * row + offset + init_offset], titles):
                ax.set_title(title, verticalalignment='top')
            titles = []
    return axarr


def process_batch(batch, batchsize, bi_net, criterion, arg_combine, sequence_length, sequence_thresholds, device):
    gt = batch[data.KEY_LABELS].to(device)

    t_core, idx_core, factor, output_factors = get_factors(batch,
                                                           batchsize,
                                                           sequence_length,
                                                           sequence_thresholds)

    out_c, out_p, lesion_pos, grid_c, grid_p = bi_net(gt[:, 0, :, :, :].unsqueeze(1),
                                                      gt[:, -1, :, :, :].unsqueeze(1),
                                                      batch[data.KEY_GLOBAL].to(device),
                                                      factor)

    pr = combine_prediction(out_c,
                            out_p,
                            factor,
                            output_factors,
                            arg_combine)

    pr_lesion, pr_core, pr_penu, pr_out_c, pr_out_p, pr_mid_c, pr_mid_p, idx_lesion, _ = get_results(batch,
                                                                                                     batchsize, pr,
                                                                                                     out_c, out_p,
                                                                                                     lesion_pos,
                                                                                                     sequence_thresholds,
                                                                                                     t_core, idx_core)

    loss = criterion(pr_core,
                     gt[:, 0, :, :, :].unsqueeze(1),  # torch.stack(gt_core, dim=0),
                     pr_lesion,
                     gt[:, 1, :, :, :].unsqueeze(1),  # torch.stack(gt_lesion, dim=0),
                     pr_penu,
                     gt[:, 2, :, :, :].unsqueeze(1),  # torch.stack(gt_penu, dim=0),
                     pr,
                     pr_out_c,
                     pr_out_p,
                     pr_mid_c,
                     pr_mid_p,
                     grid_c,
                     grid_p)

    return gt, pr, grid_c, grid_p, idx_lesion, loss


def main(arg_path, arg_length, arg_batchsize, arg_clinical, arg_commonfeature, arg_additional, arg_img2vec1,
         arg_vec2vec1, arg_grunet, arg_img2vec2, arg_vec2vec2, arg_addfactor, arg_softener, arg_loss,
         arg_epochs, arg_fold, arg_validsize, arg_seed, arg_combine, arg_clinical_grunet):

    print('arg_path, arg_length, arg_batchsize, arg_clinical, arg_commonfeature, arg_additional, arg_img2vec1,\
           arg_vec2vec1, arg_grunet, arg_img2vec2, arg_vec2vec2, arg_addfactor, arg_softener, arg_loss,\
           arg_epochs, arg_fold, arg_validsize, arg_seed, arg_combine, arg_clinical_grunet')
    print(arg_path, arg_length, arg_batchsize, arg_clinical, arg_commonfeature, arg_additional, arg_img2vec1,
          arg_vec2vec1, arg_grunet, arg_img2vec2, arg_vec2vec2, arg_addfactor, arg_softener, arg_loss,
          arg_epochs, arg_fold, arg_validsize, arg_seed, arg_combine, arg_clinical_grunet)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    zsize = 28  # change here for 2D/3D: 1 or 28
    input2d = (zsize == 1)
    convgru_kernel = 3
    if input2d:
        convgru_kernel = (1, 3, 3)
    batchsize = arg_batchsize
    sequence_thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7., 8., 9., 10.]
    assert len(sequence_thresholds) == arg_length
    sequence_length = arg_length
    #sequence_weight = [i / sum(range(arg_length + 1)) for i in range(1, arg_length + 1)]
    #sequence_thresh = [sum(sequence_weight[:i]) for i in range(arg_length + 1)]
    num_clinical_input = arg_clinical
    n_ch_feature_single = arg_commonfeature
    n_ch_affine_img2vec = arg_img2vec1  # first layer dim: 2 * n_ch_feature_single + 2 core/penu segmentation + 2 previous deform; list of length = 5
    n_ch_affine_vec2vec = arg_vec2vec1  # first layer dim: last layer dim of img2vec + 2 clinical scalars + (1 factor); list of arbitrary length > 1
    add_factor = arg_addfactor
    if add_factor:
        num_clinical_input += 1
    n_ch_additional_grid_input = arg_additional  # 1 core + 1 penumbra + 3 affine core + 3 affine penumbra + 2 previous deform
    n_ch_time_img2vec = arg_img2vec2  #[24, 25, 26, 28, 30]
    n_ch_time_vec2vec = arg_vec2vec2  #[32, 16, 1]
    n_ch_grunet = arg_grunet
    zslice = zsize // 2
    pad = (20, 20, 20)
    softening_kernel = arg_softener  # for isotropic real world space; must be odd numbers!
    n_visual_samples = min(4, batchsize)

    '''
    train_trafo = [data.UseLabelsAsImages(),
                   #data.PadImages(0,0,4,0),  TODO for 28 slices
                   data.HemisphericFlip(),
                   data.ElasticDeform2D(apply_to_images=True, random=0.95),
                   data.ToTensor()]
    valid_trafo = [data.UseLabelsAsImages(),
                   #data.PadImages(0,0,4,0),  TODO for 28 slices
                   data.ElasticDeform2D(apply_to_images=True, random=0.67, seed=0),
                   data.ToTensor()]
    
    ds_train, ds_valid = data.get_toy_seq_shape_training_data(train_trafo, valid_trafo,
                                                              [0, 1, 2, 3],  #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],  #
                                                              [4, 5, 6, 7],  #[32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43],  #
                                                              batchsize=batchsize, normalize=sequence_length, growth='fast',
                                                              zsize=zsize)
    '''

    '''
    train_trafo = [data.ResamplePlaneXY(.5),
                   data.Slice14(),
                   data.UseLabelsAsImages(),
                   data.HemisphericFlip(),
                   data.ElasticDeform2D(apply_to_images=True, random=0.95),
                   data.ClinicalTimeOnly(),
                   data.ToTensor()]
    
    ds_train, ds_valid = data.get_stroke_shape_training_data_2D(train_trafo, [1, 2, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30], batchsize=batchsize)
    '''

    modalities = ['_CBV_reg1_downsampled',
                  '_TTD_reg1_downsampled']
    labels = ['_CBVmap_subset_reg1_downsampled',
              '_FUCT_MAP_T_Samplespace_subset_reg1_downsampled',
              '_TTDmap_subset_reg1_downsampled']
    train_trafo = [data.ResamplePlaneXY(0.5),
                   data.UseLabelsAsImages(),
                   data.HemisphericFlip(),
                   data.ElasticDeform(apply_to_images=True),
                   data.ClinicalTimeOnly(),
                   data.ToTensor()]
    valid_trafo = [data.ResamplePlaneXY(0.5),
                   data.UseLabelsAsImages(),
                   data.HemisphericFlipFixedToCaseId(14),
                   data.ClinicalTimeOnly(),
                   data.ToTensor()]
    '''
    ds_train, ds_valid = data.get_stroke_shape_training_data(modalities, labels, train_trafo, valid_trafo,
                                                             list(range(32)), ratio=0.3, seed=4, batchsize=batchsize,
                                                             split=True)
    '''
    ds_train, ds_valid = data.get_stroke_prediction_training_data(modalities, labels, train_trafo, valid_trafo,
                                                                  arg_fold, arg_validsize, batchsize=arg_batchsize,
                                                                  seed=arg_seed, split=True)


    assert not n_ch_grunet or n_ch_grunet[0] == 2 * n_ch_feature_single + n_ch_additional_grid_input
    assert not n_ch_time_img2vec or n_ch_time_img2vec[0] == 2 * n_ch_feature_single + n_ch_additional_grid_input
    bi_net = BidirectionalSequence(n_ch_feature_single, n_ch_affine_img2vec, n_ch_affine_vec2vec, n_ch_time_img2vec,
                                   n_ch_time_vec2vec, n_ch_grunet, num_clinical_input, kernel_size=convgru_kernel,
                                   seq_len=sequence_length, batch_size=batchsize, depth2d=input2d, add_factor=add_factor,
                                   soften_kernel=softening_kernel, clinical_grunet=arg_clinical_grunet).to(device)

    params = [p for p in bi_net.parameters() if p.requires_grad]
    print('# optimizing params', sum([p.nelement() * p.requires_grad for p in params]),
          '/ total: Bi-RNN-Sequence', sum([p.nelement() for p in bi_net.parameters()]))

    criterion = Criterion(arg_loss)
    optimizer = torch.optim.Adam(params, lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.1)

    loss_train = []
    loss_valid = []

    for epoch in range(0, arg_epochs):
        scheduler.step()
        f, axarr = plt.subplots(n_visual_samples * 6, sequence_length + 3)
        loss_mean = 0
        inc = 0

        ### Train ###

        is_train = True
        bi_net.train(is_train)
        with torch.set_grad_enabled(is_train):

            for batch in ds_train:
                gt, pr, grid_c, grid_p, idx_lesion, loss = process_batch(batch, batchsize, bi_net, criterion,
                                                                         arg_combine, sequence_length,
                                                                         sequence_thresholds, device)

                loss_mean += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                inc += 1

                torch.cuda.empty_cache()

            loss_train.append(loss_mean/inc)

            axarr = visualise_batch(axarr,
                                    batch,
                                    gt.cpu().detach().numpy(),
                                    pr.cpu().detach().numpy(),
                                    bi_net.visual_grid.cpu().detach().numpy(),
                                    grid_c.cpu().detach().numpy(),
                                    grid_p.cpu().detach().numpy(),
                                    idx_lesion,
                                    n_visual_samples,
                                    sequence_length,
                                    sequence_thresholds,
                                    init_offset=0)
            del batch

        del pr
        del gt
        del grid_c
        del grid_p
        del idx_lesion
        del loss

        ### Validate ###

        inc = 0
        loss_mean = 0
        is_train = False
        optimizer.zero_grad()
        bi_net.train(is_train)
        with torch.set_grad_enabled(is_train):

            for batch in ds_valid:
                gt, pr, grid_c, grid_p, idx_lesion, loss = process_batch(batch, batchsize, bi_net, criterion,
                                                                         arg_combine, sequence_length,
                                                                         sequence_thresholds, device)

                loss_mean += loss.item()

                inc += 1

                torch.cuda.empty_cache()

            loss_valid.append(loss_mean/inc)

            axarr = visualise_batch(axarr,
                                    batch,
                                    gt.cpu().detach().numpy(),
                                    pr.cpu().detach().numpy(),
                                    bi_net.visual_grid.cpu().detach().numpy(),
                                    grid_c.cpu().detach().numpy(),
                                    grid_p.cpu().detach().numpy(),
                                    idx_lesion,
                                    n_visual_samples,
                                    sequence_length,
                                    sequence_thresholds,
                                    init_offset=6)
            del batch

        print('Epoch', epoch, 'last batch training loss:', loss_train[-1], '\tvalidation batch loss:', loss_valid[-1])

        if epoch % 5 == 0:
            torch.save(bi_net, arg_path.format('latest','model'))

        for ax in axarr.flatten():
            ax.title.set_fontsize(3)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
        f.subplots_adjust(hspace=0.05)
        f.savefig(arg_path.format(str(epoch),'png'), bbox_inches='tight', dpi=300)

        del f
        del axarr

        if epoch > 0:
            fig, plot = plt.subplots()
            epochs = range(1, epoch + 2)
            plot.plot(epochs, loss_train, 'r-')
            plot.plot(epochs, loss_valid, 'b-')
            plot.set_ylabel('Loss Training (r) & Validation (b)')
            fig.savefig(arg_path.format('plots','png'), bbox_inches='tight', dpi=300)
            del plot
            del fig


if __name__ == '__main__':
    print(datetime.datetime.now())
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Output path pattern', default='/share/data_zoe1/lucas/NOT_IN_BACKUP/tmp/grunet_exp_{}.{}')
    parser.add_argument('--length', type=int, help='Sequence prediction length of recurrent network', default=11)
    parser.add_argument('--batchsize', type=int, help='Batch size', default=2)
    parser.add_argument('--clinical', type=int, help='Take the first <CLINICAL> channels of clinical input vector', default=2)
    parser.add_argument('--commonfeature', type=int, help='Number of channels for common input features', default=5)
    parser.add_argument('--additional', type=int, help='Number of additional grid input channels to GRUnet', default=14)
    parser.add_argument('--img2vec1', type=int, nargs='*', help='Number of channels Image-to-vector AFFINE MODULE', default=[18, 19, 20, 21, 22])
    parser.add_argument('--vec2vec1', type=int, nargs='*', help='Number of channels Vector-to-vector AFFINE MODULE', default=[24, 20, 20, 24])
    parser.add_argument('--grunet', type=int, nargs='*', help='Number of channels GRUnet MODULE', default=[24, 28, 32, 28, 24])
    parser.add_argument('--img2vec2', type=int, nargs='*', help='Number of channels Image-to-vector LESION TIME MODULE', default=None)
    parser.add_argument('--vec2vec2', type=int, nargs='*', help='Number of channels Vector-to-vector LESION TIME MODULE', default=None)
    parser.add_argument('--addfactor', action='store_true', help='Add interpolation factor core<->penumbra to clinical vector', default=False)
    parser.add_argument('--nonlinclinical', action='store_true', help='Use upsampled clinical also for non-linear GRUnet', default=False)
    parser.add_argument('--softener', type=int, nargs='+', help='Average Pooling kernel, must be odd numbers!', default=[5, 23, 23,])
    parser.add_argument('--loss', type=int, nargs='+', help='Loss weights (%)', default=[10, 44, 10, 25, 1])
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=200)
    parser.add_argument('--fold', type=int, nargs='+', help='Ids of this training fold', default=[])
    parser.add_argument('--validsize', type=float, help='Valiation set fraction', default=0.275)
    parser.add_argument('--seed', type=int, help='Randomization seed', default=4)
    parser.add_argument('--combine', default='add', const='add', nargs='?', choices=['add', 'linear', 'split'], help='How to combine prediction from core and penumbra? Uniformly add both, linearly interpolate continously between both, or hard split in the middle.')
    args = parser.parse_args()
    assert len(args.fold) >= args.batchsize
    main(args.path, args.length, args.batchsize, args.clinical, args.commonfeature, args.additional, args.img2vec1,
         args.vec2vec1, args.grunet, args.img2vec2, args.vec2vec2, args.addfactor, args.softener, args.loss,
         args.epochs, args.fold, args.validsize, args.seed, args.combine, args.nonlinclinical)
    print(datetime.datetime.now())
