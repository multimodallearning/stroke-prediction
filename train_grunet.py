import torch
import torch.nn as nn
from common import data, metrics
from GRUnet_2 import BidirectionalSequence
import matplotlib.pyplot as plt


class Criterion(nn.Module):
    def __init__(self, weights):
        super(Criterion, self).__init__()
        self.dc = metrics.BatchDiceLoss(weights)  # weighted inversely by each volume proportion
        self.dc_mid = metrics.BatchDiceLoss([1.0])  # weighted inversely by each volume proportion
        #self.l1 = nn.L1Loss()
        self.ce = nn.CrossEntropyLoss()

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

    def forward(self, pred, target, output, out_c, out_p, pr_t_lesion, t_lesion):
        loss = 0.5 * self.dc(pred, target)
        loss += 0.2 * self.dc_mid(out_c, out_p)

        for i in range(output.size()[1]-1):
            diff = output[:, i+1] - output[:, i]
            loss += 0.02 * torch.mean(torch.abs(diff) - diff)  # monotone

        #loss += 0.2 * self.l1(pr_t_lesion, t_lesion)
        loss += 0.2 * self.ce(pr_t_lesion, torch.argmax(t_lesion, dim=1))
        #loss += 0.2 * self.ce(torch.stack((1-marker_pred.flatten(),
        #                                   marker_pred.flatten()), dim=1), marker_target.flatten().long())

        return loss


def get_title(prefix, row, idx, batch, seq_len, time):
    suffix = ''
    if idx == int(batch[data.KEY_GLOBAL][row, 0, :, :, :]):
        suffix += ' C'
    elif idx == int(batch[data.KEY_GLOBAL][row, 1, :, :, :]):
        suffix += ' L'
    elif idx == seq_len-1:
        suffix += ' P'
    return '{}({:1.1f}){}'.format(str(idx), float(time[row, idx]), suffix)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
zsize = 1  # change here for 2D/3D: 1 or 28
input2d = (zsize == 1)
convgru_kernel = 3
if input2d:
    convgru_kernel = (1, 3, 3)
batchsize = 4
sequence_length = 10
num_clinical_input = 2
n_ch_feature_single = 8
n_ch_affine_img2vec = [18, 20, 22, 26, 30]  # first layer dim: 2 * n_ch_feature_single + 2 core/penu segmentation; list of length = 5
n_ch_affine_vec2vec = [32, 28, 24]  # first layer dim: last layer dim of img2vec + 2 clinical scalars; list of arbitrary length > 1
n_ch_additional_grid_input = 8  # 1 core + 1 penumbra + 3 affine core + 3 affine penumbra
n_ch_time_img2vec = [24, 25, 26, 28, 30]  #
n_ch_time_vec2vec = [32, 16, 1]  #
n_ch_grunet = [24, 28, 32, 28, 24]
zslice = zsize // 2
pad = (20, 20, 20)
n_visual_samples = min(4, batchsize)

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
                                                          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],  #[0, 1, 2, 3],
                                                          [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43],  #[4, 5, 6, 7],
                                                          batchsize=batchsize, normalize=sequence_length, growth='fast',
                                                          zsize=zsize)

assert n_ch_grunet[0] == 2 * n_ch_feature_single + n_ch_additional_grid_input
assert n_ch_time_img2vec[0] == 2 * n_ch_feature_single + n_ch_additional_grid_input
bi_net = BidirectionalSequence(n_ch_feature_single, n_ch_affine_img2vec, n_ch_affine_vec2vec, n_ch_time_img2vec,
                               n_ch_time_vec2vec, n_ch_grunet, num_clinical_input, kernel_size=convgru_kernel,
                               seq_len=sequence_length).to(device)

params = [p for p in bi_net.parameters() if p.requires_grad]
print('# optimizing params', sum([p.nelement() * p.requires_grad for p in params]),
      '/ total: GRUnet', sum([p.nelement() for p in bi_net.parameters()]))

criterion = Criterion([0.1, 0.8, 0.1])
optimizer = torch.optim.Adam(params, lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.1)

loss_train = []
loss_valid = []

for epoch in range(0, 200):
    scheduler.step()
    f, axarr = plt.subplots(n_visual_samples * 2, sequence_length * 2)
    loss_mean = 0
    inc = 0

    ### Train ###

    is_train = True
    bi_net.train(is_train)
    with torch.set_grad_enabled(is_train):

        for batch in ds_train:
            gt = batch[data.KEY_LABELS].to(device)

            factor = torch.tensor([[0] * sequence_length] * batchsize, dtype=torch.float).cuda()
            marker_target = torch.tensor(factor)
            for b in range(batchsize):
                t_lesion = int(batch[data.KEY_GLOBAL][b, 1, :, :, :])
                t_core = int(batch[data.KEY_GLOBAL][b, 0, :, :, :])
                length = sequence_length - t_core
                t_half = t_core + length // 2
                factor[b, :t_core] = 1
                factor[b, t_core:t_half] = torch.tensor([1 - i / length for i in range(length//2)], dtype=torch.float).cuda()
                factor[b, t_half:] = torch.tensor([(length//2) / length - i / length for i in range(length - length//2)], dtype=torch.float).cuda()
                marker_target[b, t_lesion] = 1

            out_c, out_p, pr_t_lesion = bi_net(gt[:, int(batch[data.KEY_GLOBAL][b, 0, :, :, :]), :, :, :].unsqueeze(1),
                                               gt[:, -1, :, :, :].unsqueeze(1),
                                               batch[data.KEY_GLOBAL].to(device),
                                               factor)
            #output = factor.unsqueeze(2).unsqueeze(3).unsqueeze(4) * out_c + (1-factor).unsqueeze(2).unsqueeze(3).unsqueeze(4) * out_p
            output = 0.5 * out_c + 0.5 * out_p

            mask = torch.zeros(gt.size()).byte()
            mask_t_lesion = torch.zeros(gt.size()).byte()
            for b in range(batchsize):
                mask[b, int(batch[data.KEY_GLOBAL][b, 0, :, :, :]), :, :, :] = 1  # core
                mask[b, int(batch[data.KEY_GLOBAL][b, 1, :, :, :]), :, :, :] = 1  # lesion
                mask_t_lesion[b, int(batch[data.KEY_GLOBAL][b, 1, :, :, :]), :, :, :] = 1  # lesion
                mask[b, -1, :, :, :] = 1  # penumbra

            loss = criterion(output[mask].view(batchsize, 3, zsize, 128, 128),
                             gt[mask].view(batchsize, 3, zsize, 128, 128),
                             output,
                             out_c[mask_t_lesion].view(batchsize, 1, zsize, 128, 128),
                             out_p[mask_t_lesion].view(batchsize, 1, zsize, 128, 128),
                             pr_t_lesion,
                             marker_target)
            loss_mean += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            inc += 1

        loss_train.append(loss_mean/inc)

        for row in range(n_visual_samples):
            titles = []
            for i in range(sequence_length):
                axarr[row, i].imshow(gt.cpu().detach().numpy()[row, i, zslice, :, :], vmin=0, vmax=1, cmap='gray')
                titles.append(get_title('GT', row, i, batch, sequence_length, marker_target))
            for i in range(sequence_length):
                axarr[row, i + sequence_length].imshow(output.cpu().detach().numpy()[row, i, zslice, :, :], vmin=0, vmax=1, cmap='gray')
                titles.append(get_title('Pr', row, i, batch, sequence_length, pr_t_lesion))
            for ax, title in zip(axarr[row], titles):
                ax.set_title(title)
        del batch

    del output
    del loss
    del gt

    ### Validate ###

    inc = 0
    loss_mean = 0
    is_train = False
    optimizer.zero_grad()
    bi_net.train(is_train)
    with torch.set_grad_enabled(is_train):

        for batch in ds_valid:
            gt = batch[data.KEY_LABELS].to(device)

            factor = torch.tensor([[0] * sequence_length] * batchsize, dtype=torch.float).cuda()
            marker_target = torch.tensor(factor)
            for b in range(batchsize):
                t_lesion = int(batch[data.KEY_GLOBAL][b, 1, :, :, :])
                t_core = int(batch[data.KEY_GLOBAL][b, 0, :, :, :])
                length = sequence_length - t_core
                t_half = t_core + length // 2
                factor[b, :t_core] = 1
                factor[b, t_core:t_half] = torch.tensor([1 - i / length for i in range(length//2)], dtype=torch.float).cuda()
                factor[b, t_half:] = torch.tensor([(length//2) / length - i / length for i in range(length - length//2)], dtype=torch.float).cuda()
                marker_target[b, t_lesion] = 1

            out_c, out_p, pr_t_lesion = bi_net(gt[:, int(batch[data.KEY_GLOBAL][b, 0, :, :, :]), :, :, :].unsqueeze(1),
                                               gt[:, -1, :, :, :].unsqueeze(1),
                                               batch[data.KEY_GLOBAL].to(device),
                                               factor)
            #output = factor.unsqueeze(2).unsqueeze(3).unsqueeze(4) * out_c + (1-factor).unsqueeze(2).unsqueeze(3).unsqueeze(4) * out_p
            output = 0.5 * out_c + 0.5 * out_p

            mask = torch.zeros(gt.size()).byte()
            mask_t_lesion = torch.zeros(gt.size()).byte()
            for b in range(batchsize):
                mask[b, int(batch[data.KEY_GLOBAL][b, 0, :, :, :]), :, :, :] = 1  # core
                mask[b, int(batch[data.KEY_GLOBAL][b, 1, :, :, :]), :, :, :] = 1  # lesion
                mask_t_lesion[b, int(batch[data.KEY_GLOBAL][b, 1, :, :, :]), :, :, :] = 1  # lesion
                mask[b, -1, :, :, :] = 1  # penumbra

            loss = criterion(output[mask].view(batchsize, 3, zsize, 128, 128),
                             gt[mask].view(batchsize, 3, zsize, 128, 128),
                             output,
                             out_c[mask_t_lesion].view(batchsize, 1, zsize, 128, 128),
                             out_p[mask_t_lesion].view(batchsize, 1, zsize, 128, 128),
                             pr_t_lesion,
                             marker_target)
            loss_mean += loss.item()

            inc += 1

        loss_valid.append(loss_mean/inc)

        for row in range(n_visual_samples):
            titles = []
            for i in range(sequence_length):
                axarr[row + n_visual_samples, i].imshow(gt.cpu().detach().numpy()[row, i, zslice, :, :], vmin=0, vmax=1, cmap='gray')
                titles.append(get_title('GT', row, i, batch, sequence_length, marker_target))
            for i in range(sequence_length):
                axarr[row + n_visual_samples, i + sequence_length].imshow(output.cpu().detach().numpy()[row, i, zslice, :, :], vmin=0, vmax=1, cmap='gray')
                titles.append(get_title('Pr', row, i, batch, sequence_length, pr_t_lesion))
            for ax, title in zip(axarr[row + n_visual_samples], titles):
                ax.set_title(title)
        del batch

    print('Epoch', epoch, 'last batch training loss:', loss_train[-1], '\tvalidation batch loss:', loss_valid[-1])

    if epoch % 5 == 0:
        torch.save(bi_net, '/share/data_zoe1/lucas/NOT_IN_BACKUP/tmp/_GRUnet2.model')

    for ax in axarr.flatten():
        ax.title.set_fontsize(3)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    f.subplots_adjust(hspace=0.05)
    f.savefig('/share/data_zoe1/lucas/NOT_IN_BACKUP/tmp/_GRUnet2_' + str(epoch) + '.png', bbox_inches='tight', dpi=300)

    del f
    del axarr

    if epoch > 0:
        fig, plot = plt.subplots()
        epochs = range(1, epoch + 2)
        plot.plot(epochs, loss_train, 'r-')
        plot.plot(epochs, loss_valid, 'b-')
        plot.set_ylabel('Loss Training (r) & Validation (b)')
        fig.savefig('/share/data_zoe1/lucas/NOT_IN_BACKUP/tmp/_GRUnet2_plots.png', bbox_inches='tight', dpi=300)
        del plot
        del fig