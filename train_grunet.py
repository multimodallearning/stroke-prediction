import torch
import torch.nn as nn
from common import data, metrics
from GRUnet import GRUnet, GRUnetBidirectionalSequence
import matplotlib.pyplot as plt


class Criterion(nn.Module):
    def __init__(self, weights):
        super(Criterion, self).__init__()
        self.dc = metrics.BatchDiceLoss(weights)  # weighted inversely by each volume proportion
        self.dc_mid = metrics.BatchDiceLoss([1.0])  # weighted inversely by each volume proportion

    def forward(self, pred, target, output, out_c, out_p):
        loss = 0.5 * self.dc(pred, target)
        loss += 0.25 * self.dc_mid(out_c, out_p)

        for i in range(output.size()[1]-1):
            diff = output[:, i+1] - output[:, i]
            loss += 0.025 * torch.mean(torch.abs(diff) - diff)  # monotone

        return loss


def get_title(prefix, idx, batch, seq_len):
    suffix = ''
    if idx == int(batch[data.KEY_GLOBAL][row, 0, :, :, :]):
        suffix += ' core'
    elif idx == int(batch[data.KEY_GLOBAL][row, 1, :, :, :]):
        suffix += ' fuct'
    elif idx == seq_len-1:
        suffix += ' penu'
    return prefix + '[' + str(idx) + ']' + suffix


zsize = 1  # change here for 2D/3D: 1 or 28
input2d = (zsize == 1)
convgru_kernel = 3
if input2d:
    convgru_kernel = (1, 3, 3)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sequence_length = 10
num_clinical_input = 2
batchsize = 4
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
                                                          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],  #[0, 1, 2, 3],  #
                                                          [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43],  #[4, 5, 6, 7],  #
                                                          batchsize=batchsize, normalize=sequence_length, growth='fast',
                                                          zsize=zsize)

grunet = GRUnetBidirectionalSequence(GRUnet(clinical_size=num_clinical_input,
                                            hidden_sizes=[16, 32, 64, 32, 16],
                                            kernel_sizes=[convgru_kernel] * 5),
                                     GRUnet(clinical_size=num_clinical_input,
                                            hidden_sizes=[16, 32, 64, 32, 16],
                                            kernel_sizes=[convgru_kernel] * 5),
                                            sequence_length).to(device)

params = [p for p in grunet.parameters() if p.requires_grad]
print('# optimizing params', sum([p.nelement() * p.requires_grad for p in params]),
      '/ total: GRUnet', sum([p.nelement() for p in grunet.parameters()]))

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
    grunet.train(is_train)
    with torch.set_grad_enabled(is_train):

        for batch in ds_train:
            gt = batch[data.KEY_LABELS].to(device)

            t_core = []
            for b in range(batchsize):
                t_core.append(int(batch[data.KEY_GLOBAL][b, 0, :, :, :]))

            out_c, out_p, out_f = grunet(gt[:, int(batch[data.KEY_GLOBAL][b, 0, :, :, :]), :, :, :].unsqueeze(1),
                                  gt[:, -1, :, :, :].unsqueeze(1),
                                  batch[data.KEY_GLOBAL].to(device),
                                  t_core)
            output = out_f * out_c + (1-out_f) * out_p

            mask = torch.zeros(gt.size()).byte()
            mask_t_lesion = torch.zeros(gt.size()).byte()
            for b in range(batchsize):
                mask[b, t_core[b], :, :, :] = 1  # core
                mask[b, int(batch[data.KEY_GLOBAL][b, 1, :, :, :]), :, :, :] = 1  # lesion
                mask_t_lesion[b, int(batch[data.KEY_GLOBAL][b, 1, :, :, :]), :, :, :] = 1  # lesion
                mask[b, -1, :, :, :] = 1  # penumbra

            loss = criterion(output[mask].view(batchsize, 3, zsize, 128, 128),
                             gt[mask].view(batchsize, 3, zsize, 128, 128),
                             output,
                             out_c[mask_t_lesion].view(batchsize, 1, zsize, 128, 128),
                             out_p[mask_t_lesion].view(batchsize, 1, zsize, 128, 128))
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
                titles.append(get_title('GT', i, batch, sequence_length))
            for i in range(sequence_length):
                axarr[row, i + sequence_length].imshow(output.cpu().detach().numpy()[row, i, zslice, :, :], vmin=0, vmax=1, cmap='gray')
                titles.append(get_title('Pr', i, batch, sequence_length))
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
    grunet.train(is_train)
    with torch.set_grad_enabled(is_train):

        for batch in ds_valid:
            gt = batch[data.KEY_LABELS].to(device)

            t_core = []
            for b in range(batchsize):
                t_core.append(int(batch[data.KEY_GLOBAL][b, 0, :, :, :]))

            out_c, out_p, out_f = grunet(gt[:, int(batch[data.KEY_GLOBAL][b, 0, :, :, :]), :, :, :].unsqueeze(1),
                                  gt[:, -1, :, :, :].unsqueeze(1),
                                  batch[data.KEY_GLOBAL].to(device),
                                  t_core)
            output = out_f * out_c + (1-out_f) * out_p

            mask = torch.zeros(gt.size()).byte()
            mask_t_lesion = torch.zeros(gt.size()).byte()
            for b in range(batchsize):
                mask[b, t_core[b], :, :, :] = 1  # core
                mask[b, int(batch[data.KEY_GLOBAL][b, 1, :, :, :]), :, :, :] = 1  # lesion
                mask_t_lesion[b, int(batch[data.KEY_GLOBAL][b, 1, :, :, :]), :, :, :] = 1  # lesion
                mask[b, -1, :, :, :] = 1  # penumbra

            loss = criterion(output[mask].view(batchsize, 3, zsize, 128, 128),
                             gt[mask].view(batchsize, 3, zsize, 128, 128),
                             output,
                             out_c[mask_t_lesion].view(batchsize, 1, zsize, 128, 128),
                             out_p[mask_t_lesion].view(batchsize, 1, zsize, 128, 128))
            loss_mean += loss.item()

            inc += 1

        loss_valid.append(loss_mean/inc)

        for row in range(n_visual_samples):
            titles = []
            for i in range(sequence_length):
                axarr[row + n_visual_samples, i].imshow(gt.cpu().detach().numpy()[row, i, zslice, :, :], vmin=0, vmax=1, cmap='gray')
                titles.append(get_title('GT', i, batch, sequence_length))
            for i in range(sequence_length):
                axarr[row + n_visual_samples, i + sequence_length].imshow(output.cpu().detach().numpy()[row, i, zslice, :, :], vmin=0, vmax=1, cmap='gray')
                titles.append(get_title('Pr', i, batch, sequence_length))
            for ax, title in zip(axarr[row + n_visual_samples], titles):
                ax.set_title(title)
        del batch

    print('Epoch', epoch, 'last batch training loss:', loss_train[-1], '\tvalidation batch loss:', loss_valid[-1])

    if epoch % 5 == 0:
        torch.save(grunet, '/share/data_zoe1/lucas/NOT_IN_BACKUP/tmp/GRUnet.model')

    for ax in axarr.flatten():
        ax.title.set_fontsize(3)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    f.subplots_adjust(hspace=0.05)
    f.savefig('/share/data_zoe1/lucas/NOT_IN_BACKUP/tmp/GRUnet' + str(epoch) + '.png', bbox_inches='tight', dpi=300)

    del f
    del axarr

    if epoch > 0:
        fig, plot = plt.subplots()
        epochs = range(1, epoch + 2)
        plot.plot(epochs, loss_train, 'r-')
        plot.plot(epochs, loss_valid, 'b-')
        plot.set_ylabel('Loss Training (r) & Validation (b)')
        fig.savefig('/share/data_zoe1/lucas/NOT_IN_BACKUP/tmp/GRUnet_plots.png', bbox_inches='tight', dpi=300)
        del plot
        del fig
