import torch
import torch.nn as nn
from torch.autograd import Variable
from common import data, metrics
from convgru import ConvGRU
import matplotlib.pyplot as plt


class Criterion(nn.Module):
    def __init__(self, weights, seq_len=10):
        super(Criterion, self).__init__()
        self.dc = metrics.BatchDiceLoss(weights)  # weighted inversely by each volume proportion
        self.seq_len = seq_len

    def forward(self, pred, target):
        loss = self.dc(pred, target)

        for i in range(self.seq_len-1):
            diff = pred[:, i+1] - pred[:, i]
            loss += torch.mean(torch.abs(diff) - diff)  # monotone

        return loss


modalities = ['_CBV_reg1_downsampled',
              '_TTD_reg1_downsampled']
labels = ['_CBVmap_subset_reg1_downsampled',
          '_FUCT_MAP_T_Samplespace_subset_reg1_downsampled',
          '_TTDmap_subset_reg1_downsampled']

sequence_length = 10
num_layers = 3
batchsize = 2
zslice = 14
pad = (20, 20, 20)
n_visual_samples = min(4, batchsize)

train_trafo = [data.UseLabelsAsImages(),
               data.PadImages(pad[0], pad[1], pad[2], pad_value=0),
               data.HemisphericFlip(),
               data.ElasticDeform(apply_to_images=True, random=0.75),
               data.ToTensor()]
valid_trafo = [data.UseLabelsAsImages(),
               data.PadImages(pad[0], pad[1], pad[2], pad_value=0),
               data.ToTensor()]

ds_train, _ = data.get_toy_seq_shape_training_data(train_trafo, valid_trafo,
                                                   [0, 1, 2, 3, 4, 5, 6, 7],  #, 8, 9, 10, 11, 12, 13, 14, 15],
                                                   [],  #16, 17, 18, 19],
                                                   batchsize=batchsize, normalize=sequence_length)

convgru = ConvGRU(input_size=1, hidden_sizes=[12]*(num_layers-1) + [1], kernel_sizes=[3]*(num_layers-1) + [1], n_layers=num_layers).cuda()

params = [p for p in convgru.parameters() if p.requires_grad]
print('# optimizing params', sum([p.nelement() * p.requires_grad for p in params]),
      '/ total: Unet-GRU-RNN', sum([p.nelement() for p in convgru.parameters()]))

criterion = Criterion([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], seq_len=sequence_length)
optimizer = torch.optim.Adam(params, lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.1)

loss_train = []
loss_valid = []

for epoch in range(0, 225):
    scheduler.step()
    f, axarr = plt.subplots(n_visual_samples, 10)
    loss_mean = 0
    inc = 0
    train = True
    convgru.train(train)

    for batch in ds_train:
        gt = Variable(batch[data.KEY_LABELS], volatile=not train).cuda()

        del batch

        hidden = None
        output = []
        for i in range(sequence_length):
            hidden = convgru(gt[:, i, :, :, :].unsqueeze(1), hidden=hidden)
            output.append(hidden[-1])
        output = torch.cat(output, dim=1)

        loss = criterion(output, gt)
        loss_mean += float(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        inc += 1

    loss_train.append(loss_mean/inc)

    for row in range(n_visual_samples):
        axarr[row, 0].imshow(gt.cpu().data.numpy()[row, 0, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        axarr[row, 1].imshow(gt.cpu().data.numpy()[row, 2, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        axarr[row, 2].imshow(gt.cpu().data.numpy()[row, 4, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        axarr[row, 3].imshow(gt.cpu().data.numpy()[row, 7, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        axarr[row, 4].imshow(gt.cpu().data.numpy()[row, 9, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        axarr[row, 5].imshow(output.cpu().data.numpy()[row, 0, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        axarr[row, 6].imshow(output.cpu().data.numpy()[row, 2, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        axarr[row, 7].imshow(output.cpu().data.numpy()[row, 4, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        axarr[row, 8].imshow(output.cpu().data.numpy()[row, 7, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        axarr[row, 9].imshow(output.cpu().data.numpy()[row, 9, zslice, :, :], vmin=0, vmax=1, cmap='gray')
        titles = ['GT[0]', 'GT[2]', 'GT[4]', 'GT[7]', 'GT[9]', 'Pr[0]', 'Pr[2]', 'Pr[4]', 'Pr[7]', 'Pr[9]']
        for ax, title in zip(axarr[row], titles):
            ax.set_title(title)

    del output
    del hidden
    del loss
    del gt

    optimizer.zero_grad()

    train = False
    convgru.train(train)

    loss_valid.append(0.0)

    print('Epoch', epoch, 'last batch training loss:', loss_train[-1], '\tvalidation batch loss:', loss_valid[-1])

    if epoch % 5 == 0:
        torch.save(convgru, '/share/data_zoe1/lucas/NOT_IN_BACKUP/tmp/convgru.model')
    f.subplots_adjust(hspace=0.05)
    f.savefig('/share/data_zoe1/lucas/NOT_IN_BACKUP/tmp/convgru' + str(epoch) + '.png', bbox_inches='tight', dpi=300)

    del f
    del axarr

    if epoch > 0:
        fig, plot = plt.subplots()
        epochs = range(1, epoch + 2)
        plot.plot(epochs, loss_train, 'r-')
        plot.plot(epochs, loss_valid, 'b-')
        plot.set_ylabel('Loss Training (r) & Validation (b)')
        fig.savefig('/share/data_zoe1/lucas/NOT_IN_BACKUP/tmp/convgru_plots.png', bbox_inches='tight', dpi=300)
        del plot
        del fig
