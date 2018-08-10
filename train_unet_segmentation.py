import torch
import datetime
from learner.UnetSegmentationLearner import UnetSegmentationLearner
from common.model.Unet3D import Unet3D
from common import data, util, metrics


def train():
    args = util.get_args_unet_training()

    # Params / Config
    batchsize = 6  # 17 training, 6 validation
    learning_rate = 1e-3
    momentums_cae = (0.99, 0.999)
    criterion = metrics.BatchDiceLoss([1.0])  # nn.BCELoss()
    path_training_metrics = args.continuetraining
    path_saved_model = args.unetpath
    channels = args.channels
    pad = args.padding
    cuda = True

    # Unet model
    unet = Unet3D(channels)
    if cuda:
        unet = unet.cuda()

    # Model params
    params = [p for p in unet.parameters() if p.requires_grad]
    print('# optimizing params', sum([p.nelement() * p.requires_grad for p in params]),
          '/ total: unet', sum([p.nelement() for p in unet.parameters()]))

    # Optimizer with scheduler
    optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=1e-5, betas=momentums_cae)
    if args.lrsteps:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lrsteps)
    else:
        scheduler = None

    # Data
    train_transform = [data.ResamplePlaneXY(args.xyresample),
                       data.HemisphericFlipFixedToCaseId(split_id=args.hemisflipid),
                       data.PadImages(pad[0], pad[1], pad[2], pad_value=0),
                       data.RandomPatch(104, 104, 68, pad[0], pad[1], pad[2]),
                       data.ToTensor()]
    valid_transform = [data.ResamplePlaneXY(args.xyresample),
                       data.HemisphericFlipFixedToCaseId(split_id=args.hemisflipid),
                       data.PadImages(pad[0], pad[1], pad[2], pad_value=0),
                       data.RandomPatch(104, 104, 68, pad[0], pad[1], pad[2]),
                       data.ToTensor()]
    ds_train, ds_valid = data.get_stroke_shape_training_data(train_transform, valid_transform, args.fold,
                                                             args.validsetsize, batchsize=batchsize)
    print('Size training set:', len(ds_train.sampler.indices), '| Size validation set:', len(ds_valid.sampler.indices),
          '| Size batch:', batchsize)
    print('# training batches:', len(ds_train), '| # validation batches:', len(ds_valid))

    # Training
    learner = UnetSegmentationLearner(ds_train, ds_valid, unet, path_saved_model, optimizer, scheduler,
                                      n_epochs=args.epochs, path_training_metrics=path_training_metrics,
                                      path_outputs_base=args.outbasepath, criterion=criterion)
    learner.run_training()


if __name__ == '__main__':
    print(datetime.datetime.now())
    train()
    print(datetime.datetime.now())
