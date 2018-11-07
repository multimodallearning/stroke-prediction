import torch
import datetime
from learner.CaeReconstructionOnlyLearner import CaeReconstructionOnlyLearner
from common.model.Cae3D import Cae3D, Enc3D, Enc3DStep, Dec3D
from common import data, util, metrics


def train(args):
    # Params / Config
    use_validation = not args.steplearning
    learning_rate = 1e-3
    momentums_cae = (0.9, 0.999)
    weight_decay = 1e-5
    criterion = metrics.BatchDiceLoss([1.0])  # nn.BCELoss()
    channels_cae = args.channelscae
    n_globals = args.globals  # type(core/penu), tO_to_tA, NHISS, sex, age
    resample_size = int(args.xyoriginal * args.xyresample)
    alpha = 1.0
    cuda = True

    # CAE model
    if args.steplearning:
        enc = Enc3DStep(size_input_xy=resample_size, size_input_z=args.zsize,
                        channels=channels_cae, n_ch_global=n_globals, alpha=alpha)
    else:
        enc = Enc3D(size_input_xy=resample_size, size_input_z=args.zsize,
                    channels=channels_cae, n_ch_global=n_globals, alpha=alpha)
    dec = Dec3D(size_input_xy=resample_size, size_input_z=args.zsize,
                channels=channels_cae, n_ch_global=n_globals, alpha=alpha)
    cae = Cae3D(enc, dec)
    if cuda:
        cae = cae.cuda()

    # Model params
    params = [p for p in cae.parameters() if p.requires_grad]
    print('# optimizing params', sum([p.nelement() * p.requires_grad for p in params]),
          '/ total: cae', sum([p.nelement() for p in cae.parameters()]))

    # Optimizer with scheduler
    optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay, betas=momentums_cae)
    if args.lrsteps:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lrsteps)
    else:
        scheduler = None

    # Data
    common_transform = [data.ToTensor()]

    ds_train, ds_valid = data.get_toy_shape_training_data([data.HemisphericFlip(), data.ElasticDeform()] + common_transform, common_transform,
                                                          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                                                          [16 ,17 ,18, 19], batchsize=args.batchsize)

    if use_validation:
        print('Size training set:', len(ds_train.sampler.indices), 'samples | Size validation set:', len(ds_valid.sampler.indices),
              'samples | Capacity batch:', args.batchsize, 'samples')
        print('# training batches:', len(ds_train), '| # validation batches:', len(ds_valid))
    else:
        print('Size training set:', len(ds_train.sampler.indices),
              'samples | Size validation set: 0 samples | Capacity batch:', args.batchsize, 'samples')
        print('# training batches:', len(ds_train), '| # validation batches:', 0)

    # Training
    learner = CaeReconstructionOnlyLearner(ds_train, ds_valid, cae, optimizer, scheduler,
                                           n_epochs=args.epochs,
                                           path_previous_base=args.inbasepath,
                                           path_outputs_base=args.outbasepath,
                                           criterion=criterion,
                                           normalization_hours_penumbra=args.normalize)
    learner.run_training()


if __name__ == '__main__':
    print(datetime.datetime.now())
    args = util.get_args_shape_training()
    train(args)
    print(datetime.datetime.now())
