import torch
import datetime
from learner.CaePredictionLearner import CaePredictionLearner
from common import data, util, metrics
from common.model.Cae3D import Enc3D


def train(args):
    # Params / Config
    learning_rate = 1e-3
    momentums_cae = (0.9, 0.999)
    weight_decay = 1e-5
    criterion = metrics.BatchDiceLoss([1.0])  # nn.BCELoss()
    resample_size = int(args.xyoriginal * args.xyresample)
    n_globals = args.globals  # type(core/penu), tO_to_tA, NHISS, sex, age
    channels_enc = args.channelsenc
    alpha = 1.0
    cuda = True

    # TODO assert initbycae XOR channels_enc

    # CAE model
    path_saved_model = args.caepath
    cae = torch.load(path_saved_model)
    cae.freeze(True)
    if args.initbycae:
        enc = torch.load(path_saved_model).enc
    else:
        enc = Enc3D(size_input_xy=resample_size, size_input_z=args.zsize,
                    channels=channels_enc, n_ch_global=n_globals, alpha=alpha)

    if cuda:
        cae = cae.cuda()
        enc = enc.cuda()

    # Model params
    params = [p for p in enc.parameters() if p.requires_grad]
    print('# optimizing params', sum([p.nelement() * p.requires_grad for p in params]),
          '/ total new enc + old cae', sum([p for p in enc.parameters()] + [p.nelement() for p in cae.parameters()]))

    # Optimizer with scheduler
    optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay, betas=momentums_cae)
    if args.lrsteps:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lrsteps)
    else:
        scheduler = None

    # Data
    common_transform = [data.ResamplePlaneXY(args.xyresample),
                        data.HemisphericFlipFixedToCaseId(split_id=args.hemisflipid)]
    train_transform = common_transform + [data.ElasticDeform(apply_to_images=True), data.ToTensor()]
    valid_transform = common_transform + [data.ToTensor()]
    modalities = ['_unet_core', '_unet_penu']
    labels = ['_CBVmap_subset_reg1_downsampled', '_TTDmap_subset_reg1_downsampled',
              '_FUCT_MAP_T_Samplespace_subset_reg1_downsampled']
    ds_train, ds_valid = data.get_stroke_prediction_training_data(modalities, labels, train_transform, valid_transform,
                                                                  args.fold, args.validsetsize, batchsize=args.batchsize)
    print('Size training set:', len(ds_train.sampler.indices), 'samples | Size validation set:', len(ds_valid.sampler.indices),
          'samples | Capacity batch:', args.batchsize, 'samples')
    print('# training batches:', len(ds_train), '| # validation batches:', len(ds_valid))

    # Training
    learner = CaePredictionLearner(ds_train, ds_valid, cae, enc, optimizer, scheduler,
                                   n_epochs=args.epochs,
                                   path_previous_base=args.inbasepath,
                                   path_outputs_base=args.outbasepath,
                                   criterion=criterion,
                                   normalization_hours_penumbra=args.normalize)
    learner.run_training()


if __name__ == '__main__':
    print(datetime.datetime.now())
    args = util.get_args_shape_prediction_training()
    train(args)
    print(datetime.datetime.now())
