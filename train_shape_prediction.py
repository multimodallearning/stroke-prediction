import torch
import datetime
from learner.CaePredictionLearner import CaePredictionLearner
from common import data, util, metrics


def train(args):
    # Params / Config
    learning_rate = 1e-3
    momentums_cae = (0.9, 0.999)
    weight_decay = 1e-5
    criterion = metrics.BatchDiceLoss([1.0])  # nn.BCELoss()
    pad = args.padding
    pad_value = 0
    cuda = True

    # CAE model
    path_saved_model = '/data_zoe1/lucas/Linda_Segmentations/tmp/tmp_shape1_phase1.model'
    cae = torch.load(path_saved_model)
    enc = cae.enc
    cae = torch.load(path_saved_model)
    cae.freeze(True)
    if cuda:
        cae = cae.cuda()
        enc = enc.cuda()

    # Model params
    params = [p for p in enc.parameters() if p.requires_grad]
    print('# optimizing params', sum([p.nelement() * p.requires_grad for p in params]),
          '/ total new enc', sum([p.nelement() for p in cae.parameters()]))

    # Optimizer with scheduler
    optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay, betas=momentums_cae)
    if args.lrsteps:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lrsteps)
    else:
        scheduler = None

    # Data
    common_transform = [data.ResamplePlaneXY(args.xyresample),
                        data.HemisphericFlipFixedToCaseId(split_id=args.hemisflipid),
                        data.PadImages(pad[0], pad[1], pad[2], pad_value=pad_value)]
    train_transform = common_transform + [data.ElasticDeform(), data.ToTensor()]
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
                                   criterion=criterion)
    learner.run_training()


if __name__ == '__main__':
    print(datetime.datetime.now())
    args = util.get_args_shape_training()
    train(args)
    print(datetime.datetime.now())
