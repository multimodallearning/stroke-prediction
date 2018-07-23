import util
import torch
import datetime
from learner.CaeReconstructionLearner import CaeReconstructionLearner
from model.Cae3D import Cae3D, Enc3D, Dec3D
import data


def train():
    args = util.get_args_shape_training()

    # Params
    batchsize = 6  # 17 training, 6 validation
    learning_rate = 1e-3
    momentums_cae = (0.99, 0.999)
    criterion = util.BatchDiceLoss([1.0])
    path_saved_model = args.caepath
    channels_cae = args.channelscae
    n_globals = args.globals  # type(core/penu), tO_to_tA, NHISS, sex, age
    resample_size = int(args.xyoriginal * args.xyresample)
    pad = args.padding
    leakage = 0.2
    cuda = True

    # CAE model
    enc = Enc3D(size_input_xy=resample_size, size_input_z=args.zsize,
                channels=channels_cae, n_ch_global=n_globals, leakage=leakage)
    dec = Dec3D(size_input_xy=resample_size, size_input_z=args.zsize,
                channels=channels_cae, n_ch_global=n_globals, leakage=leakage)
    cae = Cae3D(enc, dec)
    if cuda:
        cae = cae.cuda()

    # Model params
    params = [p for p in cae.parameters() if p.requires_grad]
    print('# optimizing params', sum([p.nelement() * p.requires_grad for p in params]),
          '/ total: cae', sum([p.nelement() for p in cae.parameters()]))

    # Optimizer
    optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=1e-5, betas=momentums_cae)

    # Data
    train_transform = [data.ResamplePlaneXY(args.xyresample),
                       data.HemisphericFlipFixedToCaseId(split_id=args.hemisflipid),
                       data.PadImages(pad[0], pad[1], pad[2], pad_value=0),
                       data.ElasticDeform(),
                       data.ToTensor()]
    valid_transform = [data.ResamplePlaneXY(args.xyresample),
                       data.HemisphericFlipFixedToCaseId(split_id=args.hemisflipid),
                       data.PadImages(pad[0], pad[1], pad[2], pad_value=0),
                       data.ToTensor()]
    ds_train, ds_valid = util.get_data_shape_labels(train_transform, valid_transform, args.fold, args.validsetsize,
                                                    batchsize=batchsize)
    print('Size training set:', len(ds_train.sampler.indices), '| Size validation set:', len(ds_valid.sampler.indices))
    print('# training batches:', len(ds_train), '| # validation batches:', len(ds_valid))

    # Training
    learner = CaeReconstructionLearner(ds_train, ds_valid, cae, path_saved_model, optimizer, args.epochs,
                                       args.outbasepath, criterion, cuda=cuda)
    learner.run_training()


if __name__ == '__main__':
    print(datetime.datetime.now())
    train()
    print(datetime.datetime.now())
