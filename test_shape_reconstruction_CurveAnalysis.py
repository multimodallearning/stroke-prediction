import datetime
from tester.CaeReconstructionTesterCurve import CaeReconstructionTesterCurve
from common import data, util


def test():
    args = util.get_args_shape_testing()

    assert len(args.fold) == len(args.path), 'You must provide as many --fold arguments as caepath model arguments\
                                                in the exact same order!'

    # Params / Config
    modalities = ['_CBV_reg1_downsampled', '_TTD_reg1_downsampled']
    labels = ['_CBVmap_subset_reg1_downsampled', '_TTDmap_subset_reg1_downsampled',
              '_FUCT_MAP_T_Samplespace_subset_reg1_downsampled']
    normalization_hours_penumbra = args.normalize
    steps = range(6)  # fixed steps for tAdmission-->tReca: 0-5 hrs
    pad = args.padding
    pad_value = 0

    # Data
    transform = [data.ResamplePlaneXY(args.xyresample),
                 data.PadImages(pad[0], pad[1], pad[2], pad_value=pad_value),
                 data.ToTensor()]

    # Fold-wise evaluation according to fold indices and fold model for all folds and model path provided as arguments:
    for i, path in enumerate(args.path):
        print('Model ' + path + ' of fold ' + str(i+1) + '/' + str(len(args.fold)) + ' with indices: ' + str(args.fold[i]))
        ds_test = data.get_testdata_full(modalities=modalities, labels=labels, transform=transform, indices=args.fold[i])
        print('Size test set:', len(ds_test.sampler.indices), '| # batches:', len(ds_test))
        # Single case evaluation for all cases in fold
        tester = CaeReconstructionTesterCurve(ds_test, path, args.outbasepath, normalization_hours_penumbra, steps)
        tester.run_inference()


if __name__ == '__main__':
    print(datetime.datetime.now())
    test()
    print(datetime.datetime.now())
