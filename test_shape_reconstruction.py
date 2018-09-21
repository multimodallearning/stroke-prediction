import datetime
from tester.CaeReconstructionTester import CaeReconstructionTester
from common import data, util


def test(args):
    # Params / Config
    modalities = ['_CBV_reg1_downsampled', '_TTD_reg1_downsampled']
    labels = ['_CBVmap_subset_reg1_downsampled', '_TTDmap_subset_reg1_downsampled',
              '_FUCT_MAP_T_Samplespace_subset_reg1_downsampled']
    normalization_hours_penumbra = args.normalize
    pad = args.padding
    pad_value = 0

    for idx in range(len(args.path)):
        # Data
        transform = [data.ResamplePlaneXY(args.xyresample),
                     data.PadImages(pad[0], pad[1], pad[2], pad_value=pad_value),
                     data.ToTensor()]
        ds_test = data.get_testdata_full(modalities=modalities, labels=labels, transform=transform, indices=args.fold[idx])

        print('Test set:', ds_test.sampler.indices, 'vs.', args.fold)
        print('Size test set:', len(ds_test.sampler.indices), '| # batches:', len(ds_test))

        # Single case evaluation
        tester = CaeReconstructionTester(ds_test, args.path[idx], args.outbasepath, normalization_hours_penumbra)
        tester.run_inference()


if __name__ == '__main__':
    print(datetime.datetime.now())
    test(util.get_args_shape_testing())
    print(datetime.datetime.now())
