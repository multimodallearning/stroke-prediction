import datetime
from tester.CaeReconstructionTester import CaeReconstructionTester
from common.model.Cae3D import Cae3D, Dec3D, Enc3D
from common import data, util


def test():
    args = util.get_args_shape_training()

    # Params / Config
    modalities = ['_CBV_reg1_downsampled', '_TTD_reg1_downsampled']
    labels = ['_CBVmap_subset_reg1_downsampled', '_TTDmap_subset_reg1_downsampled',
              '_FUCT_MAP_T_Samplespace_subset_reg1_downsampled']
    path_saved_model = args.caepath
    normalization_hours_penumbra = args.normalize
    pad = args.padding
    pad_value = 0

    # Data
    transform = [data.ResamplePlaneXY(args.xyresample),
                 data.PadImages(pad[0], pad[1], pad[2], pad_value=pad_value),
                 data.ToTensor()]
    ds_test = data.get_testdata(modalities=modalities, labels=labels, transform=transform, indices=args.fold)

    print('Size test set:', len(ds_test.sampler.indices), '| # batches:', len(ds_test))

    # Single case evaluation
    tester = CaeReconstructionTester(ds_test, path_saved_model, args.outbasepath, normalization_hours_penumbra)
    tester.run_inference()


if __name__ == '__main__':
    print(datetime.datetime.now())
    test()
    print(datetime.datetime.now())
