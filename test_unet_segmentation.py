import datetime
from tester.UnetSegmentationTester import UnetSegmentationTester
from common.model.Unet3D import Unet3D
from common import data, util


def test(args):

    # Params / Config
    modalities = ['_CBV_reg1_downsampled', '_TTD_reg1_downsampled']
    labels = ['_CBVmap_subset_reg1_downsampled', '_TTDmap_subset_reg1_downsampled']
    path_saved_model = args.unetpath
    pad = args.padding
    pad_value = 0

    # Data
    # Trained on patches, but fully convolutional approach let us apply on bigger image (thus, omit patch transform)
    transform = [data.ResamplePlaneXY(args.xyresample),
                 data.PadImages(pad[0], pad[1], pad[2], pad_value=pad_value),
                 data.ToTensor()]
    ds_test = data.get_testdata_full(modalities=modalities, labels=labels, transform=transform, indices=args.fold)

    print('Size test set:', len(ds_test.sampler.indices), '| # batches:', len(ds_test))

    # Single case evaluation
    tester = UnetSegmentationTester(ds_test, path_saved_model, args.outbasepath, None)
    tester.run_inference()


if __name__ == '__main__':
    print(datetime.datetime.now())
    args = util.get_args_unet_training()
    test(args)
    print(datetime.datetime.now())
