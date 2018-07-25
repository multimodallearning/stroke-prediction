import datetime
from tester.UnetSegmentationTester import UnetSegmentationTester
from common.model.Unet3D import Unet3D
from common import data, util


def test():
    args = util.get_args_unet_training()

    # Params
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

    # Data
    # Trained on patches, but fully convolutional approach let us apply on bigger image (thus, omit patch transform)
    transform = [data.ResamplePlaneXY(args.xyresample),
                 data.PadImages(pad[0], pad[1], pad[2], pad_value=0),
                 data.ToTensor()]

    ds_test = data.get_testdata(modalities=['_CBV_reg1_downsampled', '_TTD_reg1_downsampled'],
                                labels=['_CBVmap_subset_reg1_downsampled', '_TTDmap_subset_reg1_downsampled',
                                             '_FUCT_MAP_T_Samplespace_subset_reg1_downsampled'],
                                transform=transform,
                                indices=args.fold)

    print('Size test set:', len(ds_test.sampler.indices), '| # batches:', len(ds_test))

    # Single case evaluation
    tester = UnetSegmentationTester(ds_test, unet, path_saved_model, args.outbasepath)
    tester.run_inference()


if __name__ == '__main__':
    print(datetime.datetime.now())
    test()
    print(datetime.datetime.now())
