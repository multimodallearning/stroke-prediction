import argparse
from common import data


# ======================= DETERMINISTIC DATA ===========================


def get_vis_samples(train_loader, valid_loader):
    n_vis_samples = 6
    visual_samples = []
    visual_times = []
    for i in train_loader.sampler.indices:
        sample = train_loader.dataset[i]
        sample[data.KEY_IMAGES] = sample[data.KEY_IMAGES].unsqueeze(0)
        sample[data.KEY_LABELS] = sample[data.KEY_LABELS].unsqueeze(0)
        sample[data.KEY_GLOBAL] = sample[data.KEY_GLOBAL].unsqueeze(0)
        visual_samples.append(sample)
        tA_to_tR_tmp = sample[data.KEY_GLOBAL][0, 1, :, :, :]
        visual_times.append(float(tA_to_tR_tmp))
        if len(visual_samples) > n_vis_samples / 2 - 1:
            break;
    if valid_loader is not None:
        for i in valid_loader.sampler.indices:
            sample = valid_loader.dataset[i]
            sample[data.KEY_IMAGES] = sample[data.KEY_IMAGES].unsqueeze(0)
            sample[data.KEY_LABELS] = sample[data.KEY_LABELS].unsqueeze(0)
            sample[data.KEY_GLOBAL] = sample[data.KEY_GLOBAL].unsqueeze(0)
            visual_samples.append(sample)
            tA_to_tR_tmp = sample[data.KEY_GLOBAL][0, 1, :, :, :]
            visual_times.append(float(tA_to_tR_tmp))
            if len(visual_samples) > n_vis_samples - 1:
                break;

    return visual_samples, visual_times


# =================================== PARSER ===========================


class ExpParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument('--fold', type=int, nargs='+', help='Fold case indices',
                          default=list(range(29)))  # Internal indices, NOT case numbers on disk)
        self.add_argument('--hemisflipid', type=float, help='Case id or greater, at which hemispheric flip is applied',
                          default=15)
        self.add_argument('--validsetsize', type=float, help='Fraction of validation set size', default=0.5)
        self.add_argument('--seed', type=int, help='Seed for any randomization', default=4)
        self.add_argument('--xyoriginal', type=int, help='Original size of slices', default=256)
        self.add_argument('--xyresample', type=int, help='Factor for resampling slices', default=0.5)
        self.add_argument('--zsize', type=int, help='Number of z slices', default=28)
        self.add_argument('--padding', type=int, nargs='+', help='Padding of patches', default=[20, 20, 20])
        self.add_argument('--lrsteps', type=int, nargs='+', help='MultiStepLR epochs', default=[])

    def parse_args(self, args=None, namespace=None):
        args = super().parse_args(args, namespace)
        print(args)
        return args


class CAEParser(ExpParser):
    def __init__(self):
        super().__init__()
        self.add_argument('--epochs', type=int, help='Number of epochs', default=300)
        self.add_argument('--batchsize', type=int, help='Batch size', default=4)
        self.add_argument('--globals', type=int, help='Number of global variables', default=5)
        self.add_argument('--normalize', type=int, help='Normalization corresponding to penumbra (hours)', default=10)
        self.add_argument('--inbasepath', type=str, help='Path and filename base for loading', default=None)
        self.add_argument('--outbasepath', type=str, help='Path and filename base for saving', default='/tmp/tmp_out')
        self.add_argument('--steplearning', action='store_true', help='Also learn interpolation step from clinical data', default=False)


class UnetParser(ExpParser):
    def __init__(self):
        super().__init__()
        self.add_argument('unetpath', type=str, help='Path to model of Unet',
                          default='/share/data_zoe1/Linda_Segmentations/tmp/unet.model')
        self.add_argument('--channels', type=int, nargs='+', help='Unet channels',
                          default=[2, 16, 32, 64, 32, 16, 32, 2])
        self.add_argument('--epochs', type=int, help='Number of epochs', default=200)
        self.add_argument('--outbasepath', type=str, help='Path and filename base for outputs',
                          default='/share/data_zoe1/lucas/Linda_Segmentations/tmp/unet')


class SDMParser(ExpParser):
    def __init__(self):
        super().__init__()
        self.add_argument('unet', type=str, help='Path to model of Segmentation Unet',
                          default='/share/data_zoe1/lucas/unet1dcm.model')
        self.add_argument('--channels', type=int, nargs='+', help='Unet channels',
                          default=[2, 16, 32, 64, 32, 16, 32, 2])
        self.add_argument('--downsample', type=int, help='Downsampling to CAE latent representation size', default=1)
        self.add_argument('--groundtruth', type=int, help='Use groundtruth instead of UNet segmentations', default=1)
        self.add_argument('--visualinspection', type=int, help='Inspect visually before it is saved', default=0)
        self.add_argument('--outbasepath', type=str, help='Path and filename base for outputs',
                          default='/share/data_zoe1/lucas/Linda_Segmentations/tmp/sdm')


def get_args_sdm():
    parser = SDMParser()
    args = parser.parse_args()
    return args


def get_args_shape_training():
    parser = CAEParser()
    parser.add_argument('--channelscae', type=int, nargs='+', help='CAE channels', default=[1, 16, 24, 32, 100, 200, 1])
    args = parser.parse_args()
    return args

def get_args_step_training():
    parser = CAEParser()
    parser.add_argument('caepath', type=str, help='Path to previously trained cae phase1 model')
    parser.add_argument('--channelscae', type=int, nargs='+', help='CAE channels', default=[1, 16, 24, 32, 100, 200, 1])
    args = parser.parse_args()
    return args


def get_args_shape_prediction_training():
    parser = CAEParser()
    parser.add_argument('caepath', type=str, help='Path to previously trained cae phase1 model')
    parser.add_argument('--channelsenc', type=int, nargs='+', help='CAE channels', default=[1, 16, 24, 32, 100, 200, 1])
    parser.add_argument('--initbycae', action='store_true', help='Init enc weights by cae\'s enc', default=False)
    args = parser.parse_args()
    return args


def get_args_shape_testing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', action='append', type=str, help='Path to model of Shape CAE')
    parser.add_argument('--fold', action='append', type=int, nargs='+', help='Fold case indices')  # Internal indices, NOT case numbers on disk)
    parser.add_argument('--normalize', type=int, help='Normalization value corresponding to penumbra (hours)',
                        default=10)
    parser.add_argument('--outbasepath', type=str, help='Path and filename base for outputs',
                        default='/share/data_zoe1/lucas/Linda_Segmentations/tmp/shape')
    parser.add_argument('--xyresample', type=int, help='Factor for resampling slices', default=0.5)
    parser.add_argument('--padding', type=int, nargs='+', help='Padding of patches', default=[20, 20, 20])
    args = parser.parse_args()
    return args


def get_args_unet_training():
    parser = UnetParser()
    args = parser.parse_args()
    return args