import argparse
import medpy.metric.binary as mpm
import numpy
import data


# =========================== BATCH LOSS ==========================


from torch.nn.modules.loss import _Loss as LossModule


class BatchDiceLoss(LossModule):
    def __init__(self, class_weights, smooth=0.0000001, dim=1):
        super(BatchDiceLoss, self).__init__()
        self._smooth = smooth
        self._dim = dim
        self._class_weights = class_weights
        print("DICE Loss weights each class' output by", class_weights)

    def forward(self, outputs, targets):
        assert targets.shape[self._dim] == len(self._class_weights), \
            'Ground truth number of classes does not match with class weight vector'
        loss = 0.0
        for label in range(len(self._class_weights)):
            iflat = outputs.narrow(self._dim, label, 1).contiguous().view(-1)
            tflat = targets.narrow(self._dim, label, 1).contiguous().view(-1)
            assert iflat.size() == tflat.size()
            intersection = (iflat * tflat).sum()
            loss += self._class_weights[label] * ((2. * intersection + self._smooth) /
                                                  ((iflat * iflat).sum() + (tflat * tflat).sum() + self._smooth))
        return 1.0 - loss


# ======================== HELPER FUNCTIONS =======================


def compute_binary_measure_numpy(result, target, threshold=0.5):
    result_binary = (result[0, 0, :, :, :] > threshold).astype(numpy.uint8)
    target_binary = (target[0, 0, :, :, :] > threshold).astype(numpy.uint8)

    dc = mpm.dc(result_binary, target_binary)
    hd = numpy.Inf
    assd = numpy.Inf
    if result_binary.any() and target_binary.any():
        hd = mpm.hd(result_binary, target_binary)
        assd = mpm.assd(result_binary, target_binary)

    return dc, hd, assd


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


def get_data_shape_labels(train_transform, valid_transform, fold_indices, ratio, seed=4, batchsize=2):
    modalities = ['_CBV_reg1_downsampled', '_TTD_reg1_downsampled']
    labels = ['_CBVmap_subset_reg1_downsampled', '_TTDmap_subset_reg1_downsampled',
              '_FUCT_MAP_T_Samplespace_subset_reg1_downsampled']

    ds_train, ds_valid = data.split_data_loader3D(modalities, labels, fold_indices, batchsize, random_seed=seed,
                                                  valid_size=ratio, train_transform=train_transform,
                                                  valid_transform=valid_transform, num_workers=0)

    return ds_train, ds_valid


# =================================== PARSER ===========================


class ExpParser(argparse.ArgumentParser):
    def __init__(self, type_of_data=None):
        super().__init__()
        self.add_argument('--fold', type=int, nargs='+', help='Fold number',
                          default=list(range(29)))  # Internal indices, NOT case numbers on disk)
        self.add_argument('--hemisflipid', type=float, help='Case id or greater, at which hemispheric flip is applied',
                          default=15)
        self.add_argument('--threshold', type=float,
                          help='Threshold to binarize segmentation in order to compute distances', default=0.5)
        self.add_argument('--validsetsize', type=float, help='Fraction of validation set size', default=0.5)
        self.add_argument('--seed', type=int, help='Seed for any randomization', default=4)
        self.add_argument('--xyoriginal', type=int, help='Original size of slices', default=256)
        self.add_argument('--xyresample', type=int, help='Factor for resampling slices', default=0.5)
        self.add_argument('--zsize', type=int, help='Number of z slices', default=28)
        self.add_argument('--padding', type=int, nargs='+', help='Padding of patches', default=[20, 20, 20])
        self._type_of_data = type_of_data

    def parse_args(self, args=None, namespace=None):
        args = super().parse_args(args, namespace)
        print(args)
        return args


class CAEParser(ExpParser):
    def __init__(self, type_of_data=None):
        super().__init__(type_of_data)
        self.add_argument('caepath', type=str, help='Path to model of Shape CAE',
                          default='/share/data_zoe1/Linda_Segmentations/tmp/cae_shape.model')
        self.add_argument('--epochs', type=int, help='Number of epochs', default=300)
        self.add_argument('--globals', type=int, help='Number of global variables', default=5)
        self.add_argument('--batchsize', type=int, help='Batchsize for training', default=1)
        self.add_argument('--channelscae', type=int, nargs='+', help='CAE channels',
                          default=[1, 24, 32, 48, 64, 500, 200, 1])
        self.add_argument('--channelsenc', type=int, nargs='+', help='2nd enc channels',
                          default=[1, 24, 32, 48, 64, 500, 200, 1])
        self.add_argument('--normalize', type=int, help='Normalization value corresponding to penumbra (hours)',
                          default=10)
        self.add_argument('--outbasepath', type=str, help='Path and filename base for outputs',
                          default='/share/data_zoe1/lucas/Linda_Segmentations/tmp/shape')


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
        self.add_argument('--testcaseid', type=int, help='Testingcaseid', default=0)
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
    args = parser.parse_args()
    return args


def get_args_unet_training():
    parser = UnetParser()
    args = parser.parse_args()
    return args