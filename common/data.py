import os
import csv
import nibabel as nib
import random
import datetime

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler

import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage.interpolation import map_coordinates, zoom
from scipy.ndimage.filters import gaussian_filter


from skimage.draw import ellipsoid, polygon


KEY_CASE_ID = 'case_id'
KEY_CLINICAL_IDX = 'clinical_idx'
KEY_IMAGES = 'images'
KEY_LABELS = 'labels'
KEY_GLOBAL = 'clinical'

DIM_HORIZONTAL_NUMPY_3D = 0
DIM_DEPTH_NUMPY_3D = 2
DIM_CHANNEL_NUMPY_3D = 3
DIM_CHANNEL_TORCH3D_5 = 1


def sdm_interpolate_numpy(seg0, seg1, t, threshold=0.5, dilate=3):
    seg1_bin = seg1 > threshold
    seg1_dist = ndi.distance_transform_edt(seg1_bin)
    seg1_dist -= ndi.distance_transform_edt(seg1 < threshold)

    seg0_bin = seg0 > threshold
    seg0_dist = ndi.distance_transform_edt(1 - seg0_bin)
    seg0_dist -= ndi.distance_transform_edt(seg0 > threshold)

    dist_t = seg1_dist * t - seg0_dist * (1 - t)

    return seg0_dist, dist_t, seg1_dist


def time_func(t, func='lin'):
    assert 0 <= t <= 1

    if func == 'lin':
        return t

    if func == 'slow':
        return pow(t, 2)
    if func == 'fast':
        return pow(t, 0.5)

    if func == 'log':  # logistic: slow-fast-slow
        return 1 / (1 + 1000 * pow(0.000001, t))

    return t


def overlay(seg0, seg1, dist_t):
    img = np.zeros(seg0.shape)
    img += seg1
    img += (dist_t > 0.5)
    img += seg0
    return img


class ElasticDeform2(object):
    """Elastic deformation of images as described in [Simard2003]
       Simard, Steinkraus and Platt, "Best Practices for Convolutional
       Neural Networks applied to Visual Document Analysis", in Proc.
       of the International Conference on Document Analysis and
       Recognition, 2003.
    """

    def __init__(self, alpha=100, sigma=4):
        self._alpha = alpha
        self._sigma = sigma

    def elastic_transform(self, image, alpha=100, sigma=4, random_state=None):
        if random_state is None:
            new_seed = datetime.datetime.now().second + datetime.datetime.now().microsecond
            random_state = np.random.RandomState(new_seed)

        shape = image.shape
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dz = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant",
                             cval=0) * alpha * 0.22  # 28/128  TODO: correct according to voxel spacing

        x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z + dz, (-1, 1))

        return map_coordinates(image, indices, order=1).reshape(shape), random_state

    def run(self, start, interpolation, end, one4all = False, random_state = None):
        interpolation, rs = self.elastic_transform(interpolation, self._alpha, self._sigma, random_state=random_state)
        if one4all:
            if start is not None:
                start, _ = self.elastic_transform(start, self._alpha, self._sigma, random_state=rs)
            if end is not None:
                end, _ = self.elastic_transform(end, self._alpha, self._sigma, random_state=rs)
        else:
            if start is not None:
                start, _ = self.elastic_transform(start, self._alpha, self._sigma, random_state=None)
            if end is not None:
                end, _ = self.elastic_transform(end, self._alpha, self._sigma, random_state=None)
        return start, interpolation, end, rs


class ToyDataset3D(Dataset):
    def __init__(self, transform=[], length=20):
        self._deform = ElasticDeform2()
        self._transform = transform

        self._item = []
        self._core = []
        self._intp = []
        self._penu = []
        for i in range(length):
            timep = random.random()
            self._item.append({KEY_CASE_ID: i, KEY_CLINICAL_IDX: timep*24})

            left_right = (random.random() < 0.5)

            seg0 = np.zeros((128, 128, 28))
            seg1 = np.zeros((128, 128, 28))

            w = random.randint(10, 30)
            h = random.randint(10, 60)
            d = random.randint(3, 11)

            off_x = (64 - w * 2) // 2 + left_right * 64
            off_y = (128 - h * 2) // 2
            off_z = (28 - d * 2) // 2
            seg1[off_x-1:off_x+2*w+2, off_y-1:off_y+2*h+2, off_z-1:off_z+2*d+2] = ellipsoid(w, h, d).astype(np.float)

            com = np.round(ndi.center_of_mass(seg1)).astype(np.int)

            w_new = random.randint(4, w // 2)
            h_new = random.randint(4, h // 2)
            d_new = random.randint(1, d // 2)

            off_x += (w - w_new) // 2
            off_y += (h - h_new) // 2
            off_z += (d - d_new) // 2
            seg0[off_x-1:off_x+2*w_new+2, off_y-1:off_y+2*h_new+2, off_z-1:off_z+2*d_new+2] = ellipsoid(w_new, h_new, d_new).astype(np.float)

            intersection = seg0 * seg1
            if np.sum(intersection) < 5:
                intersection[com[0]-2:com[0]+2, com[1]-2:com[1]+2, com[2]-2:com[2]+2] = 1

            _, dist_t, _ = sdm_interpolate_numpy(seg0, seg1, time_func(timep, 'fast'))
            seg0_d, dist_t, seg1_d = self._deform.run(seg0, dist_t, seg1, one4all=True)

            seg0_d[seg0_d < 0.5] = 0
            dist_t[dist_t < 0.5] = 0
            seg1_d[seg1_d < 0.5] = 0
            seg0_d[seg0_d > 0] = 1
            dist_t[dist_t > 0] = 1
            seg1_d[seg1_d > 0] = 1
            seg0_d = seg0_d * seg1_d
            dist_t = dist_t * seg1_d

            self._core.append(seg0_d[:, :, :, np.newaxis].astype(np.float))
            self._intp.append(dist_t[:, :, :, np.newaxis].astype(np.float))
            self._penu.append(seg1_d[:, :, :, np.newaxis].astype(np.float))

    def __len__(self):
        return len(self._item)

    def __getitem__(self, item):
        item_id = self._item[item]
        case_id = item_id[KEY_CASE_ID]

        globalss = np.zeros((1, 1, 1, 2))
        globalss[:, :, :, 1] = item_id[KEY_CLINICAL_IDX]
        globalss[:, :, :, 0] = 0

        result = {KEY_CASE_ID: case_id, KEY_IMAGES: [], KEY_LABELS: [], KEY_GLOBAL: globalss}

        result[KEY_LABELS].append(self._core[item])
        result[KEY_LABELS].append(self._penu[item])
        result[KEY_LABELS].append(self._intp[item])
        if result[KEY_LABELS]:
            result[KEY_LABELS] = np.concatenate(result[KEY_LABELS], axis=DIM_CHANNEL_NUMPY_3D)

        result[KEY_IMAGES] = np.zeros((128, 128, 28, 2))

        if self._transform:
            result = self._transform(result)

        return result


class ToyDataset3DSequence(Dataset):
    def __init__(self, transform=[], dataset_length=20, normalize=10, growth='lin', zsize=28):
        self._transform = transform

        self._labels = []
        self._item = []
        for i in range(dataset_length):
            labels = np.zeros(shape=(128, 128, 28, normalize))

            t_imgs = int(random.random()*(normalize-2))
            t_reca = int(random.random()*(normalize-2-t_imgs) + t_imgs + 1)
            self._item.append({KEY_CASE_ID: i, 'ti': t_imgs, 'tr': t_reca})

            left_right = (random.random() < 0.5)

            seg0 = np.zeros((128, 128, 28))
            seg1 = np.zeros((128, 128, 28))

            w = random.randint(10, 30)
            h = random.randint(10, 60)
            d = random.randint(3, 11)

            off_x = (64 - w * 2) // 2 + left_right * 64
            off_y = (128 - h * 2) // 2
            off_z = (28 - d * 2) // 2
            seg1[off_x-1:off_x+2*w+2, off_y-1:off_y+2*h+2, off_z-1:off_z+2*d+2] = ellipsoid(w, h, d).astype(np.float)

            com = np.round(ndi.center_of_mass(seg1)).astype(np.int)

            seg0[com[0]-2:com[0]+2, com[1]-2:com[1]+2, com[2]-2:com[2]+2] = 1

            labels[:, :, :, 0] = seg0
            for j in range(1, normalize - 1):
                _, dist_int, _ = sdm_interpolate_numpy(seg0, seg1, time_func(j/normalize, growth))
                intp = (dist_int > 0).astype(seg1.dtype)
                labels[:, :, :, j] = np.maximum(intp * seg1, labels[:, :, :, j-1])  # TODO: correct? monotone property
            labels[:, :, :, normalize - 1] = seg1

            if zsize == 1:
                labels = labels[:, :, com[2], np.newaxis, :]

            self._labels.append(labels)

            self._zsize = zsize

    def __len__(self):
        return len(self._item)

    def __getitem__(self, item):
        item_id = self._item[item]
        case_id = item_id[KEY_CASE_ID]

        globalss = np.zeros((1, 1, 1, 2))
        globalss[:, :, :, 0] = item_id['ti']  # t_imaging
        globalss[:, :, :, 1] = item_id['tr']  # t_recanalization

        result = {KEY_CASE_ID: case_id, KEY_IMAGES: [], KEY_LABELS: [], KEY_GLOBAL: globalss}

        result[KEY_LABELS] = self._labels[item].copy()

        result[KEY_IMAGES] = np.zeros((128, 128, self._zsize, 2))

        if self._transform:
            result = self._transform(result)

        return result



class StrokeLindaDataset3D(Dataset):
    """Ischemic stroke dataset with CBV, TTD, clinical data, and CBVmap, TTDmap, FUmap, and interpolations."""
    PATH_ROOT = '/share/data_zoe1/lucas/Linda_Segmentations'
    PATH_CSV = '/share/data_zoe1/lucas/Linda_Segmentations/clinical_cleaned.csv'
    FN_PREFIX = 'train'
    FN_PATTERN = '{1}/{0}{1}{2}.nii.gz'
    ROW_OFFSET = 1
    COL_OFFSET = 1

    def __init__(self, root_dir=PATH_ROOT, modalities=[], labels=[], clinical=PATH_CSV, transform=None,
                 single_case_id=None):
        self._root_dir = root_dir
        self._clinical = self._load_clinical_data_from_csv(clinical, row_offset=self.ROW_OFFSET, col_offset=0)
        self._transform = transform
        self._modalities = modalities
        self._labels = labels

        self._item_index_map = []
        for index in range(len(self._clinical)):
            case_id = int(self._clinical[index][0])
            if single_case_id is not None and single_case_id != case_id:
                continue
            self._item_index_map.append({KEY_CASE_ID: case_id, KEY_CLINICAL_IDX: index})

    def _load_clinical_data_from_csv(self, filename, col_offset=0, row_offset=0):
        result = []
        with open(filename, 'r') as f:
            rows = csv.reader(f, delimiter=',')
            for row in rows:
                if row_offset == 0:
                    result.append(row[col_offset:])
                else:
                    row_offset -= 1
        return result

    def _load_image_data_from_nifti(self, case_id, suffix):
        img_name = self.FN_PATTERN.format(self.FN_PREFIX, str(case_id), suffix)
        filename = os.path.join(self._root_dir, img_name)
        img_data = nib.load(filename).get_data()
        return img_data[:, :, :, np.newaxis]

    def __len__(self):
        return len(self._item_index_map)

    def __getitem__(self, item):
        item_id = self._item_index_map[item]
        case_id = item_id[KEY_CASE_ID]
        clinical_data = self._clinical[item_id[KEY_CLINICAL_IDX]][1:]

        result = {KEY_CASE_ID: case_id, KEY_IMAGES: [], KEY_LABELS: [], KEY_GLOBAL: []}

        for value in clinical_data:
            result[KEY_GLOBAL].append(float(value))
        if result[KEY_GLOBAL]:
            result[KEY_GLOBAL] = np.array(result[KEY_GLOBAL]).reshape((1, 1, 1, len(clinical_data)))

        for label in self._labels:
            result[KEY_LABELS].append(self._load_image_data_from_nifti(case_id, label))
        if result[KEY_LABELS]:
            result[KEY_LABELS] = np.concatenate(result[KEY_LABELS], axis=DIM_CHANNEL_NUMPY_3D)

        for modality in self._modalities:
            result[KEY_IMAGES].append(self._load_image_data_from_nifti(case_id, modality))
        if result[KEY_IMAGES]:
            result[KEY_IMAGES] = np.concatenate(result[KEY_IMAGES], axis=DIM_CHANNEL_NUMPY_3D)

        if self._transform:
            result = self._transform(result)

        return result


def emptyCopyFromSample(sample):
    result = {KEY_CASE_ID: int(sample[KEY_CASE_ID]), KEY_IMAGES: [], KEY_LABELS: [], KEY_GLOBAL: []}
    return result


def set_np_seed(workerid):
    torch_seed = torch.initial_seed()
    numpy_seed = torch_seed % np.iinfo(np.int32).max
    np.random.seed(numpy_seed)


def split_data_loader3D(modalities, labels, indices, batch_size, random_seed=None, valid_size=0.5, shuffle=True,
                        num_workers=4, pin_memory=False, train_transform=[], valid_transform=[]):
    assert ((valid_size >= 0) and (valid_size <= 1)), "[!] valid_size should be in the range [0, 1]."
    assert train_transform, "You must provide at least a numpy-to-torch transformation."
    assert valid_transform, "You must provide at least a numpy-to-torch transformation."

    # load the dataset
    dataset_train = StrokeLindaDataset3D(modalities=modalities, labels=labels,
                                         transform=transforms.Compose(train_transform))
    dataset_valid = StrokeLindaDataset3D(modalities=modalities, labels=labels,
                                         transform=transforms.Compose(valid_transform))

    items = list(set(range(len(dataset_train))).intersection(set(indices)))
    num_train = len(items)
    split = int(np.floor(valid_size * num_train))

    if shuffle == True:
        random_state = np.random.RandomState(random_seed)
        random_state.shuffle(items)

    train_idx, valid_idx = items[split:], items[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SequentialSampler(valid_idx)

    train_loader = DataLoader(dataset_train,
                    batch_size=batch_size, sampler=train_sampler,
                    num_workers=num_workers, pin_memory=pin_memory,
                    worker_init_fn=set_np_seed, drop_last=True)

    valid_loader = DataLoader(dataset_valid,
                    batch_size=batch_size, sampler=valid_sampler,
                    num_workers=num_workers, pin_memory=pin_memory, drop_last=True)

    return (train_loader, valid_loader)


def single_data_loader3D_full(modalities, labels, indices, batch_size, random_seed=None, shuffle=True,
                              num_workers=4, pin_memory=False, train_transform=[]):
    assert train_transform, "You must provide at least a numpy-to-torch transformation."

    # load the dataset
    dataset_train = StrokeLindaDataset3D(modalities=modalities, labels=labels,
                                         transform=transforms.Compose(train_transform),
                                         clinical='/share/data_zoe1/lucas/Linda_Segmentations/clinical_cleaned_full.csv')

    items = list(set(range(len(dataset_train))).intersection(set(indices)))
    print('Indices used:', items)

    if shuffle == True:
        random_state = np.random.RandomState(random_seed)
        random_state.shuffle(items)

    train_sampler = SubsetRandomSampler(items)

    train_loader = DataLoader(dataset_train, batch_size=batch_size, sampler=train_sampler,
                              num_workers=num_workers, pin_memory=pin_memory, worker_init_fn=set_np_seed)

    return train_loader


def get_toy_shape_training_data(train_transform, valid_transform, t_indices, v_indices, seed=4, batchsize=2):
    assert train_transform, "You must provide at least a numpy-to-torch transformation."

    dataset_train = ToyDataset3D(transform=transforms.Compose(train_transform))
    items = list(set(range(len(dataset_train))).intersection(set(t_indices)))
    print('Indices used:', items)
    random_state = np.random.RandomState(seed)
    random_state.shuffle(items)
    train_sampler = SubsetRandomSampler(items)
    train_loader = DataLoader(dataset_train, batch_size=batchsize, sampler=train_sampler)

    dataset_valid = ToyDataset3D(transform=transforms.Compose(valid_transform))
    items = list(set(range(len(dataset_valid))).intersection(set(v_indices)))
    print('Indices used:', items)
    random_state = np.random.RandomState(seed)
    random_state.shuffle(items)
    valid_sampler = SequentialSampler(dataset_valid)
    valid_loader = DataLoader(dataset_valid, batch_size=batchsize, sampler=valid_sampler)

    return train_loader, valid_loader


def get_toy_seq_shape_training_data(train_transform, valid_transform, t_indices, v_indices, batchsize=2, normalize=10,
                                    growth='lin', zsize=28):
    assert train_transform, "You must provide at least a numpy-to-torch transformation."

    dataset_length = len(t_indices) + len(v_indices)

    dataset = ToyDataset3DSequence(transform=transforms.Compose(train_transform), dataset_length=dataset_length,
                                   normalize=normalize, growth=growth, zsize=zsize)
    items = list(set(range(len(dataset))).intersection(set(t_indices)))
    print('Indices used:', items)
    train_sampler = SubsetRandomSampler(items)
    train_loader = DataLoader(dataset, batch_size=batchsize, sampler=train_sampler)

    if v_indices:
        dataset = ToyDataset3DSequence(transform=transforms.Compose(valid_transform), dataset_length=dataset_length,
                                       normalize=normalize, growth=growth, zsize=zsize)
        items = list(set(range(len(dataset))).intersection(set(v_indices)))
        print('Indices used:', items)
        valid_sampler = SequentialSampler(dataset)
        valid_loader = DataLoader(dataset, batch_size=batchsize, sampler=valid_sampler)
    else:
        valid_loader = None

    return train_loader, valid_loader


def get_stroke_shape_training_data(modalities, labels, train_transform, valid_transform, fold_indices, ratio, seed=4,
                                   batchsize=2, split=True):
    if split:
        return split_data_loader3D(modalities, labels, fold_indices, batchsize, random_seed=seed,
                                   valid_size=ratio, train_transform=train_transform,
                                   valid_transform=valid_transform, num_workers=0)

    tmp_valid = single_data_loader3D_full(modalities, labels, [0, 4, 8, 19, 21, 31], 3, random_seed=seed,
                                     train_transform=[ResamplePlaneXY(.5), ToTensor()], num_workers=0)  # TODO for debug purposes

    return single_data_loader3D_full(modalities, labels, fold_indices, batchsize, random_seed=seed,
                                     train_transform=train_transform, num_workers=0), tmp_valid


def get_stroke_prediction_training_data(modalities, labels, train_transform, valid_transform, fold_indices, ratio,
                                        seed=4, batchsize=2, split=True):
    if split:
        return split_data_loader3D(modalities, labels, fold_indices, batchsize, random_seed=seed,
                                   valid_size=ratio, train_transform=train_transform,
                                   valid_transform=valid_transform, num_workers=0)
    return single_data_loader3D_full(modalities, labels, fold_indices, batchsize, random_seed=seed,
                                     valid_size=ratio, train_transform=train_transform, num_workers=0), None


def get_testdata_full(modalities, labels, indices, random_seed=None, shuffle=True, num_workers=4, pin_memory=False,
                      transform=[]):
    assert transform, "You must provide at least a numpy-to-torch transformation."

    dataset = StrokeLindaDataset3D(modalities=modalities, labels=labels, transform=transforms.Compose(transform),
                                   clinical='/share/data_zoe1/lucas/Linda_Segmentations/clinical_cleaned_full.csv')

    items = list(set(range(len(dataset))).intersection(set(indices)))

    if shuffle == True:
        random_state = np.random.RandomState(random_seed)
        random_state.shuffle(items)

    sampler = SubsetRandomSampler(items)

    loader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=num_workers, pin_memory=pin_memory,
                        worker_init_fn=set_np_seed) # important to have batchsize=1 because metrics is computed on batch

    return loader


class HemisphericFlipFixedToCaseId(object):
    """Flip numpy images along X-axis."""

    def __init__(self, split_id):
        self.split_id = split_id

    def __call__(self, sample):
        if int(sample[KEY_CASE_ID]) > self.split_id:
            result = emptyCopyFromSample(sample)
            if sample[KEY_IMAGES] != []:
                result[KEY_IMAGES] = np.flip(sample[KEY_IMAGES], DIM_HORIZONTAL_NUMPY_3D).copy()
            if sample[KEY_LABELS] != []:
                result[KEY_LABELS] = np.flip(sample[KEY_LABELS], DIM_HORIZONTAL_NUMPY_3D).copy()
            if sample[KEY_GLOBAL] != []:
                result[KEY_GLOBAL] = np.flip(sample[KEY_GLOBAL], DIM_HORIZONTAL_NUMPY_3D).copy()
            return result
        return sample


class ClipImages(object):
    def __init__(self, min=None, max=None):
        self.min = min
        self.max = max

    """Clip numpy images at min and max value."""
    def __call__(self, sample):
        if sample[KEY_IMAGES] != []:
            result[KEY_IMAGES] = np.clip(sample[KEY_IMAGES], self.min, self.max).copy()
        return result


class HemisphericFlip(object):
    """Flip numpy images along X-axis."""
    def __call__(self, sample):
        if random.random() > 0.5:
            result = emptyCopyFromSample(sample)
            if sample[KEY_IMAGES] != []:
                result[KEY_IMAGES] = np.flip(sample[KEY_IMAGES], DIM_HORIZONTAL_NUMPY_3D).copy()
            if sample[KEY_LABELS] != []:
                result[KEY_LABELS] = np.flip(sample[KEY_LABELS], DIM_HORIZONTAL_NUMPY_3D).copy()
            if sample[KEY_GLOBAL] != []:
                result[KEY_GLOBAL] = np.flip(sample[KEY_GLOBAL], DIM_HORIZONTAL_NUMPY_3D).copy()
            return result
        return sample


class RandomPatch(object):
    """Random patches of certain size."""
    def __init__(self, w, h, d, pad_x, pad_y, pad_z):
        self._padx = pad_x
        self._pady = pad_y
        self._padz = pad_z
        self._w = w
        self._h = h
        self._d = d

    def __call__(self, sample):
        sx, sy, sz, _ = sample[KEY_IMAGES].shape

        rand_x = random.randint(0, sx - self._w)
        rand_y = random.randint(0, sy - self._h)
        rand_z = random.randint(0, sz - self._d)

        result = emptyCopyFromSample(sample)
        if sample[KEY_IMAGES] != []:
            result[KEY_IMAGES] = sample[KEY_IMAGES][rand_x: rand_x + self._w,
                                                    rand_y: rand_y + self._h,
                                                    rand_z: rand_z + self._d, :]
        if sample[KEY_LABELS] != []:
            result[KEY_LABELS] = sample[KEY_LABELS][rand_x: rand_x + self._w - 2 * self._padx,
                                                    rand_y: rand_y + self._h - 2 * self._pady,
                                                    rand_z: rand_z + self._d - 2 * self._padz, :]
        result[KEY_GLOBAL] = sample[KEY_GLOBAL]

        return result


class PadImages(object):
    """Pad images with constant pad_value in all 6 directions (3D)."""
    def __init__(self, pad_x, pad_y, pad_z, pad_value=0):
        self._padx = pad_x
        self._pady = pad_y
        self._padz = pad_z
        self._pad_value = float(pad_value)

    def __call__(self, sample):
        sx, sy, sz, sc = sample[KEY_IMAGES].shape
        result = emptyCopyFromSample(sample)
        if sample[KEY_IMAGES] != []:
            result[KEY_IMAGES] = np.ones((sx + 2 * self._padx, sy + 2 * self._pady, sz + 2 * self._padz, sc), dtype=np.float32) * self._pad_value
            result[KEY_IMAGES][self._padx:-self._padx, self._pady:-self._pady, self._padz:-self._padz, :] = sample[KEY_IMAGES]
        result[KEY_LABELS] = sample[KEY_LABELS]
        result[KEY_GLOBAL] = sample[KEY_GLOBAL]
        return result


class UseLabelsAsImages(object):
    def __call__(self, sample):
        result = emptyCopyFromSample(sample)
        result[KEY_IMAGES] = sample[KEY_LABELS]
        result[KEY_LABELS] = sample[KEY_LABELS]
        result[KEY_GLOBAL] = sample[KEY_GLOBAL]
        return result


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, time_dim=None):
        self.time_dim = time_dim

    def __call__(self, sample):
        result = emptyCopyFromSample(sample)
        if sample[KEY_IMAGES] != []:
            result[KEY_IMAGES] = torch.from_numpy(sample[KEY_IMAGES]).permute(3, 2, 1, 0).float()
            if self.time_dim is not None:
                result[KEY_IMAGES] = result[KEY_IMAGES].unsqueeze(self.time_dim)
        if sample[KEY_LABELS] != []:
            result[KEY_LABELS] = torch.from_numpy(sample[KEY_LABELS]).permute(3, 2, 1, 0).float()
            if self.time_dim is not None:
                result[KEY_LABELS] = result[KEY_LABELS].unsqueeze(self.time_dim)
        if sample[KEY_GLOBAL] != []:
            result[KEY_GLOBAL] = torch.from_numpy(sample[KEY_GLOBAL]).permute(3, 2, 1, 0).float()
            if self.time_dim is not None:
                result[KEY_GLOBAL] = result[KEY_GLOBAL].unsqueeze(self.time_dim)
        return result


class ElasticDeform2D(object):
    """Elastic deformation of images as described in [Simard2003]
       Simard, Steinkraus and Platt, "Best Practices for Convolutional
       Neural Networks applied to Visual Document Analysis", in Proc.
       of the International Conference on Document Analysis and
       Recognition, 2003.
    """

    def __init__(self, alpha=100, sigma=4, apply_to_images=False, random=1, seed=None):
        self._alpha = alpha
        self._sigma = sigma
        self._apply_to_images = apply_to_images
        self._random = random
        self._seed = None
        if seed is not None:
            self.seed = np.random.RandomState(seed)

    def elastic_transform(self, image, alpha=100, sigma=6, random_state=None):
        new_seed = datetime.datetime.now().second + datetime.datetime.now().microsecond
        if random_state is None:
            random_state = np.random.RandomState(new_seed)

        shape = image.shape
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

        x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

        return map_coordinates(image, indices, order=1).reshape(shape), random_state

    def __call__(self, sample):
        if random.random() < self._random:
            sample[KEY_LABELS][:, :, :, 0], random_state = self.elastic_transform(sample[KEY_LABELS][:, :, :, 0], self._alpha, self._sigma, self._seed)
            for c in range(1, sample[KEY_LABELS].shape[3]):
                sample[KEY_LABELS][:, :, :, c], _ = self.elastic_transform(sample[KEY_LABELS][:, :, :, c], self._alpha, self._sigma, random_state=random_state)
            if self._apply_to_images and sample[KEY_IMAGES] != []:
                for c in range(sample[KEY_IMAGES].shape[3]):
                    sample[KEY_IMAGES][:, :, :, c], _ = self.elastic_transform(sample[KEY_IMAGES][:, :, :, c], self._alpha, self._sigma, random_state=random_state)
        return sample


class ElasticDeform(object):
    """Elastic deformation of images as described in [Simard2003]
       Simard, Steinkraus and Platt, "Best Practices for Convolutional
       Neural Networks applied to Visual Document Analysis", in Proc.
       of the International Conference on Document Analysis and
       Recognition, 2003.
    """

    def __init__(self, alpha=100, sigma=4, apply_to_images=False, random=1, seed=None):
        self._alpha = alpha
        self._sigma = sigma
        self._apply_to_images = apply_to_images
        self._random = random
        self._seed = None
        if seed is not None:
            self.seed = np.random.RandomState(seed)

    def elastic_transform(self, image, alpha=100, sigma=4, random_state=None):
        new_seed = datetime.datetime.now().second + datetime.datetime.now().microsecond
        if random_state is None:
            random_state = np.random.RandomState(new_seed)

        shape = image.shape
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dz = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha * 0.22  # 28/128  TODO: correct according to voxel spacing

        x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z + dz, (-1, 1))

        return map_coordinates(image, indices, order=1).reshape(shape), random_state

    def __call__(self, sample):
        if random.random() < self._random:
            sample[KEY_LABELS][:, :, :, 0], random_state = self.elastic_transform(sample[KEY_LABELS][:, :, :, 0], self._alpha, self._sigma, self._seed)
            for c in range(1, sample[KEY_LABELS].shape[3]):
                sample[KEY_LABELS][:, :, :, c], _ = self.elastic_transform(sample[KEY_LABELS][:, :, :, c], self._alpha, self._sigma, random_state=random_state)
            if self._apply_to_images and sample[KEY_IMAGES] != []:
                for c in range(sample[KEY_IMAGES].shape[3]):
                    sample[KEY_IMAGES][:, :, :, c], _ = self.elastic_transform(sample[KEY_IMAGES][:, :, :, c], self._alpha, self._sigma, random_state=random_state)
        return sample


class ResamplePlaneXY(object):
    """Down- or upsample images."""
    def __init__(self, scale_factor=1, mode='nearest'):
        self._scale_factor = scale_factor
        if mode == 'bilinear':
            self._order = 1
        else:
            self._order = 0

    def __call__(self, sample):
        result = emptyCopyFromSample(sample)
        result[KEY_GLOBAL] = sample[KEY_GLOBAL]

        if sample[KEY_IMAGES] != []:
            sx, sy = ndi.zoom(sample[KEY_IMAGES][:, :, 0], self._scale_factor, order=0).shape[0:2]  # just for init
            result[KEY_IMAGES] = sample[KEY_IMAGES][:sx, :sy, :, :]  # just for init correctly sized array with random values
            for c in range(sample[KEY_IMAGES].shape[DIM_CHANNEL_NUMPY_3D]):
                for z in range(sample[KEY_IMAGES].shape[DIM_DEPTH_NUMPY_3D]):
                    result[KEY_IMAGES][:, :, z, c] = ndi.zoom(sample[KEY_IMAGES][:, :, z, c], self._scale_factor, order=self._order)

        if sample[KEY_LABELS] != []:
            sx, sy = ndi.zoom(sample[KEY_LABELS][:, :, 0], self._scale_factor, order=0).shape[0:2]  # just for init
            result[KEY_LABELS] = sample[KEY_LABELS][:sx, :sy, :, :]  # just for init correctly sized array with random values
            for c in range(sample[KEY_LABELS].shape[DIM_CHANNEL_NUMPY_3D]):
                for z in range(sample[KEY_LABELS].shape[DIM_DEPTH_NUMPY_3D]):
                    result[KEY_LABELS][:, :, z, c] = ndi.zoom(sample[KEY_LABELS][:, :, z, c], self._scale_factor, order=self._order)

        return result