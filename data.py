import os
import csv
import nibabel as nib
import random
import datetime

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


KEY_CASE_ID = 'case_id'
KEY_CLINICAL_IDX = 'clinical_idx'
KEY_IMAGES = 'images'
KEY_LABELS = 'labels'
KEY_GLOBAL = 'clinical'

DIM_HORIZONTAL_NUMPY_3D = 0
DIM_DEPTH_NUMPY_3D = 2
DIM_CHANNEL_NUMPY_3D = 3
DIM_CHANNEL_TORCH3D_5 = 1


class StrokeLindaDataset3D(Dataset):
    """Ischemic stroke dataset with CBV, TTD, clinical data, and CBVmap, TTDmap, FUmap, and interpolations."""
    PATH_ROOT = '/share/data_zoe1/lucas/Linda_Segmentations'
    PATH_CSV = '/share/data_zoe1/lucas/Linda_Segmentations/clinical_cleaned.csv'
    FN_PREFIX = 'train'
    FN_PATTERN = '{1}/{0}{1}{2}.nii.gz'
    ROW_OFFSET = 1
    COL_OFFSET = 1

    def __init__(self, root_dir=PATH_ROOT, modalities='', labels='', clinical=PATH_CSV, transform=None,
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
        result[KEY_GLOBAL] = np.array(result[KEY_GLOBAL]).reshape((1, 1, 1, len(clinical_data)))

        for label in self._labels:
            result[KEY_LABELS].append(self._load_image_data_from_nifti(case_id, label))
        result[KEY_LABELS] = np.concatenate(result[KEY_LABELS], axis=DIM_CHANNEL_NUMPY_3D)

        for modality in self._modalities:
            result[KEY_IMAGES].append(self._load_image_data_from_nifti(case_id, modality))
        result[KEY_IMAGES] = np.concatenate(result[KEY_IMAGES], axis=DIM_CHANNEL_NUMPY_3D)

        if self._transform:
            result = self._transform(result)

        return result


def emptyCopyFromSample(sample):
    result = {KEY_CASE_ID: int(sample[KEY_CASE_ID])}
    return result


def set_np_seed(workerid):
    torch_seed = torch.initial_seed()
    numpy_seed = torch_seed % np.iinfo(np.int32).max
    np.random.seed(numpy_seed)


def split_data_loader3D(modalities, labels, indices, batch_size, random_seed=None, valid_size=0.5, shuffle=True,
                        num_workers=4, pin_memory=False, train_transform=[], valid_transform=[]):

    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

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
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(dataset_train,
                    batch_size=batch_size, sampler=train_sampler,
                    num_workers=num_workers, pin_memory=pin_memory,
                    worker_init_fn=set_np_seed)

    valid_loader = DataLoader(dataset_valid,
                    batch_size=batch_size, sampler=valid_sampler,
                    num_workers=num_workers, pin_memory=pin_memory)

    return (train_loader, valid_loader)


class HemisphericFlipFixedToCaseId(object):
    """Flip numpy images along X-axis."""

    def __init__(self, split_id):
        self.split_id = split_id

    def __call__(self, sample):
        if int(sample[KEY_CASE_ID]) > self.split_id:
            result = emptyCopyFromSample(sample)
            result[KEY_IMAGES] = np.flip(sample[KEY_IMAGES], DIM_HORIZONTAL_NUMPY_3D).copy()
            result[KEY_LABELS] = np.flip(sample[KEY_LABELS], DIM_HORIZONTAL_NUMPY_3D).copy()
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
        result[KEY_IMAGES] = sample[KEY_IMAGES][rand_x: rand_x + self._w, rand_y: rand_y + self._h, rand_z: rand_z + self._d, :]
        result[KEY_LABELS] = sample[KEY_LABELS][rand_x: rand_x + self._w - 2 * self._padx, rand_y: rand_y + self._h - 2 * self._pady, rand_z: rand_z + self._d - 2 * self._padz, :]
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
        result[KEY_IMAGES] = np.ones((sx + 2 * self._padx, sy + 2 * self._pady, sz + 2 * self._padz, sc), dtype=np.float32) * self._pad_value
        result[KEY_IMAGES][self._padx:-self._padx, self._pady:-self._pady, self._padz:-self._padz, :] = sample[KEY_IMAGES]
        result[KEY_LABELS] = sample[KEY_LABELS]
        result[KEY_GLOBAL] = sample[KEY_GLOBAL]
        return result


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        result = emptyCopyFromSample(sample)
        result[KEY_IMAGES] = torch.from_numpy(sample[KEY_IMAGES]).permute(3, 2, 1, 0)
        result[KEY_LABELS] = torch.from_numpy(sample[KEY_LABELS]).permute(3, 2, 1, 0)
        result[KEY_GLOBAL] = torch.from_numpy(sample[KEY_GLOBAL]).permute(3, 2, 1, 0)
        return result


class ElasticDeform(object):
    """Elastic deformation of images as described in [Simard2003]
       Simard, Steinkraus and Platt, "Best Practices for Convolutional
       Neural Networks applied to Visual Document Analysis", in Proc.
       of the International Conference on Document Analysis and
       Recognition, 2003.
    """

    def __init__(self, alpha=100, sigma=4, apply_to_images=False):
        self._alpha = alpha
        self._sigma = sigma
        self._apply_to_images = apply_to_images

    def elastic_transform(self, image, alpha=100, sigma=4, random_state=None):
        new_seed = datetime.datetime.now().second + datetime.datetime.now().microsecond
        if random_state is None:
            random_state = np.random.RandomState(new_seed)

        shape = image.shape
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dz = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha * 0.22  # 28/128

        x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z + dz, (-1, 1))

        return map_coordinates(image, indices, order=1).reshape(shape), random_state

    def __call__(self, sample):
        sample[KEY_LABELS][:, :, :, 0], random_state = self.elastic_transform(sample[KEY_LABELS][:, :, :, 0],
                                                                              self._alpha, self._sigma)
        for c in range(1, sample[KEY_LABELS].shape[3]):
            sample[KEY_LABELS][:, :, :, c], _ = self.elastic_transform(sample[KEY_LABELS][:, :, :, c], self._alpha,
                                                                       self._sigma, random_state=random_state)
        if self._apply_to_images:
            for c in range(sample[KEY_IMAGES].shape[3]):
                sample[KEY_IMAGES][:, :, :, c], _ = self.elastic_transform(sample[KEY_IMAGES][:, :, :, c], self._alpha,
                                                                           self._sigma, random_state=random_state)
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
        sx, sy = ndi.zoom(sample[KEY_IMAGES][:, :, 0], self._scale_factor, order=0).shape[0:2]  # just for init
        result[KEY_IMAGES] = sample[KEY_IMAGES][:sx, :sy, :, :]  # just for init correctly sized array with random values
        result[KEY_LABELS] = sample[KEY_LABELS][:sx, :sy, :, :]  # just for init correctly sized array with random values
        result[KEY_GLOBAL] = sample[KEY_GLOBAL]
        for c in range(sample[KEY_IMAGES].shape[DIM_CHANNEL_NUMPY_3D]):
            for z in range(sample[KEY_IMAGES].shape[DIM_DEPTH_NUMPY_3D]):
                result[KEY_IMAGES][:, :, z, c] = ndi.zoom(sample[KEY_IMAGES][:, :, z, c], self._scale_factor, order=self._order)
        for c in range(sample[KEY_LABELS].shape[DIM_CHANNEL_NUMPY_3D]):
            for z in range(sample[KEY_LABELS].shape[DIM_DEPTH_NUMPY_3D]):
                result[KEY_LABELS][:, :, z, c] = ndi.zoom(sample[KEY_LABELS][:, :, z, c], self._scale_factor, order=self._order)
        return result
