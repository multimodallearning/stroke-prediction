import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib


# Custom overlay function for Part1
def overlaySegment_part1(img, seg, alpha=0.75):
    overlay = img.unsqueeze(2).expand(img.size(0), img.size(1), 3).detach().numpy() / torch.max(
        img).detach().cpu().numpy()
    label_r = seg.unsqueeze(2).expand(seg.size(0), seg.size(1), 3).detach().numpy() * 0.5
    label_r[:, :, 0] = label_r[:, :, 0] * 2
    overlay[label_r > 0.5] = (1 - alpha) * overlay[label_r > 0.5] + alpha * label_r[label_r > 0.5]
    return overlay


# Custom plot function for Part1
def visualise_sample_part1(axs, sample, result, is_training, epoch, every_epoch, z_slice):
    """Imshow first sample of batch"""
    offset = 0
    phase = 'training'
    if not is_training:
        phase = 'validation'
        offset = 2

    image = sample['image'][0, 0, z_slice, :, :].cpu()
    label = sample['label'][0, 0, z_slice, :, :].cpu()
    preds = (result.detach()[0, 0, z_slice, :, :].cpu() > 0).float()

    axs[epoch // every_epoch][0 + offset].imshow(overlaySegment_part1(image, label))
    axs[epoch // every_epoch][0 + offset].set_title(str(epoch) + ': sample ' + phase + ' label')
    axs[epoch // every_epoch][0 + offset].grid(False)
    axs[epoch // every_epoch][1 + offset].imshow(overlaySegment_part1(image, preds))
    axs[epoch // every_epoch][1 + offset].set_title(str(epoch) + ': sample ' + phase + ' prediction')
    axs[epoch // every_epoch][1 + offset].grid(False)
    return axs


# initialise network weights
def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def countParam(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


# just a copied helper function from the last notebook...
# used to downscale the 3D volumes
class Scale(object):
    """Scale tensors spatially."""

    def __init__(self, width=1, height=1, depth=1):
        assert width or height or depth
        self.width = width
        self.height = height
        self.depth = depth

    def __call__(self, sample):
        sample['image'] = F.interpolate(
            sample['image'],
            scale_factor=(self.depth, self.height, self.width),
            mode='trilinear'
        )

        sample['label'] = F.interpolate(
            sample['label'].float(),
            scale_factor=(self.depth, self.height, self.width),
            mode='nearest'
        ).long()

        return sample


def augmentAffine(img_in, seg_in, strength=0.05):
    """
    3D affine augmentation on image and segmentation mini-batch on GPU.
    (affine transf. is centered: trilinear interpolation and zero-padding used for sampling)
    :input: img_in batch (torch.cuda.FloatTensor), seg_in batch (torch.cuda.LongTensor)
    :return: augmented BxCxTxHxW image batch (torch.cuda.FloatTensor), augmented BxTxHxW seg batch (torch.cuda.LongTensor)
    """
    B, C, D, H, W = img_in.size()
    affine_matrix = (torch.eye(3, 4).unsqueeze(0) + torch.randn(B, 3, 4) * strength).to(img_in.device)

    meshgrid = F.affine_grid(affine_matrix, torch.Size((B, 1, D, H, W)))

    img_out = F.grid_sample(img_in, meshgrid, padding_mode='border')
    seg_out = F.grid_sample(seg_in.float().unsqueeze(1), meshgrid, mode='nearest').long().squeeze(1)

    return img_out, seg_out


def loadRegData(str_fix_img, str_fix_label, str_mov_img, str_mov_label):
    fixed_img = nib.load(str_fix_img).get_data()
    fixed_lab = nib.load(str_fix_label).get_data()

    moving_img = nib.load(str_mov_img).get_data()
    moving_lab = nib.load(str_mov_label).get_data()

    # as in the first part of the tutorial, we switch axes and add channel & batch
    # dimensions to our tensors

    fixed_img = torch.from_numpy(fixed_img).permute(2, 1, 0).unsqueeze(0).unsqueeze(0)
    fixed_lab = torch.from_numpy(fixed_lab).permute(2, 1, 0).unsqueeze(0).unsqueeze(0).float()

    moving_img = torch.from_numpy(moving_img).permute(2, 1, 0).unsqueeze(0).unsqueeze(0)
    moving_lab = torch.from_numpy(moving_lab).permute(2, 1, 0).unsqueeze(0).unsqueeze(0).float()

    # also, we subsample the images with the function from part 1...
    fixed_sample = {'image': fixed_img, 'label': fixed_lab}
    moving_sample = {'image': moving_img, 'label': moving_lab}

    Scale_03 = Scale(0.3, 0.3, 0.3)
    fixed_sample_sub = Scale_03(fixed_sample)
    moving_sample_sub = Scale_03(moving_sample)

    fixed_img = fixed_sample_sub['image']
    fixed_lab = fixed_sample_sub['label'].float()
    moving_img = moving_sample_sub['image']
    moving_lab = moving_sample_sub['label'].float()

    sz_0_max = torch.max(torch.Tensor([fixed_img.size(2), moving_img.size(2)])).item()
    sz_1_max = torch.max(torch.Tensor([fixed_img.size(3), moving_img.size(3)])).item()
    sz_2_max = torch.max(torch.Tensor([fixed_img.size(4), moving_img.size(4)])).item()

    p_fix_0 = int(sz_0_max - fixed_img.size(2))
    p_fix_1 = int(sz_1_max - fixed_img.size(3))
    p_fix_2 = int(sz_2_max - fixed_img.size(4))

    p_mov_0 = int(sz_0_max - moving_img.size(2))
    p_mov_1 = int(sz_1_max - moving_img.size(3))
    p_mov_2 = int(sz_2_max - moving_img.size(4))

    p3d_fix = (0, p_fix_2, 0, p_fix_1, 0, p_fix_0)
    p3d_mov = (0, p_mov_2, 0, p_mov_1, 0, p_mov_0)

    fixed_img = torch.nn.functional.pad(fixed_img, p3d_fix, 'replicate')
    fixed_lab = torch.nn.functional.pad(fixed_lab, p3d_fix, 'replicate')

    moving_img = torch.nn.functional.pad(moving_img, p3d_mov, 'replicate')
    moving_lab = torch.nn.functional.pad(moving_lab, p3d_mov, 'replicate')

    return fixed_img, fixed_lab, moving_img, moving_lab


def MINDSSC3d(img_in, kernel_hw=2, delta=3):
    d = delta
    H = img_in.size(2);
    W = img_in.size(3);
    D = img_in.size(4)
    # define spatial offset layout for 12 self-similarity patches
    theta_ssc = torch.Tensor(2, 12, 3)
    theta_ssc[0, :, 0] = torch.Tensor([-d, -d, 0, 0, 0, 0, 0, 0, 0, 0, +d, +d]) / H
    theta_ssc[0, :, 1] = torch.Tensor([0, 0, 0, 0, +d, +d, -d, -d, 0, 0, 0, 0]) / W
    theta_ssc[0, :, 2] = torch.Tensor([0, 0, -d, -d, 0, 0, 0, 0, +d, +d, 0, 0]) / D
    theta_ssc[1, :, 0] = torch.Tensor([0, 0, 0, +d, 0, 0, -d, 0, -d, +d, 0, 0]) / H
    theta_ssc[1, :, 1] = torch.Tensor([0, +d, -d, 0, 0, 0, 0, 0, 0, 0, -d, +d]) / W
    theta_ssc[1, :, 2] = torch.Tensor([-d, 0, 0, 0, -d, +d, 0, +d, 0, 0, 0, 0]) / D
    C = theta_ssc.size(1)
    theta_ssc = theta_ssc.cuda()
    with torch.no_grad():
        # create regular 3D sampling grid for all feature locations
        grid_xyz = F.affine_grid(torch.eye(3, 4).unsqueeze(0), (1, 1, H, W, D)).view(1, 1, -1, 1, 3).cuda()
        # compute patch distances with box-filter (as described in MIND paper)
        sampled = F.grid_sample(img_in, grid_xyz + theta_ssc[0, :, :].view(1, -1, 1, 1, 3)).view(1, C, H, W, D)
        sampled -= F.grid_sample(img_in, grid_xyz + theta_ssc[1, :, :].view(1, -1, 1, 1, 3)).view(1, C, H, W, D)
        mind = F.avg_pool3d(torch.abs(sampled) ** 2, kernel_hw * 2 + 1, stride=1, padding=kernel_hw)
        # use MIND equation to obtain contrast-invariant features for registration
        mind -= torch.min(mind, 1, keepdim=True)[0]
        mind /= (torch.sum(mind, 1, keepdim=True) + 0.001)
        mind = torch.exp(-mind)
    del sampled;
    del grid_xyz;
    del theta_ssc;
    torch.cuda.empty_cache()
    return mind


def img_warp(img, id_field, param_field):
    # expects the image to warp, the full resolution identity field
    # and the coarser param_field containing the displacments

    # 1) upsamples the coarse grid to the same resolution as the image resolution
    param_field = param_field.permute(0, 4, 1, 2, 3)
    param_field = torch.nn.functional.interpolate(param_field,
                                                  size=(img.size(2), img.size(3), img.size(4)),
                                                  mode='trilinear', align_corners=True)
    # 2) smoothes the field... (also a kind of regularisation)
    smoother = torch.nn.Sequential(torch.nn.AvgPool3d(kernel_size=5, stride=1, padding=2),
                                   torch.nn.AvgPool3d(kernel_size=5, stride=1, padding=2),
                                   torch.nn.AvgPool3d(kernel_size=5, stride=1, padding=2))
    param_field = smoother(param_field)
    param_field = param_field.permute(0, 2, 3, 4, 1)
    # 3) employing grid_sampler to interpolate off-grid image values
    warped = torch.nn.functional.grid_sample(img, param_field + id_field,
                                             padding_mode="border")
    return warped


def label_warp(img, id_field, param_field):
    # same as above, but the grid_sampling uses nearest neighbor interpolation
    # to preserve labels -> no interpolation between 0 & 1 e.g. and "adding"
    # false labels as 0.5
    param_field = param_field.permute(0, 4, 1, 2, 3)
    param_field = torch.nn.functional.interpolate(param_field,
                                                  size=(img.size(2), img.size(3), img.size(4)),
                                                  mode='trilinear', align_corners=True)
    smoother = torch.nn.Sequential(torch.nn.AvgPool3d(kernel_size=5, stride=1, padding=2),
                                   torch.nn.AvgPool3d(kernel_size=5, stride=1, padding=2),
                                   torch.nn.AvgPool3d(kernel_size=5, stride=1, padding=2))
    param_field = smoother(param_field)
    param_field = param_field.permute(0, 2, 3, 4, 1)
    warped = torch.nn.functional.grid_sample(img, param_field + id_field,
                                             padding_mode="border",
                                             mode='nearest')  # HERE: nearest neighbor
    return warped


def jacobian_determinant_3d(dense_flow):
    B, _, H, W, D = dense_flow.size()

    dense_pix = dense_flow.to(dense_flow.device).flip(1) / (torch.Tensor([H - 1, W - 1, D - 1]) / 2).view(1, 3, 1, 1, 1)
    gradz = nn.Conv3d(3, 3, (3, 1, 1), padding=(1, 0, 0), bias=False, groups=3)
    gradz.weight.data[:, 0, :, 0, 0] = torch.tensor([-0.5, 0, 0.5]).view(1, 3).repeat(3, 1)
    gradz.to(dense_flow.device)
    grady = nn.Conv3d(3, 3, (1, 3, 1), padding=(0, 1, 0), bias=False, groups=3)
    grady.weight.data[:, 0, 0, :, 0] = torch.tensor([-0.5, 0, 0.5]).view(1, 3).repeat(3, 1)
    grady.to(dense_flow.device)
    gradx = nn.Conv3d(3, 3, (1, 1, 3), padding=(0, 0, 1), bias=False, groups=3)
    gradx.weight.data[:, 0, 0, 0, :] = torch.tensor([-0.5, 0, 0.5]).view(1, 3).repeat(3, 1)
    gradx.to(dense_flow.device)
    with torch.no_grad():
        jacobian = torch.cat((gradz(dense_pix), grady(dense_pix), gradx(dense_pix)), 0) + torch.eye(3, 3).view(3, 3, 1,
                                                                                                               1, 1).to(
            dense_flow.device)
        jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
        jac_det = jacobian[0, 0, :, :, :] * (
                    jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :,
                                                                                                  :]) - \
                  jacobian[1, 0, :, :, :] * (
                              jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[2,
                                                                                                            1, :, :,
                                                                                                            :]) + \
                  jacobian[2, 0, :, :, :] * (
                              jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[1,
                                                                                                            1, :, :, :])

    return jac_det
