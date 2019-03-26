import torch.nn.functional as F
import torch.nn as nn
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.tight_layout()
plt.axis('off')

import nibabel as nib
import numpy as np

import time
from tqdm import tqdm as tqdm

import scipy.ndimage as ndi

from common.bvm_utils import img_warp, label_warp, jacobian_determinant_3d, overlaySegment_part1


########################################################################################################################


def get_gatter(size):
    gatter = torch.ones(size, requires_grad=False).cuda()
    gatter[:, :, 1::4, :, :] = 0.75
    gatter[:, :, 2::4, :, :] = 0.5
    gatter[:, :, 3::4, :, :] = 0.75
    gatter[:, :, :, 3::12, :] = 0
    gatter[:, :, :, 4::12, :] = 0
    #gatter[:, :, :, 5::12, :] = 0
    gatter[:, :, :, :, 3::12] = 0
    gatter[:, :, :, :, 4::12] = 0
    #gatter[:, :, :, :, 5::12] = 0
    return gatter


def dice_val(f_label, m_label):
    numerator = 2.0 * torch.sum(f_label.view(-1) * m_label.view(-1))
    denominator = torch.sum(f_label.view(-1)) + torch.sum(m_label.view(-1))
    return numerator / denominator


def loadRegData(str_fix_img, str_fix_label, str_mov_img, str_mov_label, str_fuct_label):
    fixed_img = nib.load(str_fix_img).get_data()
    fixed_lab = nib.load(str_fix_label).get_data()

    moving_img = nib.load(str_mov_img).get_data()
    moving_lab = nib.load(str_mov_label).get_data()

    fuct = nib.load(str_fuct_label)
    fuct_lab = fuct.get_data()

    fixed_img = torch.from_numpy(fixed_img).permute(2, 1, 0).unsqueeze(0).unsqueeze(0)
    fixed_lab = torch.from_numpy(fixed_lab).permute(2, 1, 0).unsqueeze(0).unsqueeze(0).float()

    moving_img = torch.from_numpy(moving_img).permute(2, 1, 0).unsqueeze(0).unsqueeze(0)
    moving_lab = torch.from_numpy(moving_lab).permute(2, 1, 0).unsqueeze(0).unsqueeze(0).float()

    fuct_lab = torch.from_numpy(fuct_lab).permute(2, 1, 0).unsqueeze(0).unsqueeze(0).float()

    return fixed_img, fixed_lab, moving_img, moving_lab, fuct_lab, fuct.get_affine()


def dFdx(field):
    # expects a flowfield : 1xDxHxWx3 -> last dim: x,y,z
    # computes per component derivative (last 3 components in x_dir)
    # and clips it accordingly in the other dimensions, so the size will be diminished by 2 equally
    dx = (field[:, 2:, 1:-1, 1:-1, 0] - field[:, :-2, 1:-1, 1:-1, 0]) / 2.0
    dy = (field[:, 1:-1, 2:, 1:-1, 0] - field[:, 1:-1, :-2, 1:-1, 0]) / 2.0
    dz = (field[:, 1:-1, 1:-1, 2:, 0] - field[:, 1:-1, 1:-1, :-2, 0]) / 2.0
    return torch.stack((dx, dy, dz), dim=4)


def dFdy(field):
    # expects a flowfield : 1xDxHxWx3 -> last dim: x,y,z
    # computes per component derivative (last 3 components in x_dir)
    # and clips it accordingly in the other dimensions, so the size will be diminished by 2 equally
    dx = (field[:, 2:, 1:-1, 1:-1, 1] - field[:, :-2, 1:-1, 1:-1, 1]) / 2.0
    dy = (field[:, 1:-1, 2:, 1:-1, 1] - field[:, 1:-1, :-2, 1:-1, 1]) / 2.0
    dz = (field[:, 1:-1, 1:-1, 2:, 1] - field[:, 1:-1, 1:-1, :-2, 1]) / 2.0
    return torch.stack((dx, dy, dz), dim=4)


def dFdz(field):
    # expects a flowfield : 1xDxHxWx3 -> last dim: x,y,z
    # computes per component derivative (last 3 components in x_dir)
    # and clips it accordingly in the other dimensions, so the size will be diminished by 2 equally
    dx = (field[:, 2:, 1:-1, 1:-1, 2] - field[:, :-2, 1:-1, 1:-1, 2]) / 2.0
    dy = (field[:, 1:-1, 2:, 1:-1, 2] - field[:, 1:-1, :-2, 1:-1, 2]) / 2.0
    dz = (field[:, 1:-1, 1:-1, 2:, 2] - field[:, 1:-1, 1:-1, :-2, 2]) / 2.0
    return torch.stack((dx, dy, dz), dim=4)


def _transform_vec_field(vec_field_0, vec_field_i):
    grid_id = torch.nn.functional.affine_grid(torch.eye(3, 4).view(1, 3, 4), vec_field_0.size()).cuda()
    grid_i = vec_field_i.permute(0, 2, 3, 4, 1)
    return torch.nn.functional.grid_sample(vec_field_0, grid_id + grid_i)


def _integrate_vec_field(vec_field_0, vec_field_i):
    return vec_field_i + _transform_vec_field(vec_field_0, vec_field_i)


def _negate(vec_field):
    return -vec_field


########################################################################################################################


class BendingEnergyReg(torch.nn.Module):
    def __init__(self):
        super(BendingEnergyReg, self).__init__()

    def forward(self, grid):
        # compute dx, dy, dz
        dx = dFdx(grid)
        dy = dFdy(grid)
        dz = dFdz(grid)

        # second derivatives & mixed terms
        dxx = dFdx(dx)
        dyy = dFdy(dy)
        dzz = dFdz(dz)
        dxy = dFdy(dx)
        dyz = dFdz(dy)
        dxz = dFdz(dx)

        # putting it all together
        penalty = torch.sum(dxx ** 2 + dyy ** 2 + dzz ** 2 + 2.0 * (dxy ** 2 + dyz ** 2 + dxz ** 2))

        return penalty * (1.0 / torch.numel(dxx))


# define a torch.nn.Module that contains the updatable parameters
class displacement_grid(nn.Module):
    def __init__(self, sz0, sz1, sz2):
        super(displacement_grid, self).__init__()
        self.grid = torch.nn.Parameter(torch.zeros(1, sz0, sz1, sz2, 3))

    def forward(self, i, n, scale_factor=1):
        assert i <= n
        init_grid = self.grid / n
        last_grid = torch.zeros(init_grid.permute(0, 4, 1, 2, 3).size()).cuda()
        for _ in range(i):
            last_grid = (_integrate_vec_field(init_grid.permute(0, 4, 1, 2, 3), last_grid))

        if scale_factor != 1:
            last_grid = F.interpolate(F.interpolate(last_grid, scale_factor=scale_factor), scale_factor=1/scale_factor)

        last_grid = last_grid.permute(0, 2, 3, 4, 1)

        return last_grid


########################################################################################################################

ids = [2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]  # TODO 1 IS MISSING
threshs = [0.3, 0.4, 0.5]

results = [[[]] * len(threshs)] * len(ids)

for n_integration_steps in [10]:
    for reg_iters in [500]:
        for alpha in [10]:
            for beta in [0]:
                for thx, thresh in enumerate(threshs):
                    for idx, id in enumerate(ids):
                        id = str(id)
                        print('\n\nID\t' + id + '\tThresh\t' + str(thresh) + '\t:')

                        start_time = time.time()

                        with torch.no_grad():
                            data_base_path = '/share/data_rosita2/lucas/Linda_Segmentations/'

                            fixed_image_path = data_base_path + '/' + id + '/train' + id + '_CBV_reg1_downsampled.nii.gz'
                            fixed_label_path = data_base_path + '/' + id + '/train' + id + '_CBVmap_subset_reg1_downsampled.nii.gz'

                            moving_image_path = data_base_path + '/' + id + '/train' + id + '_TTD_reg1_downsampled.nii.gz'
                            moving_label_path = data_base_path + '/' + id + '/train' + id + '_TTDmap_reg1_downsampled.nii.gz'

                            fuct_label_path = data_base_path + '/' + id + '/train' + id + '_FUCT_MAP_T_Samplespace_subset_reg1_downsampled.nii.gz'

                            fixed_image, \
                            fixed_label, \
                            moving_image, \
                            moving_label, \
                            fuct_label,\
                            fuct_affine = loadRegData(fixed_image_path, fixed_label_path,
                                                      moving_image_path, moving_label_path,
                                                      fuct_label_path)

                        plt_slice = 13

                        fixed_imshow = fixed_image.clone()
                        fixed_imshow[fixed_imshow < 0] = 0
                        fixed_imshow[fixed_imshow > 12] = 12
                        fixed_imshow /= 12

                        moving_imshow = moving_image.clone()
                        moving_imshow[moving_imshow < 0] = 0
                        moving_imshow[moving_imshow > 40] = 40
                        moving_imshow /= 40
                        moving_imshow = moving_imshow ** (1 / 3)

                        dice_init = dice_val(fixed_label, moving_label)

                        with torch.no_grad():
                            fixed_feat = ndi.distance_transform_edt(fixed_label.numpy() < 0.5)
                            fixed_feat = torch.nn.functional.tanh(torch.from_numpy(fixed_feat) / 50).float()
                            moving_feat = ndi.distance_transform_edt(moving_label.numpy() < 0.5)
                            moving_feat = torch.nn.functional.tanh(torch.from_numpy(moving_feat) / 50).float()

                            print(fixed_feat.size())
                            print(moving_feat.size())


                        '''
                        plt_slice = 13
                        plt.figure(figsize=(15, 8))
                        plt.subplot(121)
                        plt.imshow(fixed_feat[0, 0, plt_slice, :, :].detach().cpu().numpy())
                        plt.subplot(122)
                        plt.imshow(moving_feat[0, 0, plt_slice, :, :].detach().cpu().numpy())
                        '''


                        sim_measure = torch.nn.MSELoss()  # L1Loss()


                        gatter = get_gatter(fixed_image.size())



                        # define the coarse parameter grid -> identity field at first
                        with torch.no_grad():
                            sub_factor = 4  # 8

                            s0 = fixed_image.size(2)
                            s1 = fixed_image.size(3)
                            s2 = fixed_image.size(4)

                            s0_sub = int(s0 / sub_factor)
                            s1_sub = int(s1 / sub_factor)
                            s2_sub = int(s2 / sub_factor)

                            m0, m1, m2 = torch.meshgrid(torch.linspace(-1, 1, s0),
                                                        torch.linspace(-1, 1, s1),
                                                        torch.linspace(-1, 1, s2))

                            id_field = torch.stack((m2, m1, m0), dim=3).unsqueeze(0)

                        if True:
                            with torch.enable_grad():
                                # this is the parameter tensor
                                displ = displacement_grid(s0_sub, s1_sub, s2_sub).cuda()

                                reg_regul = BendingEnergyReg()

                                sim_evo = torch.zeros(reg_iters, requires_grad=False)
                                regul_evo = torch.zeros(reg_iters, requires_grad=False)

                                reg_adam = torch.optim.Adam(displ.parameters(), lr=0.001)  # [displ], lr=0.001)

                                id_field = id_field.cuda().detach()
                                fixed_feat = fixed_feat.cuda().detach()
                                moving_feat = moving_feat.cuda().detach()

                                progress = tqdm(range(reg_iters), desc='progress')

                                for rdx in progress:
                                    reg_adam.zero_grad()

                                    last_grid = displ(n_integration_steps, n_integration_steps)
                                    neg_grid = _negate(last_grid)

                                    warped_feat = img_warp(moving_feat, id_field, last_grid)
                                    warped_feat_negated = img_warp(fixed_feat, id_field, neg_grid)

                                    similarity_val = sim_measure(warped_feat, fixed_feat) + \
                                                     beta * sim_measure(warped_feat_negated, moving_feat)
                                    regularizer_val = reg_regul(last_grid) + beta * reg_regul(neg_grid)
                                    optim_val = similarity_val + alpha * regularizer_val

                                    optim_val.backward()
                                    reg_adam.step()

                                    sim_evo[rdx] = similarity_val.item()
                                    regul_evo[rdx] = regularizer_val.item()

                                    progress.set_postfix(sim=similarity_val.item(), reg=regularizer_val.item())

                            plt.figure()
                            plt.plot(sim_evo.numpy(), label='Similarity')
                            plt.plot(regul_evo.numpy(), label='Regularizer')
                            plt.legend()

                            torch.save(displ, '/share/data_rosita2/lucas/NOT_IN_BACKUP/tmp/exps/miccai_train' + id + '_thr' + str(thresh) + '_.displ')
                        else:
                            with torch.enable_grad():
                                id_field = id_field.cuda().detach()
                                fixed_feat = fixed_feat.cuda().detach()
                                moving_feat = moving_feat.cuda().detach()
                                displ = torch.load('/share/data_rosita2/lucas/NOT_IN_BACKUP/tmp/exps/miccai_train' + id + '_thr' + str(thresh) + '_.displ').cuda()

                        gatter_warped = img_warp(gatter, id_field, displ.grid)
                        warped_feat = img_warp(moving_feat, id_field, displ.grid)
                        last_grid = displ(n_integration_steps, n_integration_steps)
                        gatter_warped_integrated = img_warp(gatter, id_field, last_grid)
                        warped_feat_integrated = img_warp(moving_feat, id_field, last_grid)

                        plt.figure(figsize=(30, 30))
                        plt.subplot(331)
                        plt.title('Moving warped')
                        plt.imshow(warped_feat[0, 0, plt_slice, :, :].detach().cpu() == 0, cmap='gray', vmin=0, vmax=1)
                        plt.subplot(332)
                        plt.title('Moving warped (int.)')
                        plt.imshow(warped_feat_integrated[0, 0, plt_slice, :, :].detach().cpu() == 0, cmap='gray', vmin=0, vmax=1)
                        plt.subplot(333)
                        plt.title('Fixed')
                        plt.imshow(fixed_feat[0, 0, plt_slice, :, :].cpu().detach() == 0, cmap='gray', vmin=0, vmax=1)

                        plt.subplot(334)
                        plt.title('Moving DM warped')
                        plt.imshow(warped_feat[0, 0, plt_slice, :, :].detach().cpu(), cmap='gray', vmin=0, vmax=1)
                        plt.subplot(335)
                        plt.title('Moving DM warped (int.)')
                        plt.imshow(warped_feat_integrated[0, 0, plt_slice, :, :].detach().cpu(), cmap='gray', vmin=0, vmax=1)
                        plt.subplot(336)
                        plt.title('Fixed DM')
                        plt.imshow(fixed_feat[0, 0, plt_slice, :, :].cpu().detach(), cmap='gray', vmin=0, vmax=1)

                        plt.subplot(337)
                        plt.title('Field warped')
                        plt.imshow(gatter_warped[0, 0, plt_slice, :, :].detach().cpu(), cmap='gray', vmin=0, vmax=1)
                        plt.subplot(338)
                        plt.title('Field warped (int.)')
                        plt.imshow(gatter_warped_integrated[0, 0, plt_slice, :, :].detach().cpu(), cmap='gray', vmin=0, vmax=1)
                        plt.subplot(339)
                        plt.title('Moving')
                        plt.imshow(moving_feat[0, 0, plt_slice, :, :].detach().cpu() == 0, cmap='gray', vmin=0, vmax=1)

                        plt.savefig('/share/data_rosita2/lucas/NOT_IN_BACKUP/tmp/exps/miccai_train' + id + '_thr' + str(thresh) + '_0.png', bbox_inches="tight")

                        negate_displgrid = _negate(displ.grid)
                        negate_last_grid = _negate(last_grid)
                        gatter_warped = img_warp(gatter, id_field, negate_displgrid)
                        gatter_warped_integrated = img_warp(gatter, id_field, negate_last_grid)
                        warped_feat = img_warp(fixed_feat, id_field, negate_displgrid)
                        warped_feat_integrated = img_warp(fixed_feat, id_field, negate_last_grid)

                        plt.figure(figsize=(30, 30))
                        plt.subplot(331)
                        plt.title('Fixed warped')
                        plt.imshow(warped_feat[0, 0, plt_slice, :, :].detach().cpu() == 0, cmap='gray', vmin=0, vmax=1)
                        plt.subplot(332)
                        plt.title('Fixed warped (int.)')
                        plt.imshow(warped_feat_integrated[0, 0, plt_slice, :, :].detach().cpu() == 0, cmap='gray', vmin=0, vmax=1)
                        plt.subplot(333)
                        plt.title('Moving')
                        plt.imshow(moving_feat[0, 0, plt_slice, :, :].cpu().detach() == 0, cmap='gray', vmin=0, vmax=1)

                        plt.subplot(334)
                        plt.title('Fixed DM warped')
                        plt.imshow(warped_feat[0, 0, plt_slice, :, :].detach().cpu(), cmap='gray', vmin=0, vmax=1)
                        plt.subplot(335)
                        plt.title('Fixed DM warped (int.)')
                        plt.imshow(warped_feat_integrated[0, 0, plt_slice, :, :].detach().cpu(), cmap='gray', vmin=0, vmax=1)
                        plt.subplot(336)
                        plt.title('Moving DM')
                        plt.imshow(moving_feat[0, 0, plt_slice, :, :].cpu().detach(), cmap='gray', vmin=0, vmax=1)

                        plt.subplot(337)
                        plt.title('Field warped')
                        plt.imshow(gatter_warped[0, 0, plt_slice, :, :].detach().cpu(), cmap='gray', vmin=0, vmax=1)
                        plt.subplot(338)
                        plt.title('Field warped (int.)')
                        plt.imshow(gatter_warped_integrated[0, 0, plt_slice, :, :].detach().cpu(), cmap='gray', vmin=0, vmax=1)
                        plt.subplot(339)
                        plt.title('Fixed')
                        plt.imshow(fixed_feat[0, 0, plt_slice, :, :].detach().cpu() == 0, cmap='gray', vmin=0, vmax=1)

                        plt.savefig('/share/data_rosita2/lucas/NOT_IN_BACKUP/tmp/exps/miccai_train' + id + '_thr' + str(thresh) + '_1.png', bbox_inches="tight")

                        with torch.no_grad():
                            warped_label = label_warp(moving_label.cuda(), id_field, displ.grid).cpu()
                            warped_label_integrated = label_warp(moving_label.cuda(), id_field, last_grid).cpu()
                            warped_label_integrated_time = label_warp(moving_label.cuda(), id_field, displ(int(round(n_integration_steps*thresh)), n_integration_steps)).cpu()
                            warped_label_integrated_time_resampled = label_warp(moving_label.cuda(), id_field, displ(int(round(n_integration_steps * thresh)), n_integration_steps, 0.25)).cpu()

                            img = nib.Nifti1Image(warped_label_integrated_time[0,0].permute(2,1,0).numpy(), fuct_affine)
                            nib.save(img, '/share/data_rosita2/lucas/NOT_IN_BACKUP/tmp/exps/miccai_train' + id + '_thr' + str(thresh) + '.nii.gz')

                            print('Dice init       :', float(dice_init))
                            print('Dice reg        :', float(dice_val(fixed_label, warped_label)))
                            print('Dice regI       :', float(dice_val(fixed_label, warped_label_integrated)))
                            print('Dice ' + str(thresh) + 'I     FU:', float(dice_val(fuct_label, warped_label_integrated_time)))
                            print('Dice ' + str(thresh) + 'I     FU:', float(dice_val(fuct_label, warped_label_integrated_time_resampled)), '(resampled)')

                            results[idx][thx].append(float(dice_init))
                            results[idx][thx].append(float(dice_val(fixed_label, warped_label)))
                            results[idx][thx].append(float(dice_val(fixed_label, warped_label_integrated)))
                            results[idx][thx].append(float(dice_val(fuct_label, warped_label_integrated_time)))
                            results[idx][thx].append(float(dice_val(fuct_label, warped_label_integrated_time_resampled)))

                            plt.figure(figsize=(30, 6))
                            plt.subplot(151)
                            plt.imshow(overlaySegment_part1(fixed_imshow[0, 0, plt_slice, :, :], warped_label[0, 0, plt_slice, :, :]))
                            plt.subplot(152)
                            plt.imshow(warped_label[0, 0, plt_slice, :, :], cmap='gray', vmin=0, vmax=1)
                            plt.subplot(153)
                            plt.imshow(warped_label_integrated[0, 0, plt_slice, :, :], cmap='gray', vmin=0, vmax=1)
                            plt.subplot(154)
                            plt.imshow(overlaySegment_part1(fixed_imshow[0, 0, plt_slice, :, :], warped_label_integrated[0, 0, plt_slice, :, :]))
                            plt.subplot(155)
                            plt.imshow(overlaySegment_part1(fixed_imshow[0, 0, plt_slice, :, :], fixed_label[0, 0, plt_slice, :, :]))
                            plt.savefig('/share/data_rosita2/lucas/NOT_IN_BACKUP/tmp/exps/miccai_train' + id + '_thr' + str(thresh) + '_2.png', bbox_inches="tight")

                            # reverse
                            negate_displgrid = _negate(displ.grid)
                            negate_last_grid = _negate(last_grid)
                            warped_label_negated = label_warp(fixed_label.cuda(), id_field, negate_displgrid).cpu()
                            warped_label_integrated_negated = label_warp(fixed_label.cuda(), id_field, negate_last_grid).cpu()
                            warped_label_integrated_negated_time = label_warp(fixed_label.cuda(), id_field, _negate(displ(int(round(n_integration_steps*thresh)), n_integration_steps))).cpu()
                            warped_label_integrated_negated_time_resampled = label_warp(fixed_label.cuda(), id_field, _negate(displ(int(round(n_integration_steps*thresh)), n_integration_steps, 0.25))).cpu()

                            print('Dice reg  neg   :', float(dice_val(moving_label, warped_label_negated)))
                            print('Dice regI neg   :', float(dice_val(moving_label, warped_label_integrated_negated)))
                            print('Dice ' + str(thresh) + 'I neg FU:', float(dice_val(fuct_label, warped_label_integrated_negated_time)))
                            print('Dice ' + str(thresh) + 'I neg FU:', float(dice_val(fuct_label, warped_label_integrated_negated_time_resampled)), '(resampled)')

                            results[idx][thx].append(float(dice_val(moving_label, warped_label_negated)))
                            results[idx][thx].append(float(dice_val(moving_label, warped_label_integrated_negated)))
                            results[idx][thx].append(float(dice_val(fuct_label, warped_label_integrated_negated_time)))
                            results[idx][thx].append(float(dice_val(fuct_label, warped_label_integrated_negated_time_resampled)))

                            plt.figure(figsize=(30, 6))
                            plt.subplot(151)
                            plt.imshow(overlaySegment_part1(moving_imshow[0, 0, plt_slice, :, :], warped_label_negated[0, 0, plt_slice, :, :]))
                            plt.subplot(152)
                            plt.imshow(warped_label_negated[0, 0, plt_slice, :, :], cmap='gray', vmin=0, vmax=1)
                            plt.subplot(153)
                            plt.imshow(warped_label_integrated_negated[0, 0, plt_slice, :, :], cmap='gray', vmin=0, vmax=1)
                            plt.subplot(154)
                            plt.imshow(overlaySegment_part1(moving_imshow[0, 0, plt_slice, :, :],
                                                            warped_label_integrated_negated[0, 0, plt_slice, :, :]))
                            plt.subplot(155)
                            plt.imshow(overlaySegment_part1(moving_imshow[0, 0, plt_slice, :, :], moving_label[0, 0, plt_slice, :, :]))
                            plt.savefig('/share/data_rosita2/lucas/NOT_IN_BACKUP/tmp/exps/miccai_train' + id + '_thr' + str(thresh) + '_3.png', bbox_inches="tight")

                            plt.figure(figsize=(30, 6))
                            plt.subplot(151)
                            plt.title('FU 0.5 penu moving')
                            plt.imshow(warped_label_integrated_time[0, 0, plt_slice, :, :], cmap='gray', vmin=0, vmax=1)
                            plt.subplot(152)
                            plt.title('FU 0.5 penu moving (resampled)')
                            plt.imshow(warped_label_integrated_time_resampled[0, 0, plt_slice, :, :], cmap='gray', vmin=0, vmax=1)
                            plt.subplot(153)
                            plt.title('FU GT')
                            plt.imshow(fuct_label[0, 0, plt_slice, :, :], cmap='gray', vmin=0, vmax=1)
                            plt.subplot(154)
                            plt.title('FU 0.5 core moving (resampled)')
                            plt.imshow(warped_label_integrated_negated_time_resampled[0, 0, plt_slice, :, :], cmap='gray', vmin=0, vmax=1)
                            plt.subplot(155)
                            plt.title('FU 0.5 core moving')
                            plt.imshow(warped_label_integrated_negated_time[0, 0, plt_slice, :, :], cmap='gray', vmin=0, vmax=1)
                            plt.savefig('/share/data_rosita2/lucas/NOT_IN_BACKUP/tmp/exps/miccai_train' + id + '_thr' + str(thresh) + '_4.png', bbox_inches="tight")


                        dense_field = torch.nn.functional.interpolate(displ.grid.permute(0, 4, 1, 2, 3),
                                                                      size=(id_field.size(1), id_field.size(2), id_field.size(3)),
                                                                      mode='trilinear', align_corners=True).permute(0, 2, 3, 4, 1)

                        jac_det = jacobian_determinant_3d((id_field + dense_field).cpu().permute(0, 4, 1, 2, 3))

                        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

                        neg_frac = torch.mean((jac_det < 0).float())
                        print('displ.grid %0.3f' % (jac_det.std()), '%0.3f' % (neg_frac))

                        results[idx][thx].append(float(jac_det.std()))
                        results[idx][thx].append(float(neg_frac))

                        dense_field = torch.nn.functional.interpolate(displ(n_integration_steps, n_integration_steps).permute(0, 4, 1, 2, 3),
                                                                      size=(id_field.size(1), id_field.size(2), id_field.size(3)),
                                                                      mode='trilinear', align_corners=True).permute(0, 2, 3, 4, 1)

                        jac_det = jacobian_determinant_3d((id_field + dense_field).cpu().permute(0, 4, 1, 2, 3))

                        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

                        neg_frac = torch.mean((jac_det < 0).float())
                        print('integrated %0.3f' % (jac_det.std()), '%0.3f' % (neg_frac))

                        results[idx][thx].append(float(jac_det.std()))
                        results[idx][thx].append(float(neg_frac))


                        print('Time processed:', time.time() - start_time)

f = open('/share/data_rosita2/lucas/NOT_IN_BACKUP/tmp/exps/miccai_thr' + str(thresh) + '.txt', 'a')
f.write('n_integration_steps' + str(n_integration_steps) + '\n')
f.write('reg_iters' + str(reg_iters) + '\n')
f.write('alpha' + str(alpha) + '\n')
f.write('beta' + str(beta) + '\n')
f.write('id thresh DCinit DCreg DCregI DCfuI DCfuI(resampled) DCregNeg DCregNegI DCNegfuI DCNegfuI(resampled) gridJacDet gridNegFrac gridIJacDet gridINegFrac\n')
for thx, thresh in enumerate(threshs):
    for idx, id in enumerate(ids):
        print(id, thresh, ' '.join([str(item) for item in results[idx][thx]]))
        f.write(str(id) + ' ' + str(thresh) + ' ' + ' '.join([str(item) for item in results[idx][thx]]) + '\n')
f.close()