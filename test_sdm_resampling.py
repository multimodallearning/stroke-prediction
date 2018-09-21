#from common.model.Unet3D import Unet3D  TODO Unet live segmentation
#import common.dto.UnetDto as UnetDtoInit  TODO Unet live segmentation
from common import data, util, metrics
import torch
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
import scipy.ndimage.measurements as scim
import scipy.ndimage.morphology as scimorph
import datetime
import matplotlib.pyplot as plt
from torch.autograd import Variable


def sdm_interpolate_numpy(core, penu, interpolation, threshold=0.5, zoom=12, dilate=3, resample=True):
    penu_bin = penu[0, 0, :, :, :] > threshold
    penu_dist = ndi.distance_transform_edt(penu_bin)
    penu_dist -= ndi.distance_transform_edt(penu[0, 0, :, :, :] < threshold)
    latent_penu = ndi.zoom(penu_dist, (1, 1.0 / zoom, 1.0 / zoom))
    if not resample:
        recon_penu = penu_dist  # NO DOWNSAMPLING
    del penu_dist
    del penu

    core_bin = (core[0, 0, :, :, :] > threshold)
    if not core_bin.any():  # all signal below threshold, thus missing binary segmentation
        cog = [int(v) for v in scim.center_of_mass(penu_bin)]
        core_bin[cog[0], cog[1], cog[2]] = 1
        core_bin = scimorph.binary_dilation(core_bin, iterations=dilate)
        print('------------------------------------> artifical core', cog)
    del penu_bin
    core_dist = ndi.distance_transform_edt(1 - core_bin) - ndi.distance_transform_edt(
        core[0, 0, :, :, :] > threshold)
    del core_bin
    del core
    if not resample:
        recon_core = core_dist  # NO DOWNSAMPLING
    latent_core = ndi.zoom(core_dist, (1, 1.0 / zoom, 1.0 / zoom))
    del core_dist

    if resample:
        recon_core = ndi.zoom(latent_core, (1, zoom, zoom))[:, 2:130, 2:130]
        recon_penu = ndi.zoom(latent_penu, (1, zoom, zoom))[:, 2:130, 2:130]

    latent_intp = latent_penu * interpolation - latent_core * (1 - interpolation)

    if not resample:
        recon_intp = recon_penu * interpolation - recon_core * (1 - interpolation)
    else:
        recon_intp = ndi.zoom(latent_intp, (1, zoom, zoom))[:, 2:130, 2:130]

    return recon_core, recon_intp, recon_penu, latent_core, latent_intp, latent_penu


def get_normalized_time(batch, normalization_hours_penumbra):
    to_to_ta = batch[data.KEY_GLOBAL][:, 0, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5).type(torch.FloatTensor)
    normalization = torch.ones(to_to_ta.size()[0], 1).type(torch.FloatTensor) * \
                    normalization_hours_penumbra - to_to_ta.squeeze().unsqueeze(data.DIM_CHANNEL_TORCH3D_5)
    return to_to_ta, normalization


def infer():
    args = util.get_args_sdm()

    print('Evaluate validation set', args.fold)

    # Params / Config
    normalization_hours_penumbra = 10
    #channels_unet = args.channels  TODO Unet live segmentation
    #pad = args.padding  TODO Unet live segmentation

    transform = [data.ResamplePlaneXY(args.xyresample),
                 data.HemisphericFlipFixedToCaseId(split_id=args.hemisflipid),
                 #data.PadImages(pad[0], pad[1], pad[2], pad_value=0),  TODO Unet live segmentation
                 data.ToTensor()]

    ds_test = data.get_testdata_full(modalities=['_unet_core', '_unet_penu'],  # modalities=['_CBV_reg1_downsampled', '_TTD_reg1_downsampled'],  TODO Unet live segmentation
                                     labels=['_CBVmap_subset_reg1_downsampled', '_TTDmap_subset_reg1_downsampled',
                                        '_FUCT_MAP_T_Samplespace_subset_reg1_downsampled'],
                                     transform=transform,
                                     indices=args.fold)

    # Unet
    #unet = None  TODO Unet live segmentation
    #if not args.groundtruth:  TODO Unet live segmentation
    #    unet = Unet3D(channels=channels_unet)  TODO Unet live segmentation
    #    unet.load_state_dict(torch.load(args.unet))  TODO Unet live segmentation
    #    unet.train(False)  # fixate regularization for forward-only!  TODO Unet live segmentation

    for sample in ds_test:
        case_id = sample[data.KEY_CASE_ID].cpu().numpy()[0]

        nifph = nib.load('/share/data_zoe1/lucas/Linda_Segmentations/' + str(case_id) + '/train' + str(case_id) +
                         '_CBVmap_reg1_downsampled.nii.gz').affine

        to_to_ta, normalization = get_normalized_time(sample, normalization_hours_penumbra)

        lesion = Variable(sample[data.KEY_LABELS][:, 2, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5))
        if args.groundtruth:
            core = Variable(sample[data.KEY_LABELS][:, 0, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5))
            penu = Variable(sample[data.KEY_LABELS][:, 1, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5))
        else:
            #dto = UnetDtoInit.init_dto(Variable(sample[data.KEY_IMAGES]), None, None)  TODO Unet live segmentation
            #dto = unet(dto)  TODO Unet live segmentation
            #core = dto.outputs.core  TODO Unet live segmentation
            #penu = dto.outputs.penu,  TODO Unet live segmentation
            core = Variable(sample[data.KEY_IMAGES][:, 0, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5))
            penu = Variable(sample[data.KEY_IMAGES][:, 1, :, :, :].unsqueeze(data.DIM_CHANNEL_TORCH3D_5))

        ta_to_tr = sample[data.KEY_GLOBAL][:, 1, :, :, :].squeeze().unsqueeze(data.DIM_CHANNEL_TORCH3D_5)
        time_to_treatment = Variable(ta_to_tr.type(torch.FloatTensor) / normalization)

        del to_to_ta
        del normalization

        recon_core, recon_intp, recon_penu, latent_core, latent_intp, latent_penu = \
            sdm_interpolate_numpy(core.data.cpu().numpy(), penu.data.cpu().numpy(), threshold=0.5,
                                  interpolation=time_to_treatment.data.cpu().numpy().squeeze(), zoom=12,
                                  resample=args.downsample)

        print(int(sample[data.KEY_CASE_ID]), 'TO-->TR', float(time_to_treatment))

        if args.visualinspection:
            fig, axes = plt.subplots(3, 4)

            axes[0, 0].imshow(core.cpu().data.numpy()[0, 0, 16, :, :], cmap='gray', vmin=0, vmax=1)
            axes[1, 0].imshow(lesion.cpu().data.numpy()[0, 0, 16, :, :], cmap='gray', vmin=0, vmax=1)
            axes[2, 0].imshow(penu.cpu().data.numpy()[0, 0, 16, :, :], cmap='gray', vmin=0, vmax=1)

            axes[0, 1].imshow(latent_core[16, :, :], cmap='gray')
            axes[1, 1].imshow(latent_intp[16, :, :], cmap='gray')
            axes[2, 1].imshow(latent_penu[16, :, :], cmap='gray')

            axes[0, 2].imshow(recon_core[16, :, :], cmap='gray')
            axes[1, 2].imshow(recon_intp[16, :, :], cmap='gray')
            axes[2, 2].imshow(recon_penu[16, :, :], cmap='gray')

            axes[0, 3].imshow(recon_core[16, :, :] < 0, cmap='gray', vmin=0, vmax=1)
            axes[1, 3].imshow(recon_intp[16, :, :] > 0, cmap='gray', vmin=0, vmax=1)
            axes[2, 3].imshow(recon_penu[16, :, :] > 0, cmap='gray', vmin=0, vmax=1)
            plt.show()

        results = metrics.binary_measures_numpy((recon_intp > 0).astype(np.float),
                                                lesion.cpu().data.numpy()[0, 0, :, :, :], binary_threshold=0.5)

        c_res = metrics.binary_measures_numpy((recon_core < 0).astype(np.float),
                                               core.cpu().data.numpy()[0, 0, :, :, :], binary_threshold=0.5)

        p_res = metrics.binary_measures_numpy((recon_penu > 0).astype(np.float),
                                               penu.cpu().data.numpy()[0, 0, :, :, :], binary_threshold=0.5)

        with open('/data_zoe1/lucas/Linda_Segmentations/tmp/sdm_results.txt', 'a') as f:
            print('Evaluate case: {} - DC:{:.3}, HD:{:.3}, ASSD:{:.3}, Core recon DC:{:.3}, Penu recon DC:{:.3}'.format(case_id,
                results.dc, results.hd,  results.assd, c_res.dc, p_res.dc), file=f)

        zoomed = ndi.interpolation.zoom(recon_intp.transpose((2, 1, 0)), zoom=(2, 2, 1))
        nib.save(nib.Nifti1Image((zoomed > 0).astype(np.float32), nifph), args.outbasepath + '_' + str(case_id) + '_lesion.nii.gz')
        del zoomed

        zoomed = ndi.interpolation.zoom(lesion.cpu().data.numpy().astype(np.int8).transpose((4, 3, 2, 1, 0))[:, :, :, 0, 0], zoom=(2, 2, 1))
        nib.save(nib.Nifti1Image(zoomed, nifph), args.outbasepath + '_' + str(case_id) + '_fuctgt.nii.gz')
        del zoomed

        zoomed = ndi.interpolation.zoom(recon_core.transpose((2, 1, 0)), zoom=(2, 2, 1))
        nib.save(nib.Nifti1Image((zoomed < 0).astype(np.float32), nifph), args.outbasepath + '_' + str(case_id) + '_core.nii.gz')
        del zoomed

        zoomed = ndi.interpolation.zoom(recon_penu.transpose((2, 1, 0)), zoom=(2, 2, 1))
        nib.save(nib.Nifti1Image((zoomed > 0).astype(np.float32), nifph), args.outbasepath + '_' + str(case_id) + '_penu.nii.gz')

        del nifph

        del sample


if __name__ == '__main__':
    print(datetime.datetime.now())
    infer()
    print(datetime.datetime.now())
