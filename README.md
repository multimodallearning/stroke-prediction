# stroke-prediction
Stroke growth prediction

... to be written...

train_shape_prediction.py /share/data_zoe1/lucas/Linda_Segmentations/tmp/tmp_shape_f3.model --epochs 200 --outbasepath /share/data_zoe1/lucas/Linda_Segmentations/tmp/tmp_shape --channelscae 1 16 24 32 100 200 0 1 --channelsenc 1 16 24 32 100 200 0 1 --validsetsize 0.275 --fold 17 6 2 26 11 4 1 21 16 27 24 18 9 22 12 0 3 8 23 25 7 10 19

train_unet_segmentation.py /share/data_zoe1/lucas/Linda_Segmentations/tmp/tmp_unet_f3.model --epochs 200 --outbasepath /share/data_zoe1/lucas/Linda_Segmentations/tmp/tmp --channels 2 16 32 64 32 16 32 2 --validsetsize 0.275 --fold 17 6 2 26 11 4 1 21 16 27 24 18 9 22 12 0 3 8 23 25 7 10 19

sdm_resampling.py /share/data_zoe1/lucas/Linda_Segmentations/tmp/tmp_unet_f3.model --testcaseid 22 --downsample 0 --groundtruth 1

