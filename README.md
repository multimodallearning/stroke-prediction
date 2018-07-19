# stroke-prediction
Stroke infarct growth prediction

## Objective
Learning to Predict Stroke Outcome based on Multivariate CT Images

## Data
The source code is working from within the IMI network at University of Luebeck, as the closed dataset of 29 subjects is only accessable if you are member of the bvstaff group. The filenames have been renamed and cases are represented as a subfolder. CTP modalities CBV and TTD are used as input, corresponding manual segmentations for core and penumbra, as well as follow-up lesion segmentation (FUCTMap). The directory contains more files since the work for the Master's thesis of Linda Aulmann.

## Setup
Set up a Python 3.5 environment including the packages of [requirements.txt](requirements.txt) file.

## Structure of repository
The repository consists of two subfolder:
- experiment: contains the Learner and DTO files for running the different trainings
- model: contains the models to be learned

Further, there are the following files:
- data.py: defines the dataset as mentioned under section [Data](README.md#data) and contains required transformations
- util.py: contains helper functions

## Usage
Activate the above environment under section [Setup](README.md#setup).

For learning the shape space on the manual segmentations run the following command:

`train_shape_prediction.py /share/data_zoe1/lucas/Linda_Segmentations/tmp/tmp_shape_f3.model --epochs 200 --outbasepath /share/data_zoe1/lucas/Linda_Segmentations/tmp/tmp_shape --channelscae 1 16 24 32 100 200 0 1 --channelsenc 1 16 24 32 100 200 0 1 --validsetsize 0.275 --fold 17 6 2 26 11 4 1 21 16 27 24 18 9 22 12 0 3 8 23 25 7 10 19`

The `--fold` is an arbitrary but fixed list of indices between 0 and 28 to specify a fold out of the 29 dataset subjects, from which a fraction specified by `--validsetsize` will be used as validation data (e.g. for 0.275 and the above fold it means that 17 training and 6 validation cases are used by the Learner).

Always specify a `--outbasepath` to where files are being saved. This includes the `*.model` files when a new validation minimum has been achieved, and `*.png` files that plot the losses, metrics and visuale some samples during the training run.

Train a Unet with the same fold as specified before, to use the Unet segmentation for further training of an adapted encoder to predict on segmentations of unseen CTP modalities:

`train_unet_segmentation.py /share/data_zoe1/lucas/Linda_Segmentations/tmp/tmp_unet_f3.model --epochs 200 --outbasepath /share/data_zoe1/lucas/Linda_Segmentations/tmp/tmp --channels 2 16 32 64 32 16 32 2 --validsetsize 0.275 --fold 17 6 2 26 11 4 1 21 16 27 24 18 9 22 12 0 3 8 23 25 7 10 19`

For comparison, you can run a shape interpolation via signed distance maps:

`sdm_resampling.py /share/data_zoe1/lucas/Linda_Segmentations/tmp/tmp_unet_f3.model --testcaseid 22 --downsample 0 --groundtruth 1`

