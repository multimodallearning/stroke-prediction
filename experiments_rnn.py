import os
import argparse
import datetime
from train_grunet import main


BATCHSIZE = 2
CLINICAL = 2  # only clinical time values
EPOCHS = 200
VALIDSIZE = 0.275
SEED = 4

DEFAULT_IDX = 0


lengths = [11,
           16]
commonfeatures = [5,
                  4,  # less parameters
                  2]  # less parameters #2
additionals = [10,  # both
               -1,  # affine
               4,   # nonlin
               10,  # less parameters
               12,  # both, with clinical time
               6,  # nonlin, with clinical time
               -1,  # affine, less parameters
               12,  # less parameters, with clinical time
               12]  # less parameters #2, with clinical time
img2vec1s = [[14, 19, 20, 21, 22],  # both
             [14, 19, 20, 21, 22],  # affine
             None,                  # nonlin
             [12, 17, 18, 19, 20],  # less parameters
             [14, 19, 20, 21, 22],  # both, with clinical time
             None,                  # nonlin, with clinical time
             [12, 17, 18, 19, 20],  # affine, less parameters
             [12, 17, 18, 19, 20],  # less parameters, with clinical time
             [8, 14, 16, 18, 20]]   # less parameters #2, with clinical time
vec2vec1s = [[24, 20, 20, 24],  # both
             [24, 20, 20, 24],  # affine
             None,              # nonlin
             [22, 24],          # less parameters
             [24, 20, 20, 24],  # both, with clinical time
             None,              # nonlin, with clinical time
             [22, 24],          # affine, less parameters
             [22, 24],          # less parameters, with clinical time
             [22, 24]]          # less parameters #2, with clinical time
grunets = [[20, 28, 32, 28, 24],  # both
           None,                  # affine
           [14, 28, 32, 28, 24],  # nonlin
           [18, 24, 26, 20, 16],  # less parameters
           [22, 28, 32, 28, 24],  # both, with clinical time
           [16, 28, 32, 28, 24],  # nonlin, with clinical time
           None,                  # affine, less parameters
           [20, 25, 26, 20, 16],  # less parameters, with clinical time
           [16, 20, 24, 20, 18]]  # less parameters #2, with clinical time
img2vec2s = [None]
vec2vec2s = [None]
addfactors = [False, True]
upsampledclinicals = [False, True]
softeners = [[5, 23, 23],  # soften offsets NOT images
             [7, 31, 31]]  # soften offsets NOT images
losses = [[10, 44, 10, 25, 0, 1, 0],    # with monotone for seq_len=11
          [15, 45, 15, 25, 0, 0, 0],    # w/o  monotone for seq_len=11
          [22, 23, 22, 22, 0, 1, 0],    # equally weighted for seq_len=11
          [11, 50, 11, 12, 8, 0.5, 0.0000001]]  # with monotone for seq_len=16, and middle overlap
folds = [[17, 6, 2, 26, 11, 4],
         [1, 21, 16, 27, 24, 18],
         [15, 20, 28, 14, 5, 13],
         [9, 22, 12, 0, 3, 8],
         [23, 25, 7, 10, 19]]
combines = ['add', 'linear', 'split']

parser = argparse.ArgumentParser()
parser.add_argument('id', type=int, help='ID of experiment', default=-1)
parser.add_argument('fold', type=int, help='fold id [0-4]', default=0)
args = parser.parse_args()

path = '/share/data_zoe1/lucas/NOT_IN_BACKUP/tmp/exp' + str(args.id)
fold = []
for i, f in enumerate(folds):
    if i != args.fold:
        fold += f

length = lengths[DEFAULT_IDX]
commonfeature = commonfeatures[DEFAULT_IDX]
additional = additionals[DEFAULT_IDX]
img2vec1 = img2vec1s[DEFAULT_IDX]
vec2vec1 = vec2vec1s[DEFAULT_IDX]
grunet = grunets[DEFAULT_IDX]
img2vec2 = img2vec2s[DEFAULT_IDX]
vec2vec2 = vec2vec2s[DEFAULT_IDX]
addfactor = addfactors[DEFAULT_IDX]
softener = softeners[DEFAULT_IDX]
combine = combines[DEFAULT_IDX]
loss = losses[DEFAULT_IDX]
upsampledclinical = upsampledclinicals[DEFAULT_IDX]


if args.id == 0:
    print(args.id, '/ RUN DEFAULTS')
elif args.id == 1:
    print(args.id, '/ RUN DEFAULTS --> affine')
    additional = additionals[1]
    img2vec1 = img2vec1s[1]
    vec2vec1 = vec2vec1s[1]
    grunet = grunets[1]
elif args.id == 2:
    print(args.id, '/ RUN DEFAULTS --> nonlin')
    additional = additionals[2]
    img2vec1 = img2vec1s[2]
    vec2vec1 = vec2vec1s[2]
    grunet = grunets[2]
elif args.id == 3:
    print(args.id, '/ RUN DEFAULTS --> w/o monotone loss')
    loss = losses[1]
elif args.id == 4:
    print(args.id, '/ RUN DEFAULTS --> combine linear')
    combine = combines[1]
elif args.id == 5:
    print(args.id, '/ RUN DEFAULTS --> combine split')
    combine = combines[2]
elif args.id == 6:
    print(args.id, '/ RUN DEFAULTS --> stronger regularized / less parameters')
    additional = additionals[3]
    img2vec1 = img2vec1s[3]
    vec2vec1 = vec2vec1s[3]
    grunet = grunets[3]
    softener = softeners[1]
    commonfeature = commonfeatures[1]
elif args.id == 7:
    print(args.id, '/ RUN DEFAULTS --> affine, combine linear')
    additional = additionals[1]
    img2vec1 = img2vec1s[1]
    vec2vec1 = vec2vec1s[1]
    grunet = grunets[1]
    combine = combines[1]
elif args.id == 8:
    print(args.id, '/ RUN DEFAULTS --> affine, combine split')
    additional = additionals[1]
    img2vec1 = img2vec1s[1]
    vec2vec1 = vec2vec1s[1]
    grunet = grunets[1]
    combine = combines[2]
elif args.id == 9:
    print(args.id, '/ RUN DEFAULTS --> nonlin, combine linear')
    additional = additionals[2]
    img2vec1 = img2vec1s[2]
    vec2vec1 = vec2vec1s[2]
    grunet = grunets[2]
    combine = combines[1]
elif args.id == 10:
    print(args.id, '/ RUN DEFAULTS --> nonlin, combine split')
    additional = additionals[2]
    img2vec1 = img2vec1s[2]
    vec2vec1 = vec2vec1s[2]
    grunet = grunets[2]
    combine = combines[2]
elif args.id == 11:
    print(args.id, '/ RUN DEFAULTS --> with upsampled clinical for GRUnet')
    upsampledclinical = upsampledclinicals[1]
    additional = additionals[4]
    img2vec1 = img2vec1s[4]
    vec2vec1 = vec2vec1s[4]
    grunet = grunets[4]
elif args.id == 13:
    print(args.id, '/ RUN DEFAULTS --> nonlin, with upsampled clinical for GRUnet')
    upsampledclinical = upsampledclinicals[1]
    additional = additionals[5]
    img2vec1 = img2vec1s[5]
    vec2vec1 = vec2vec1s[5]
    grunet = grunets[5]
elif args.id == 14:
    print(args.id, '/ RUN DEFAULTS --> with upsampled clinical for GRUnet, combine split')
    upsampledclinical = upsampledclinicals[1]
    additional = additionals[4]
    img2vec1 = img2vec1s[4]
    vec2vec1 = vec2vec1s[4]
    grunet = grunets[4]
    combine = combines[2]
elif args.id == 15:
    print(args.id, '/ RUN DEFAULTS --> nonlin, with upsampled clinical for GRUnet, combine split')
    upsampledclinical = upsampledclinicals[1]
    additional = additionals[5]
    img2vec1 = img2vec1s[5]
    vec2vec1 = vec2vec1s[5]
    grunet = grunets[5]
    combine = combines[2]
elif args.id == 16:
    print(args.id, '/ RUN DEFAULTS --> affine, stronger regularized / less parameters')
    additional = additionals[6]
    img2vec1 = img2vec1s[6]
    vec2vec1 = vec2vec1s[6]
    grunet = grunets[6]
    softener = softeners[1]
    commonfeature = commonfeatures[1]
elif args.id == 99:
    print(args.id, 'DEBUG / TESTING PURPOSES')
    upsampledclinical = upsampledclinicals[1]
    commonfeature = commonfeatures[2]
    additional = additionals[8]
    img2vec1 = img2vec1s[8]
    vec2vec1 = vec2vec1s[8]
    grunet = grunets[8]
    softener = softeners[1]
    combine = combines[2]
    loss = losses[3]
    length = lengths[1]
else:
    raise Exception('No valid experiment id given')

if not os.path.isdir(path):
    os.mkdir(path)
FILENAME = '/f' + str(args.fold + 1) + '_epoch_{}.{}'

print(datetime.datetime.now())
main(path+FILENAME, length, BATCHSIZE, CLINICAL, commonfeature, additional, img2vec1, vec2vec1, grunet, img2vec2, vec2vec2,
     addfactor, softener, loss, EPOCHS, fold, VALIDSIZE, SEED, combine, upsampledclinical)
print(datetime.datetime.now())