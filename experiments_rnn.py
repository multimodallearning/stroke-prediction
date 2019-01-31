import os
import argparse
import datetime
from train_grunet import main


FILENAME = '/epoch{}.{}'
BATCHSIZE = 2
CLINICAL = 2  # only clinical time values
EPOCHS = 200
VALIDSIZE = 0.275
SEED = 4

DEFAULT_IDX = 0


lengths = [11,
           25]
commonfeatures = [5,
                  4]  # less parameters
additionals = [14,  # both
               -1,  # affine
               8,   # nonlin
               14]  # less parameters
img2vec1s = [[18, 19, 20, 21, 22],  # both
             [18, 19, 20, 21, 22],  # affine
             None,                  # nonlin
             [16, 17, 18, 19, 20]]  # less parameters
vec2vec1s = [[24, 20, 20, 24],  # both
             [24, 20, 20, 24],  # affine
             None,              # nonlin
             [22, 24]]          # less parameters
grunets = [[24, 28, 32, 28, 24],  # both
           None,                  # affine
           [18, 28, 32, 28, 24],  # nonlin
           [22, 24, 26, 20, 16]]  # less parameters
img2vec2s = [None]
vec2vec2s = [None]
addfactors = [False, True]
softeners = [[23, 23, 1],  # soften offsets NOT images; third channel of size 3 for x,y,z offset
             [31, 31, 1]]  # soften offsets NOT images; third channel of size 3 for x,y,z offset
losses = [[10, 44, 10, 25, 1],    # with monotone for seq_len=11
          [15, 45, 15, 25, 0],    # w/o  monotone for seq_len=11
          [22, 23, 22, 22, 1]]    # equally weighted for seq_len=11
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
else:
    raise Exception('No valid experiment id given')

os.mkdir(path)

print(datetime.datetime.now())
main(path+FILENAME, length, BATCHSIZE, CLINICAL, commonfeature, additional, img2vec1, vec2vec1, grunet, img2vec2, vec2vec2,
     addfactor, softener, loss, EPOCHS, fold, VALIDSIZE, SEED, combine)
print(datetime.datetime.now())