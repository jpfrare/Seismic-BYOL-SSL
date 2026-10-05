import numpy as np
from scipy.io import loadmat
from pathlib import Path
from base.ImagenetReader import *

'''aparentemente entries_train[i][2] é o wnid da imagem, entries_train[i][1] é o indice desse mesmo wnid em class_ids'''
train_entries = Path('/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/extras/entries-TRAIN.npy')
val_entries = Path('/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/extras/entries-VAL.npy')


entries_train = np.load(train_entries)
entries_val = np.load(val_entries)

print(entries_train.shape)
train_image = entries_train[10]
print(train_image)

print(entries_val.shape)
val_image = entries_val[10]
print(val_image)



