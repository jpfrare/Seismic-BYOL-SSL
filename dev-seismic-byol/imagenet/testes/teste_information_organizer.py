from base.InformationOrganizer import *
import argparse
import os
from pathlib import Path
from base.ImagenetReader import ImagenetReader, ImagenetValReader

#paths
DATASET_ROOT = "/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/train"
TRAIN_ENTRIES = "/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/extras_v3/entries-TRAIN.npy"
VAL_ROOT = "/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/val"
GT_ROOT = "/petrobr/parceirosbr/home/joao.frare/workspace/spfm/sharedata/datasets/ImageNet_2012/extra_files/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"
MAT_ROOT = "/petrobr/parceirosbr/home/joao.frare/workspace/spfm/sharedata/datasets/ImageNet_2012/extra_files/ILSVRC2012_devkit_t12/data/meta.mat"

finetune_organizer = FinetuningOrganizer('/petrobr/parceirosbr/home/joao.frare/workspace/spfm/Seismic-Byol/dev-seismic-byol/imagenet/logs+checkpoints')
finetune_organizer.set_readers(DATASET_ROOT, TRAIN_ENTRIES, VAL_ROOT, GT_ROOT, MAT_ROOT)
print(finetune_organizer)