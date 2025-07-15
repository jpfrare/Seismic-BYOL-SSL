# main.py

import numpy as np
from pathlib import Path
import os

from functions import *

from minerva.transforms.transform import *
from minerva.transforms.random_transform import *

from minerva.data.readers import TiffReader, PartialPatchedZarrReader, NumpyArrayReader
from minerva.data.datasets import SimpleDataset
from minerva.data.data_modules import MinervaDataModule

from minerva.models.ssl.byol import BYOL
from minerva.models.nets.image.deeplabv3 import DeepLabV3Backbone

from minerva.pipelines.lightning_pipeline import SimpleLightningPipeline
from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning import Trainer

from a700 import A700DataModule

input_size = (256, 256)



# Transforms
random_flip = RandomFlip(possible_axis=1, seed=1)
random_crop = RandomCrop(crop_size=input_size)
random_rotation = RandomRotation(degrees=25, prob=0.2)
transpose_to_HWC = Transpose([1, 2, 0])
transpose_to_CHW = Transpose([2, 0, 1])    
cast_to_tensor = CastTo(dtype=np.float32)
repeat = Repeat(axis=2, n_repetitions=3)

byol_transform_pipeline = TransformPipeline([
    transpose_to_HWC,
    repeat,
    random_crop,
    random_flip,
    random_rotation,
    transpose_to_CHW,
])

constrastive_transform = ContrastiveTransform(byol_transform_pipeline)

# Dataset
    
path = "/parceirosbr/asml/datasets/a700"
    
# entries = os.listdir(path)

# print(f"Total entries: {len(entries)}")
# for entry in entries:2
#     full_path = os.path.join(path, entry)
#     if os.path.isfile(full_path):
#         print(f"[FILE]   {entry}")
#     elif os.path.isdir(full_path):
#         print(f"[FOLDER] {entry}")
    
data_module = A700DataModule(
    subset='both',
    normalization_strategy='z-sample',
    crop_size=0,
    batch_size=8,
    transform=constrastive_transform,
    root=path
)

loader = data_module.train_dataloader()
data_iter = iter(loader)
item = next(data_iter)


print(item[0][0])
print(type(item[0]))
print(len(item))
print(item[0].shape)
print(type(item[0][0][0][0][0]))

# dataset = data_module.train_dataset.concatenation 

# data = transpose_to_HWC(dataset[0])
# flip = Flip(axis=1)

# data = flip(data)
# print(data)

# print(data.shape)




