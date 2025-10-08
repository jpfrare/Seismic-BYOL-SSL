# main.py
import os
import numpy as np
from pathlib import Path

from functions import *

from minerva.transforms.transform import *
from minerva.transforms.random_transform import RandomFlip, RandomCrop, RandomRotation
from torchvision.models.segmentation import deeplabv3_resnet50
from lightning.pytorch.strategies import DDPStrategy

from minerva.data.readers import TiffReader
from minerva.data.datasets import SimpleDataset
from minerva.data.data_modules import MinervaDataModule

from minerva.models.ssl.byol import BYOL
from minerva.models.nets.image.deeplabv3 import DeepLabV3Backbone

from minerva.pipelines.lightning_pipeline import SimpleLightningPipeline
from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning import Trainer
from lightning.fabric import seed_everything

from a700 import *



test_dataset = "a700"

dataset_mapping = get_dataset_mapping()
data_path = dataset_mapping[test_dataset]
print(data_path)

random_flip = RandomFlip(possible_axis=[0, 1])
random_crop = RandomCrop(crop_size=(256,256))
random_rotation = RandomRotation(degrees=25, prob=0.2)
transpose_to_HWC = Transpose([1, 2, 0])
transpose_to_CHW = Transpose([2, 0, 1])    
cast_to_tensor = CastTo(dtype=np.float32)
repeat = Repeat(axis=0, n_repetitions=3)
unsqueeze = Unsqueeze(axis=0)



aux_transform_pipeline = TransformPipeline(
    [
        RandomFlip(possible_axis=[0, 1]),   # 513, 513
        RandomRotation(degrees=25, prob=0.2),
        Unsqueeze(axis=0),  # 1, 513, 513
        Repeat(axis=0, n_repetitions=3),    # 3, 513, 513
        CastTo(dtype=np.float32),
    ]
)

constrastive_transform = ContrastiveTransform(
    aux_transform_pipeline
)

transform_pipeline = TransformPipeline(
    [
        RandomCrop(crop_size=input_size),    # 3, crop_size, crop_size
        constrastive_transform,
    ]
) 


data_module = A150DataModule(
    root=data_path,
    subset='both',
    batch_size=4,
    num_workers=os.cpu_count() if os.cpu_count() < 24 else 24 ,
    transforms=transform_pipeline,    
    drop_last=True,
)

data_module.setup(stage='train')
sample_batch = next(iter(data_module.train_dataloader()))
print("Sample batch:", sample_batch[0].shape)
print("Type: ", type(sample_batch))



