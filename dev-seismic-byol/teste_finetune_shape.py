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

from namss import *



finetune_data = "seam_ai_N"

dataset_mapping = get_dataset_mapping()
data_path = dataset_mapping[finetune_data]
print(data_path)

if finetune_data == "f3" or finetune_data == "f3_N":
    logger.info("Using padding of (256,704)")
    padding = Padding(256, 704)
elif finetune_data == "seam_ai" or finetune_data == "seam_ai_N":
    logger.info("Using padding of (1008,592)")
    padding = Padding(1008, 592)
    
transform_pipeline = TransformPipeline(
    [
        padding,
        Transpose([2, 0, 1]),   # C, H, W
        CastTo(dtype=np.float32),
    ]
)

cap = 1.0

if cap == 1.0 and type(cap) == float:
    
    logger.info("Using 100% of train data")
    train_dataset = SeismicFullDataset(root=data_path, partition='train', transform=transform_pipeline)
    data_module = SeismicDataModule(
        root = data_path,
        batch_size=4,
        cap=cap,
        drop_last=True,
        transform=transform_pipeline,
        test_transform=transform_pipeline,
        train_dataset = train_dataset,
        val_dataset = None,
        test_dataset = None,
    )
    
else:
    logger.info(f'Using {cap} samples of train data')
    data_module = SeismicDataModule(
        root=data_path,
        batch_size=4,
        cap=cap,
        drop_last=False,
        transform=transform_pipeline,
        test_transform=transform_pipeline,
        train_dataset = None,
        val_dataset = None,
        test_dataset = None,
    )


data_module.setup(stage='train')
sample_batch = next(iter(data_module.train_dataloader()))
print("Sample batch:", sample_batch[0].shape)
print("Type: ", type(sample_batch))

# Save an example from the dataset
example_image = sample_batch[0][1].permute(1, 2, 0).numpy()  # Convert CHW to HWC for visualization
plt.imshow(example_image, cmap='gray')
plt.axis('off')
plt.savefig('finetune_example.png')
print("Saved example image as finetune_example.png")



