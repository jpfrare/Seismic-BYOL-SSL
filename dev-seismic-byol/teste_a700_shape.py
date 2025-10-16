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
        RandomCrop(crop_size=(256,256)),    # 3, crop_size, crop_size
        constrastive_transform,
    ]
)   
logger.info(f"Transforms built for a700")


data_module = A150DataModule(
    root=data_path,
    subset='both',
    batch_size=4,
    num_workers=os.cpu_count() if os.cpu_count() < 24 else 24 ,
    transforms=transform_pipeline,    
    drop_last=True,
)

data_module.setup(stage='train')
sample_batch = next(iter(data_module.val_dataloader()))
print("Sample batch:", sample_batch[0].shape)
import matplotlib.pyplot as plt

print("Type: ", type(sample_batch), len(sample_batch))

# Save an example from the dataset
example_image = sample_batch[0][0].permute(1, 2, 0).numpy()  # Convert CHW to HWC for visualization
plt.imshow(example_image, cmap='gray')
plt.axis('off')
plt.savefig('a700_example.png')
# print("Saved example image as a700_example.png")



backbone = DeepLabV3Backbone(num_classes=6)
model = BYOL(backbone=backbone, 
                learning_rate=1e-5,
                )

out = model.validation_step(sample_batch, batch_idx=0)
print(out)