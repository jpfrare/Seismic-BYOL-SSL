# main.py

import numpy as np
from pathlib import Path

from functions import *

from minerva.transforms.transform import *
from minerva.transforms.random_transform import *

from minerva.data.readers import (
    TiffReader,
    LazyPaddedPatchedZarrReader,
    NumpyArrayReader,
)
from minerva.data.datasets import SimpleDataset
from minerva.data.data_modules import MinervaDataModule

from minerva.models.ssl.byol import BYOL
from minerva.models.nets.image.deeplabv3 import DeepLabV3Backbone

from minerva.pipelines.lightning_pipeline import SimpleLightningPipeline
from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning import Trainer


def main(
    input_size,
    dataset_name,
    batch_size,
    num_epochs,
    repetition,
    learning_rate,
    data_path,
    ckpt_path,
    log_path,
    gpus,
):

    model_name = f"V{repetition}_pretrain_{dataset_name}_In{str(input_size[0])}_B{batch_size}_E{num_epochs}"

    # Transforms
    random_flip = RandomFlip(possible_axis=1)
    random_crop = RandomCrop(crop_size=input_size)
    random_rotation = RandomRotation(degrees=25, prob=0.2)
    transpose = Transpose([2, 0, 1])
    cast_to_tensor = CastTo(dtype=np.float32)

    byol_transform_pipeline = TransformPipeline(
        [
            random_crop,
            random_flip,
            random_rotation,
            transpose,
            cast_to_tensor,
        ]
    )

    constrastive_transform = ContrastiveTransform(byol_transform_pipeline)

    # Dataset

    if dataset_name == "s0":
        train_img_reader = LazyPaddedPatchedZarrReader(
            path=data_path,
            data_shape=(1, 6625, 2001),
            stride=None,
            pad_width=None,
        )

    elif dataset_name == "a700":
        train_img_reader = NumpyArrayReader(path=data_path)
    else:
        train_img_reader = TiffReader(path=data_path)

    pretrain_dataset = SimpleDataset(
        readers=train_img_reader, transforms=constrastive_transform, return_single=True
    )

    data_module = MinervaDataModule(
        train_dataset=pretrain_dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle_train=True,
        name=dataset_name,
    )

    # Modelo
    backbone = DeepLabV3Backbone(num_classes=6)
    model = BYOL(backbone=backbone, learning_rate=learning_rate)

    # Logger, Checkpoints, Trainer
    log_dir = Path(log_path) / model_name / dataset_name
    ckpt_dir = Path(ckpt_path) / model_name / dataset_name
    logger = CSVLogger(log_dir, name=model_name, version=dataset_name)
    ckpt_callback = ModelCheckpoint(save_top_k=1, save_last=True, dirpath=ckpt_dir)

    trainer = Trainer(
        accelerator="gpu",
        logger=logger,
        callbacks=[ckpt_callback],
        max_epochs=num_epochs,
        strategy="auto",
        devices=gpus,
    )

    pipeline = SimpleLightningPipeline(
        model=model,
        trainer=trainer,
        log_dir=log_dir,
        save_run_status=True,
    )

    pipeline.run(data_module, task="fit")
