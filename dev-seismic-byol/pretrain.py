# main.py

import numpy as np
from pathlib import Path

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

    model_name = f'V{repetition}_pretrain_{dataset_name}_In{str(input_size[0])}_B{batch_size}_E{num_epochs}'
    logger.info(f'Model name: {model_name}')
 
 
    # Transforms
    random_flip = RandomFlip(possible_axis=[1])
    random_crop = RandomCrop(crop_size=input_size)
    random_rotation = RandomRotation(degrees=25, prob=0.2)
    transpose_to_HWC = Transpose([1, 2, 0])
    transpose_to_CHW = Transpose([2, 0, 1])    
    cast_to_tensor = CastTo(dtype=np.float32)
    repeat = Repeat(axis=2, n_repetitions=3)
    

    if dataset_name == 's0' or dataset_name == 'a700':
        byol_transform_pipeline = TransformPipeline([
            transpose_to_HWC,
            repeat,
            random_crop,
            random_flip,
            random_rotation,
            transpose_to_CHW,
            # cast_to_tensor,
        ])
        
    else: 
        byol_transform_pipeline = TransformPipeline([
            random_crop,
            random_flip,
            random_rotation,
            transpose_to_CHW,
            cast_to_tensor,
        ])
        

    constrastive_transform = ContrastiveTransform(byol_transform_pipeline)
    logger.info(f"Transforms built for {dataset_name}")
    
    # Dataset
    
    if dataset_name == 's0':
        train_img_reader = PartialPatchedZarrReader(
            path=data_path,
            data_shape=(1, 512, 512),
            stride=(1, 6625,  2001),
            pad_width=None,
            index_bounds=[(2000, 0, 0), (4000, 6625, 2001)],
            )
        
    elif dataset_name != 'a700':
        train_img_reader = TiffReader(path=data_path)
        
    logger.info(f"Readers built!")
    
    
    if dataset_name != 'a700':
    
        pretrain_dataset = SimpleDataset(
            readers=train_img_reader,
            transforms=constrastive_transform,
            return_single=True
        )
    
    if dataset_name == 'a700':
        data_module = A700DataModule(
            subset='both',
            normalization_strategy='z-sample',
            crop_size=0,
            batch_size=batch_size,
            transform=constrastive_transform,
            root=data_path
        )

    else:
        data_module = MinervaDataModule(
            train_dataset=pretrain_dataset,
            batch_size=batch_size,
            drop_last=True,
            shuffle_train=True,
            name=dataset_name
        )

    logger.info(f'DataModule assembled')

    # Modelo
    backbone = DeepLabV3Backbone(num_classes=6)
    model = BYOL(backbone=backbone, learning_rate=learning_rate)
    logger.info(f"Model built: {type(model).__name__}")
    # Logger, Checkpoints, Trainer
    log_dir = Path(log_path) / model_name / dataset_name
    ckpt_dir = Path(ckpt_path) / model_name / dataset_name
    CSVlogger = CSVLogger(log_dir, name=model_name, version=dataset_name)
    ckpt_callback = ModelCheckpoint(save_top_k=1, save_last=True, dirpath=ckpt_dir)
    logger.info("Loggers and checkpoints built")

    trainer = Trainer(
        accelerator='gpu',
        logger=CSVlogger,
        callbacks=[ckpt_callback],
        max_epochs=num_epochs,
        strategy='ddp_find_unused_parameters_true',
        devices=gpus
    )
    logger.info("Trainer instantiated")

    pipeline = SimpleLightningPipeline(
        model=model,
        trainer=trainer,
        log_dir=log_dir,
        save_run_status=True,
    )

    pipeline.run(data_module, task="fit")
