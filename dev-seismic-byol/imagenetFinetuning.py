# -------------------- Python base --------------------
import os
import argparse
from pathlib import Path
from typing import Optional, Union, Literal

# -------------------- PyTorch --------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
import torchmetrics
from torchmetrics import Accuracy, JaccardIndex, F1Score

# -------------------- Torchvision --------------------
import torchvision.models
from torchvision.models import resnet50
from torchvision.models.segmentation import deeplabv3_resnet50

# -------------------- Lightning --------------------
import lightning as L
from lightning import Trainer
from lightning.fabric import seed_everything
from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

# -------------------- Minerva --------------------
from minerva.models.nets import SimpleSupervisedModel
from minerva.models.nets.image.deeplabv3 import DeepLabV3Backbone, DeepLabV3
from minerva.models.loaders import FromPretrained
from minerva.pipelines.lightning_pipeline import SimpleLightningPipeline

# Data & Transforms (Padding incluído aqui)
from minerva.data.readers import TiffReader, PNGReader
from minerva.data.datasets import SimpleDataset
from minerva.transforms.transform import TransformPipeline, Transpose, Padding
from minerva.transforms.random_transform import RandomCrop

#--------------------- Locais & Custom -----------------------
from functions import *  

from imagenet.ImagenetDataset import ImagenetDataset, StratifiedSubset
from imagenet.ImagenetReader import ImagenetReader, ImagenetValReader
from imagenet.ImagenetModel import ImagenetModel

parser = argparse.ArgumentParser(
    description= "Finetuning on Imagenet"
)

parser.add_argument(
    "--per_class", type= int, help= "per_clas model used in pretrain"
)

parser.add_argument(
    "--repetition", type= int, help= "repetition used"
)

parser.add_argument(
    "--finetune_dataset", type= str, help= "dataset used in finetuning (seam_ai_N or f3_N)"
)

args = parser.parse_args()
seed_everything(args.repetition)

#------------------------------------PATHS---------------------------
root = '/petrobr/parceirosbr/home/joao.frare/workspace/spfm/Seismic-Byol/dev-seismic-byol/'

model_name = f"V{args.repetition}_pretrain_imagenet_{args.per_class}_per_class"
mapping = get_dataset_mapping()
pretrain_ckpt_path = f"{root}/checkpoints/ckpt_vinicius/pretrain/{args.repetition}/{model_name}/imagenet"
train_ckpt_path= f"{root}/checkpoints/ckpt_vinicius/train_patch/{args.repetition}/{model_name}/finetune_{args.finetune_dataset}"
log_path = f"{root}/checkpoints/logs_vinicius/train_patch/{args.repetition}/{model_name}/finetune_{args.finetune_dataset}"

#----------------------------------MODELO - Transfer Learning---------------------------
num_classes = 6
deeplab_backbone = DeepLabV3Backbone(num_classes=num_classes)

#importing_pretrained_model
resnet50_backbone = resnet50(replace_stride_with_dilation=[False, True, True], weights= None)
resnet50_backbone.fc = nn.Identity()
fc = nn.Linear(2048, 1000)
pretrained_model = ImagenetModel(backbone= resnet50_backbone, fc= fc)

weighted_backbone = FromPretrained(model= pretrained_model, ckpt_path= f'{pretrain_ckpt_path}/best.ckpt', strict= False, error_on_missing_keys= False).backbone
weighted_state_dict= get_state_dict(weighted_backbone)
deeplab_backbone.load_state_dict(weighted_state_dict, strict= False)

model = DeepLabV3(
    backbone=deeplab_backbone,
    learning_rate=1e-5,
    num_classes=num_classes,
    freeze_backbone=False,
)

#----------------------------Dados - Modelagem----------------------------------------
dataset_path = mapping[args.finetune_dataset]

if args.finetune_dataset == 'seam_ai_N':
    print("Using padding of (256,704)")
    padding = Padding(256, 704)
elif args.finetune_dataset == 'f3_N':
    print("Using padding of (1008,592)")
    padding = Padding(1008, 592)

transform_pipeline = TransformPipeline([
    padding,
    Transpose([2,0,1])
])

train_dataset = SeismicFullDataset(root=dataset_path, partition='train', transform=transform_pipeline)
data_module = SeismicDataModule(
    root = dataset_path,
    batch_size=8,
    cap=1.0,
    drop_last=True,
    transform=transform_pipeline,
    test_transform=transform_pipeline,
    train_dataset = train_dataset,
    val_dataset = None,
    test_dataset = None,
    )

csv_logger = CSVLogger(log_path, name=model_name, version= args.finetune_dataset)
#------------------------TRAINER----------------------------------------------------------
trainer = Trainer(
    logger= csv_logger,
    max_epochs= 50,
    limit_val_batches = 1.0,
    strategy= 'auto',
    devices= 1,
    check_val_every_n_epoch=True
)

pipeline = SimpleLightningPipeline(
    model=model,
    trainer=trainer,
    log_dir=log_path,
    save_run_status=True,
)

pipeline.run(data_module, task="fit")
    
num_classes = 6
    
metrics = {
    "mIoU": JaccardIndex(
        num_classes=num_classes, average="macro", task="multiclass"
    ),
    "acc": Accuracy(num_classes=num_classes, task="multiclass"),
    "f1-weighted": F1Score(
        num_classes=num_classes, task="multiclass", average="weighted"
    ),
}
    
pipeline = SimpleLightningPipeline(
    model=model,
    trainer=trainer,
    log_dir=log_path,
    save_run_status=True,
    seed=args.repetition,
    apply_metrics_per_sample=False,
    classification_metrics=metrics,
)
    
pipeline.run(data_module, task="evaluate")