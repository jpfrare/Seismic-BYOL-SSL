# -------------------- Python base --------------------
import os
import argparse
from pathlib import Path

# -------------------- PyTorch --------------------
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

# -------------------- Torchvision --------------------
from torchvision.models import resnet50
from torchvision import transforms

# -------------------- Lightning --------------------
import lightning as L
from lightning import Trainer
from lightning.fabric import seed_everything

# Logger e checkpoint
from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

# -------------------- Minerva --------------------
from minerva.models.nets import SimpleSupervisedModel
from minerva.pipelines.lightning_pipeline import SimpleLightningPipeline
from minerva.data.data_modules import MinervaDataModule

#---------------------Criados-----------------------
from imagenet.ImagenetDataset import ImagenetDataset, StratifiedSubset
from imagenet.ImagenetReader import ImagenetReader, ImagenetValReader
from imagenet.ImagenetModel import ImagenetModel

#ARGUMENTOS DO TREINO
parser = argparse.ArgumentParser(
    description= "Initializing Supervised Train..."
)

parser.add_argument(
    "--per_class", type= int, default= 1000, help= "how many images per class are we going to use?"
)

parser.add_argument(
    "--repetition", type= int, default= 9, help= "repetition"
)

parser.add_argument(
    "--teste", action= 'store_true'
)

args = parser.parse_args()

if args.teste:
    args.repetition = 9
    print("Running in TEST mode")
    args.per_class = 10
    max_steps = 10000
    devices = 1
    strategy = "auto"
    precision = 32
else:
    max_steps = 217717
    devices = 1
    strategy = "auto"
    precision = "16-mixed"

if args.per_class < 0 or args.per_class > 1300000:
    raise KeyError("error, incorrect amount of samples")

seed_everything(args.repetition)

#--------------------------------------PATHS IMPORTANTES-----------------------------------------------
DATASET_ROOT = "/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/train"
TRAIN_ENTRIES = "/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/extras_v3/entries-TRAIN.npy""
VAL_ROOT = "/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/val"
VAL_ENTRIES = "/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/extras/entries-VAL.npy"
PRETRAIN_LOGS_PATH = f"checkpoints/logs_vinicius/pretrain/{args.repetition}"
PRETRAIN_CKPT_PATH = f"checkpoints/ckpt_vinicius/pretrain/{args.repetition}"
#--------------------------------------NOME DO MODELO-------------------------------------------------
model_name = f"V{args.repetition}_pretrain_imagenet_{args.per_class}"

#--------------------------------------DADOS----------------------------------------------------------
train_transform_pipeline = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
train_reader = ImagenetReader(DATASET_ROOT, TRAIN_ENTRIES)
train_dataset = ImagenetDataset(
    readers= train_reader,
    transforms= train_transform_pipeline,
)
train_subset = StratifiedSubset(train_dataset, args.per_class, args.repetition)



val_transform_pipeline = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])
val_reader = ImagenetValReader(VAL_ROOT, VAL_ENTRIES)
val_dataset = ImagenetDataset(
    readers= val_reader,
    transforms= val_transform_pipeline
)

data_module = MinervaDataModule(
            train_dataset=train_subset,
            val_dataset= val_dataset,
            batch_size=512,
            drop_last=True,
            shuffle_train=True,
            name="imagenet",
            num_workers=min(os.cpu_count(), 24)
        )

#------------------------------------MODELO-----------------------------------------------------------
backbone = resnet50(weights=None)
backbone.fc = nn.Identity()
fc = nn.Linear(2048, 1000)
loss_fn = nn.CrossEntropyLoss()

model = ImagenetModel(
    backbone=backbone,
    fc=fc,
    loss_fn=loss_fn,
    optimizer=torch.optim.SGD,
    optimizer_kwargs={
        "lr": 0.2,
        "momentum": 0.9,
        "weight_decay": 1e-4
    },
    lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
    lr_scheduler_kwargs={
        "mode": "min",
        "factor": 0.1,
        "patience": 5,
        "verbose": True
    }
)
#-----------------------------------DIRETORIOS, LOGGERS E CALLBACKS----------------------------------------
log_dir = Path(PRETRAIN_LOGS_PATH)/model_name/"imagenet"
ckpt_dir = Path(PRETRAIN_CKPT_PATH)/model_name/"imagenet"
CSVlogger = CSVLogger(log_dir, name=model_name, version="imagenet")

ckpt_callback = ModelCheckpoint(
    save_top_k=1,                      #salvar o melhor checkpoint
    save_last=True,                     #salvar o último checkpoint
    dirpath=ckpt_dir,
    )

early_stop_callback = EarlyStopping(
    monitor= 'val_loss',                 #métrica vigiada
    min_delta = 0.001,                   #minima variação aceitável
    patience= 8,                         #quantas variações menores que min ele tolera antes de parar o treino
    verbose= True,
    mode= 'min'                         #mode min = minimização de perda
)

#-------------------------------------TRAINER E PIPELINE--------------------------------------


trainer = Trainer(
    accelerator='gpu',
    devices=devices,
    strategy=strategy,
    precision=precision,
    logger=CSVlogger,
    callbacks=[ckpt_callback, early_stop_callback],
    max_steps= max_steps,
    log_every_n_steps=100,
    val_check_interval=1000, # Valida a cada 1000 iterações
    benchmark=True           # Ganho de performance na H100
)

pipeline = SimpleLightningPipeline(
    model=model,
    trainer=trainer,
    log_dir=log_dir,
    save_run_status=True,
)

pipeline.run(data_module, task="fit")

