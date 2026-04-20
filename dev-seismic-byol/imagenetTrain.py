# -------------------- Python base --------------------
import os
import argparse
from pathlib import Path

# -------------------- PyTorch --------------------
import torch
import torch.nn as nn

# -------------------- Torchvision --------------------
from torchvision.models import resnet50
from torchvision import transforms

# -------------------- Lightning --------------------
import lightning as L
from lightning import Trainer
from lightning.fabric import seed_everything

# Logger e checkpoint
from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint

# -------------------- Minerva --------------------
from minerva.models.nets import SimpleSupervisedModel
from minerva.pipelines.lightning_pipeline import SimpleLightningPipeline
from minerva.data.data_modules import MinervaDataModule
from minerva.data.datasets import *
from minerva.data.readers import imagenetReader

#ARGUMENTOS DO TREINO
parser = argparse.ArgumentParser(
    description= "Initializing Supervised Train..."
)

parser.add_argument(
    "--number_samples", type= int, default= 1000, help= "how many samples we will use in train"
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
    args.number_samples = 500
    max_epochs = 1
    devices = 1
    strategy = "auto"
    precision = 32
else:
    max_epochs = 100
    devices = 1
    strategy = "auto"
    precision = "16-mixed"

if args.number_samples > 1000000:
    raise KeyError("error, very large number of samples")

seed_everything(args.repetition)

#--------------------------------------PATHS IMPORTANTES-----------------------------------------------
DATASET_ROOT = "/petrobr/parceirosbr/home/vinicius.soares/workspace/spfm/datasets/ImageNet_2012/train"
PRETRAIN_LOGS_PATH = f"checkpoints/logs_vinicius/pretrain/{args.repetition}"
PRETRAIN_CKPT_PATH = f"checkpoints/ckpt_vinicius/pretrain/{args.repetition}"
#--------------------------------------NOME DO MODELO-------------------------------------------------
model_name = f"V{args.repetition}_pretrain_imagenet_{args.number_samples/1000}k"

#--------------------------------------DADOS----------------------------------------------------------
transform_pipeline = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

reader = imagenetReader(DATASET_ROOT)
dataset = SimpleDataset(
    readers= reader,
    transforms= transform_pipeline,
    return_single = False
)
subset = StratifiedSubset(dataset, args.number_samples, args.repetition)

data_module = MinervaDataModule(
            train_dataset=subset,
            batch_size=128,
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

model = SimpleSupervisedModel(
    backbone=backbone,
    fc=fc,
    loss_fn=loss_fn,
    optimizer=torch.optim.SGD,
    optimizer_kwargs={
        "lr": 0.05,
        "momentum": 0.9,
        "weight_decay": 1e-4
    },
    lr_scheduler=torch.optim.lr_scheduler.StepLR,
    lr_scheduler_kwargs={
        "step_size": 30,
        "gamma": 0.1
    }
)
#-----------------------------------DIRETORIOS E LOGGERS----------------------------------------
log_dir = Path(PRETRAIN_LOGS_PATH)/model_name/"imagenet"
ckpt_dir = Path(PRETRAIN_CKPT_PATH)/model_name/"imagenet"
CSVlogger = CSVLogger(log_dir, name=model_name, version="imagenet")

ckpt_callback = ModelCheckpoint(
    save_top_k=1, 
    save_last=True, 
    dirpath=ckpt_dir,
    )

#-------------------------------------TRAINER E PIPELINE--------------------------------------


trainer = Trainer(
    accelerator='gpu',
    devices=devices,
    strategy=strategy,
    precision=precision,
    logger=CSVlogger,
    callbacks=[ckpt_callback],
    max_epochs=max_epochs,
    log_every_n_steps=30,
)

pipeline = SimpleLightningPipeline(
    model=model,
    trainer=trainer,
    log_dir=log_dir,
    save_run_status=True,
)

pipeline.run(data_module, task="fit")

