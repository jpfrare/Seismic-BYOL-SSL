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
    print("🚀 Running in FAST TEST mode")
    args.repetition = 9
    args.per_class = 5          # 5 imagens por classe = 5.000 imagens total => com o batch size de 512 => 9 steps
    max_steps = 10              # Suficiente para ver o log e 2 validações
    devices = 1
    strategy = "auto"
    precision = 32              # 32-bit para evitar instabilidade em passos curtos
    
    # Parâmetros cruciais para não dar erro de "interval"
    log_every_n_steps = 5       # Loga rápido para você ver no terminal
    limit_val_batches = 10      # Não precisa validar as 50k imagens no teste!
else:
    max_steps = 217717
    devices = 1
    strategy = "auto"
    precision = "bf16-mixed"
    log_every_n_steps = 30
    limit_val_batches = 1.0

if args.per_class < 0 or args.per_class > 1200:
    raise KeyError("error, incorrect amount of samples")

seed_everything(args.repetition)

#--------------------------------------PATHS IMPORTANTES-----------------------------------------------
DATASET_ROOT = "/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/train"
TRAIN_ENTRIES = "/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/extras_v3/entries-TRAIN.npy"
VAL_ROOT = "/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/val"
VAL_ENTRIES = "/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/extras/entries-VAL.npy"
PRETRAIN_LOGS_PATH = f"checkpoints/logs_vinicius/pretrain/{args.repetition}"
PRETRAIN_CKPT_PATH = f"checkpoints/ckpt_vinicius/pretrain/{args.repetition}"
#--------------------------------------NOME DO MODELO-------------------------------------------------
model_name = f"V{args.repetition}_pretrain_imagenet_{args.per_class}_per_class"

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
    ImagenetReader= train_reader,
    transform= train_transform_pipeline,
)
train_subset = StratifiedSubset(train_dataset, args.per_class, args.repetition)

#a validação funciona da seguinte forma, a cada x steps o trainer pega um dado da validação e aplica no modelo para inferir se o aprendizado está fluindo

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
    ImagenetReader= val_reader,
    transform= val_transform_pipeline
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
    monitor='val_loss',                # monitorar a val_loss
    save_top_k=1,                      # Salva apenas o menor val_loss
    save_last=True,                    # Salva o estado final para resume
    dirpath=ckpt_dir,
    filename='best'                    
    auto_insert_metric_name=False
)

early_stop_callback = EarlyStopping(
    monitor= 'val_loss',                 #métrica vigiada
    min_delta = 0.001,                   #minima variação aceitável
    patience= 15,                        #quantas variações menores que min_delta consecutivas ele tolera antes de parar o treino
    verbose= True,
    mode= 'min'                         #mode min = minimização de perda
)

callbacks = [ckpt_callback]
if not args.teste:
    callbacks.append(early_stop_callback)

#-------------------------------------TRAINER E PIPELINE--------------------------------------


trainer = Trainer(
    accelerator='gpu',
    devices=devices,
    strategy=strategy,
    precision=precision,
    logger=CSVlogger,
    callbacks= callbacks,
    max_steps= max_steps,
    val_check_interval=1.0,                                             #vai validar depois de uma época
    limit_val_batches=limit_val_batches,                                #quantos batches serão usados na validação
    log_every_n_steps=log_every_n_steps,              
    benchmark=True,
    gradient_clip_val = 1.0                                             #não deixa o gradiente estourar
)

pipeline = SimpleLightningPipeline(
    model=model,
    trainer=trainer,
    log_dir=log_dir,
    save_run_status=True,
)

pipeline.run(data_module, task="fit")

