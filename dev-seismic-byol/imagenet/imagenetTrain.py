import argparse 
import os
from pathlib import Path

import torch.nn.functional as F
import torchmetrics

# -------------------- Timm (Otimização e Augmentation) --------------------
import timm
import timm.optim
import timm.scheduler
from timm.data import create_transform

# -------------------- Lightning --------------------
import lightning as L
from lightning import Trainer
from lightning.fabric import seed_everything
from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

# -------------------- Minerva & Custom Modules --------------------
from minerva.pipelines.lightning_pipeline import SimpleLightningPipeline
from minerva.data.data_modules import MinervaDataModule

from base.ImagenetDataset import ImagenetDataset, StratifiedSubset
from base.ImagenetReader import ImagenetReader, ImagenetValReader
from base.ImagenetModel import ImagenetModel

#---------------------------------ARGUMENTOS DO TREINO------------------------
parser = argparse.ArgumentParser(
    description= "Initializing Supervised PreTrain..."
)
parser.add_argument(
    "--per_class", type= int, default= 1000, help= "how many images per class are we going to use?"
)
parser.add_argument(
    "--repetition", type= int, default= 9, help= "repetition"
)
parser.add_argument(
    "--num_classes", type= int, default= 1000, help= "how many classes are we going to use?"
)
#------------------------------------------------------------------------------
args = parser.parse_args()
batch_size = 1024
train_imagenet_size = 1281167

max_steps = train_imagenet_size*150//batch_size + 30          #número de passos para se treinar uma imagenet completa por 200 épocas
devices = 2
strategy = "ddp"
precision = "bf16-mixed"
limit_val_batches = 1.0
log_every_n_steps = 100

if args.per_class < 0:
    raise KeyError("error, incorrect amount of samples")
elif args.per_class > 1200:
    print("using full imagenet dataset!")

seed_everything(args.repetition)

#--------------------------------------PATHS IMPORTANTES-----------------------------------------------
DATASET_ROOT = "/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/train"
TRAIN_ENTRIES = "/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/extras_v3/entries-TRAIN.npy"
VAL_ROOT = "/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/val"
GT_ROOT = "/petrobr/parceirosbr/home/joao.frare/workspace/spfm/sharedata/datasets/ImageNet_2012/extra_files/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"
MAT_ROOT = "/petrobr/parceirosbr/home/joao.frare/workspace/spfm/sharedata/datasets/ImageNet_2012/extra_files/ILSVRC2012_devkit_t12/data/meta.mat"
PRETRAIN_LOGS_PATH = f"checkpoints+logs/logs/pretrain/{args.repetition}"
PRETRAIN_CKPT_PATH = f"checkpoints+logs/checkpoints/pretrain/{args.repetition}"
#--------------------------------------NOME DO MODELO-------------------------------------------------
model_name = f"V{args.repetition}_pretrain_imagenet_{args.per_class}_per_class_{args.num_classes}_num_classes"

#--------------------------------------DADOS----------------------------------------------------------
train_transform_pipeline = create_transform(
    input_size=224,
    is_training=True,
    auto_augment='rand-m7-n2-mstd0.5', 
    interpolation='bicubic',
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)

train_reader = ImagenetReader(DATASET_ROOT, TRAIN_ENTRIES)
train_dataset = ImagenetDataset(
    ImagenetReader= train_reader,
    transform= train_transform_pipeline,
)
train_subset = StratifiedSubset(train_dataset, args.per_class, args.repetition)

#a validação funciona da seguinte forma, a cada x steps o trainer pega um dado da validação e aplica no modelo para inferir se o aprendizado está fluindo

val_transform_pipeline = create_transform(
    input_size=224,
    is_training=False,
    interpolation='bicubic',
    crop_pct=0.95, # Ajuste para bater com a tabela
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)

val_reader = ImagenetValReader(VAL_ROOT, GT_ROOT, MAT_ROOT)
val_dataset = ImagenetDataset(
    ImagenetReader= val_reader,
    transform= val_transform_pipeline
)

data_module = MinervaDataModule(
            train_dataset=train_subset,
            val_dataset= val_dataset,
            batch_size=batch_size,
            drop_last=True,
            shuffle_train=True,
            name="imagenet",
            num_workers=10
        )

#------------------------------------MODELO-----------------------------------------------------------
backbone = timm.create_model('resnet_50', num_classes= 0)
model = ImagenetModel(
    num_classes= args.num_classes,
    optimizer=timm.optim.Lamb,
    optimizer_kwargs={
        "lr": 0.008,
        "weight_decay": 0.02
    },
    lr_scheduler=timm.scheduler.CosineLRScheduler,
    lr_scheduler_kwargs={
        "t_initial": max_steps,
        "warmup_t": int(max_steps*0.05),              
        "warmup_lr_init": 2e-6,       
        "lr_min": 1e-6,             
        "t_in_epochs": False     
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
    filename='best',                    
    auto_insert_metric_name=False
)

lr_monitor = LearningRateMonitor(logging_interval= 'step', log_momentum= True)

callbacks = [ckpt_callback, lr_monitor]
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
)

pipeline = SimpleLightningPipeline(
    model=model,
    trainer=trainer,
    log_dir=log_dir,
    save_run_status=True,
)

last_ckpt = ckpt_dir/"last.ckpt"

if last_ckpt.exists():
    pipeline.run(data_module, task="fit", ckpt_path= last_ckpt)
else:
    pipeline.run(data_module, task="fit")

