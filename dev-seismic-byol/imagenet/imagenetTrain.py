import argparse 
import os
from pathlib import Path

# -------------------- Torch & TorchVision --------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

# -------------------- Timm (Modelos, Otimização e Augmentation) --------------------
import timm
import timm.optim
from timm.data import Mixup, create_transform
from timm.loss import BinaryCrossEntropy

# -------------------- Lightning --------------------
import lightning as L
from lightning import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.optim.lr_scheduler import OneCycleLR

# -------------------- Minerva & Custom Modules (Seus Módulos) --------------------
from minerva.pipelines.lightning_pipeline import SimpleLightningPipeline
from minerva.data.data_modules import MinervaDataModule

#--------------------Base-------------------------------------------------------
from base.ImagenetDataset import ImagenetDataset, StratifiedSubset
from base.ImagenetReader import ImagenetReader, ImagenetValReader
from base.ImagenetModel import ImagenetModel
from base.InformationOrganizer import TrainOrganizer

#---------------------------------ARGUMENTOS DO TREINO------------------------
organizer = TrainOrganizer(data_root= '/petrobr/parceirosbr/home/joao.frare/workspace/spfm/Seismic-Byol/dev-seismic-byol/imagenet/logs+checkpoints')
#------------------------------------------------------------------------------
train_imagenet_size = 1281167                                      #número de imagens de treino total do Imagenet

devices = 2                                                        #número de gpus a serem utilizados        
strategy= 'ddp'      
batch_size = 1024

accumulate_grad_batches = 1                                        #variável que carrega o batch total de pouco no trainer, dribla problemas físicos (quantidade de VRAM)
real_batch_size = accumulate_grad_batches * batch_size * devices
max_steps = train_imagenet_size*100//real_batch_size + 30          #número de passos para se treinar uma imagenet completa por 150 épocas


precision= "16-mixed" if torch.cuda.is_available() else "32"       #precisão -> quanto maior melhor
limit_val_batches = 1.0
log_every_n_steps = 100
num_classes = organizer.args.num_classes                           #número de classes padrão (1000), caso necessário, será modificado

seed_everything(organizer.args.repetition)

#--------------------------------------PATHS IMPORTANTES-----------------------------------------------
DATASET_ROOT = "/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/train"
TRAIN_ENTRIES = "/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/extras_v3/entries-TRAIN.npy"
VAL_ROOT = "/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/val"
GT_ROOT = "/petrobr/parceirosbr/home/joao.frare/workspace/spfm/sharedata/datasets/ImageNet_2012/extra_files/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"
MAT_ROOT = "/petrobr/parceirosbr/home/joao.frare/workspace/spfm/sharedata/datasets/ImageNet_2012/extra_files/ILSVRC2012_devkit_t12/data/meta.mat"
#--------------------------------------TRANSFORMAÇÕES---------------------------------------------------------
train_transform_pipeline = create_transform( 
    input_size=224,
    is_training=True,
    auto_augment='rand-m6-n2-mstd0.5', 
    interpolation='bicubic',
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)
#validação: a cada x steps se passa o conjunto de validação (imagens inéditas que não são processadas no gradiente) para aferir a 
#eficiência do treino
val_transform_pipeline = create_transform(
    input_size=224,
    is_training=False,
    interpolation='bicubic',
    crop_pct=0.95,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)

#--------------------------------------------READERS E DATASET---------------------------------------------
train_reader = ImagenetReader(DATASET_ROOT, TRAIN_ENTRIES)
val_reader = ImagenetValReader(VAL_ROOT, GT_ROOT, MAT_ROOT)

if organizer.args.reduction_mode == 'taxonomic':
    #fará o agrupamento top down ou bottom up
    mode = 'Top Down' if organizer.args.top_down else 'Bottom Up'
    num_classes = val_reader.to_coarse_classes(top_down= organizer.args.top_down, level= organizer.args.level, mat_path= MAT_ROOT)
    train_reader.to_coarse_classes(top_down= organizer.args.top_down, level= organizer.args.level, mat_path= MAT_ROOT)

    print(f'Using {num_classes} after {mode} clustering!')

val_dataset = ImagenetDataset(
    ImagenetReader= val_reader,
    transform= val_transform_pipeline
)

train_dataset = ImagenetDataset(
    ImagenetReader= train_reader,
    transform= train_transform_pipeline,
)

if organizer.args.reduction_mode == 'default':
    #instancia o subset no caso do número de classes e/ou imagens por classe ser variável
    train_dataset = StratifiedSubset(dataset= train_dataset, per_class= organizer.args.per_class, seed= organizer.args.repetition, num_classes= num_classes)

# Coleta a afinidade real de CPUs entregues pelo cgroup do SLURM no nó
try:
    cpus_disponiveis = len(os.sched_getaffinity(0))
except AttributeError:
    cpus_disponiveis = os.cpu_count() or 1
num_workers = min(24, cpus_disponiveis)


data_module = MinervaDataModule(
            train_dataset=train_dataset,
            val_dataset= val_dataset,
            batch_size=batch_size,
            drop_last=True,
            shuffle_train=True,
            name="imagenet",
            num_workers= num_workers
        )

#------------------------------------MODELO-----------------------------------------------------------
backbone = timm.create_model('resnet50', num_classes= 0, pretrained= False)
fc = nn.Linear(2048, num_classes)
batch_level_transforms = Mixup(
    mixup_alpha= 0.1,
    cutmix_alpha= 1.0,
    prob= 1.0,
    mode= 'batch',
    switch_prob= 0.5,
    num_classes= num_classes
)
train_loss_fn = BinaryCrossEntropy()
val_loss_fn = BinaryCrossEntropy()

train_metrics = {}
val_metrics = {
    'val_acc1': torchmetrics.Accuracy(task= 'multiclass', num_classes= num_classes, top_k= 1),
    'val_acc5': torchmetrics.Accuracy(task= 'multiclass', num_classes= num_classes, top_k= 5)}

model = ImagenetModel(
    num_classes= num_classes,
    optimizer=timm.optim.Lamb,
    optimizer_kwargs={
        "lr": 0.008,
        "weight_decay": 0.02
    },
    lr_scheduler=OneCycleLR,
    lr_scheduler_kwargs={
        "max_lr": 0.008,
        "total_steps": max_steps,              
        "pct_start": 0.05,       
        "anneal_strategy": 'cos',             
        "div_factor": 4000,
        "final_div_factor": 2000     
    },

    batch_level_transforms= batch_level_transforms,
    train_loss_fn = train_loss_fn,
    train_metrics= train_metrics,
    val_loss_fn= val_loss_fn,
    val_metrics= val_metrics,
    backbone= backbone,
    fc= fc,
    num_gpus= devices
)
#-----------------------------------DIRETORIOS, LOGGERS E CALLBACKS----------------------------------------
CSVlogger = CSVLogger(organizer.log_dir, name=organizer.model_name, version="imagenet")

ckpt_callback = ModelCheckpoint(
    monitor='val_loss',                # monitorar a val_loss
    save_top_k=1,                      # Salva apenas o menor val_loss
    save_last=True,                    # Salva o estado final para resume
    dirpath=organizer.ckpt_dir,
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
    accumulate_grad_batches = accumulate_grad_batches,
    val_check_interval=1.0,                                             #vai validar depois de uma época
    limit_val_batches=limit_val_batches,                                #quantos batches serão usados na validação
    log_every_n_steps=log_every_n_steps,              
    benchmark=True,
)

pipeline = SimpleLightningPipeline(
    model=model,
    trainer=trainer,
    log_dir=organizer.log_dir,
    save_run_status=True,
)

last_ckpt = Path(organizer.ckpt_dir)/"last.ckpt"

if last_ckpt.exists():
    pipeline.run(data_module, task="fit", ckpt_path= last_ckpt)
else:
    pipeline.run(data_module, task="fit")

