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
from timm.scheduler import CosineLRScheduler

# -------------------- Lightning --------------------
import lightning as L
from lightning import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

# -------------------- Minerva & Custom Modules (Seus Módulos) --------------------
from minerva.pipelines.lightning_pipeline import SimpleLightningPipeline
from minerva.data.data_modules import MinervaDataModule
from minerva.models.loaders import FromPretrained

#--------------------Base-------------------------------------------------------
from base.ImagenetDataset import ImagenetDataset, DefaultTrainSubset, DefaultValSubset
from base.ImagenetReader import ImagenetReader, ImagenetValReader
from base.ImagenetModel import ImagenetModel
from base.InformationOrganizer import PretrainEvaluationOrganizer

from base.ImagenetDataset import DefaultValSubset
from base.utils import *



DATASET_ROOT = "/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/train"
TRAIN_ENTRIES = "/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/extras_v3/entries-TRAIN.npy"
VAL_ROOT = "/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/val"
GT_ROOT = "/petrobr/parceirosbr/home/joao.frare/workspace/spfm/sharedata/datasets/ImageNet_2012/extra_files/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"
MAT_ROOT = "/petrobr/parceirosbr/home/joao.frare/workspace/spfm/sharedata/datasets/ImageNet_2012/extra_files/ILSVRC2012_devkit_t12/data/meta.mat"

organizer = PretrainEvaluationOrganizer(data_root= '/petrobr/parceirosbr/spfm/joao.frare/logs+checkpoints_imagenet')


print('Starting ImageNet Linear Readout Evaluation')
print(f'infos: \n {organizer}')

#----------------------------------------------------------------valores importantes:
full_imagenet_size = 1281167                                       #número de imagens de treino total do Imagenet

devices = 1                                                        #número de gpus a serem utilizados        
strategy= 'auto'      
batch_size = 2048

accumulate_grad_batches = 1                                        
real_batch_size = accumulate_grad_batches * batch_size * devices
max_steps = full_imagenet_size*25//real_batch_size + 30          


precision= "16-mixed" if torch.cuda.is_available() else "32"       #precisão -> quanto maior melhor
limit_val_batches = 1.0
log_every_n_steps = 600

seed_everything(organizer.args.repetition)

#-------------------------------------------------pegando os pesos e colocando no modelo:
ckpt = torch.load(Path(organizer.ckpt_dir)/'best.ckpt', map_location = 'cpu')
state_dict = ckpt['state_dict']


new_state_dict = {k.replace('backbone.',''): v for k,v in state_dict.items() if k.startswith('backbone.')} #isso precisa ser feito pq na hora de salvar o lightning coloca o prefixo 'backbone.' no state dict
backbone = timm.create_model('resnet50', num_classes= 0, pretrained= False)
missing, unexpected = backbone.load_state_dict(new_state_dict, strict= False)
check_transfer_learning(missing)

fc = nn.Linear(2048, 1000)

train_loss_fn = nn.CrossEntropyLoss()
val_loss_fn = nn.CrossEntropyLoss()

train_metrics = {}
val_metrics = {
    'val_acc1': torchmetrics.Accuracy(task= 'multiclass', num_classes= 1000, top_k= 1),
    'val_acc5': torchmetrics.Accuracy(task= 'multiclass', num_classes= 1000, top_k= 5)}

model = ImagenetModel(
    num_classes= 1000,
    optimizer=timm.optim.Lamb,
    optimizer_kwargs={
        "lr": 0.001,
        "weight_decay": 0.01
    },
    lr_scheduler=CosineLRScheduler,
    lr_scheduler_kwargs={
        "t_initial": max_steps,
        "lr_min": 1e-6,

        "cycle_mul": 1.0,
        "cycle_decay": 0.5,
        "cycle_limit": 1,

        "warmup_t": int(0.05*max_steps),
        "warmup_lr_init": 1e-4,
        "warmup_prefix": False,

        "t_in_epochs": False,

        "noise_range_t": None,
        "noise_pct": 0.67,
        "noise_std": 1.0,
        "noise_seed": organizer.args.repetition,

        "k_decay": 1.0,
        "initialize": True,
    },

    batch_level_transforms= None,
    train_loss_fn = train_loss_fn,
    train_metrics= train_metrics,
    val_loss_fn= val_loss_fn,
    val_metrics= val_metrics,
    backbone= backbone,
    fc= fc,
    num_gpus= devices
)

model.freeze_backbone()

#---------------------------------------------------------------readers e dataset (com transformações):
train_transform_pipeline = create_transform( 
    input_size=160,
    is_training=True,
    scale=(0.08, 1.0),
    ratio=(0.75, 4 / 3),
    hflip=0.5,
    vflip=0.0,
    color_jitter=0.4,
    re_prob= 0.0,
    re_mode= 'pixel',
    re_count= 1,
    interpolation='bicubic', 
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)

val_transform_pipeline = create_transform(
    input_size=224,
    is_training=False,
    interpolation='bicubic',
    crop_pct=0.95,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)

train_reader = ImagenetReader(root= DATASET_ROOT, entries_path= TRAIN_ENTRIES)
val_reader = ImagenetValReader(root= VAL_ROOT, gt_path= GT_ROOT, mat_path= MAT_ROOT)

val_dataset = ImagenetDataset(
    ImagenetReader= val_reader,
    transform= val_transform_pipeline
)

train_dataset = ImagenetDataset(
    ImagenetReader= train_reader,
    transform= train_transform_pipeline,
)

#pipeline de finetuning e posterior avaliação
if organizer.parser.evaluate_top1:
    try:
        cpus_disponiveis = len(os.sched_getaffinity(0))
    except AttributeError:
        cpus_disponiveis = os.cpu_count() or 1
    num_workers = min(24, cpus_disponiveis)

    data_module = MinervaDataModule(
                train_dataset=train_dataset,
                val_dataset= val_dataset,
                test_dataset= val_dataset,
                batch_size=batch_size,
                drop_last=True,
                shuffle_train=True,
                name="imagenet",
                num_workers= num_workers,
                additional_train_dataloader_kwargs={"persistent_workers": True, "pin_memory": True, "drop_last": True},
                additional_val_dataloader_kwargs={"persistent_workers": True, "pin_memory": True, "drop_last": False}
            )

    #-----------------------------------DIRETORIOS, LOGGERS E CALLBACKS----------------------------------------
    CSVlogger = CSVLogger(Path(organizer.evaluation_dir), name= '', version='')

    ckpt_callback = ModelCheckpoint(
        monitor='val_acc1',                # monitorar a val_acc1
        mode='max',
        save_top_k=1,                      # Salva apenas o maior val_acc1
        save_last=False,                   
        dirpath=Path(organizer.evaluation_ckpt_dir),
        filename='best',                    
        auto_insert_metric_name=False
    )

    lr_monitor = LearningRateMonitor(logging_interval= 'step', log_momentum= True)

    callbacks = [ckpt_callback, lr_monitor]


    #-----------------------------treino e avaliação

    if not Path(organizer.evaluation_ckpt_dir/'best.ckpt').exists():
        trainer = Trainer(
            accelerator='gpu',
            devices=devices,
            strategy=strategy,
            precision=precision,
            logger=CSVlogger,
            callbacks= callbacks,
            max_steps= max_steps,
            accumulate_grad_batches = accumulate_grad_batches,
            check_val_every_n_epoch=None,
            val_check_interval=625,                                              
            limit_val_batches=limit_val_batches,                                 
            log_every_n_steps=log_every_n_steps,              
            benchmark=True,
        )
        pipeline = SimpleLightningPipeline(
            model=model,
            trainer=trainer,
            log_dir=Path(organizer.evaluation_dir),
            save_run_status=True,
        )
        pipeline.run(data_module, task= 'fit')

    trainer = Trainer(
        accelerator='gpu',
        devices=1,
        strategy='auto',
        precision= precision
    )

    metrics = {'Accuracy Top-1': torchmetrics.Accuracy(task= 'multiclass', num_classes= 1000, top_k= 1)}
    eval_pipeline = SimpleLightningPipeline(
        model= model,
        trainer= trainer,
        log_dir= Path(organizer.evaluation_dir),
        seed=organizer.args.repetition,
        classification_metrics=metrics
    )

    eval_pipeline.run(data_module, task= 'evaluate', ckpt_path = Path(organizer.evaluation_ckpt_dir)/'best.ckpt')