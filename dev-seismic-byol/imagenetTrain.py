# -------------------- Python base --------------------
import os
import argparse
from pathlib import Path

# -------------------- PyTorch --------------------
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR

# -------------------- Torchvision --------------------
from torchvision.models import resnet50
from torchvision import transforms

# -------------------- Lightning --------------------
import lightning as L
from lightning import Trainer
from lightning.fabric import seed_everything

# Logger e checkpoint
from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

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
batch_size = 1024

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
    limit_val_batches = 1.0

    log_every_n_steps = args.per_class*200//batch_size

if args.per_class < 0 or args.per_class > 1200:
    raise KeyError("error, incorrect amount of samples")

seed_everything(args.repetition)

#--------------------------------------PATHS IMPORTANTES-----------------------------------------------
DATASET_ROOT = "/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/train"
TRAIN_ENTRIES = "/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/extras_v3/entries-TRAIN.npy"
VAL_ROOT = "/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/val"
GT_ROOT = "/petrobr/parceirosbr/home/joao.frare/workspace/spfm/sharedata/datasets/ImageNet_2012/extra_files/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"
MAT_ROOT = "/petrobr/parceirosbr/home/joao.frare/workspace/spfm/sharedata/datasets/ImageNet_2012/extra_files/ILSVRC2012_devkit_t12/data/meta.mat"
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
            num_workers=min(os.cpu_count(), 24)
        )

#------------------------------------MODELO-----------------------------------------------------------
backbone = resnet50(weights=None)
backbone.fc = nn.Identity()
fc = nn.Linear(2048, 1000)
loss_fn = nn.CrossEntropyLoss(label_smoothing= 0.1)

steps_to_10_epochs = (10*args.per_class*1000)/batch_size
ratio = steps_to_10_epochs/max_steps

model = ImagenetModel(
    backbone=backbone,
    fc=fc,
    loss_fn=loss_fn,
    optimizer=torch.optim.SGD,
    optimizer_kwargs={
        "lr": 0.4,
        "momentum": 0.9,
        "weight_decay": 1e-4
    },
    lr_scheduler=torch.optim.lr_scheduler.OneCycleLR,
    lr_scheduler_kwargs={
        "max_lr": 0.4,
        "total_steps": max_steps, 
        "pct_start": ratio,               
        "anneal_strategy": 'cos',       #forma como o learning rate vai osclinar -> em cosseno mesmo
        "div_factor": 25.0,             # learning_rate inicial = max/div_factor
        "final_div_factor": 1000.0      # learning_rate_final = max/final_div_factor
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

early_stop_callback = EarlyStopping(
    monitor= 'val_loss',                 #métrica vigiada
    min_delta = 0.0001,                   #minima variação aceitável
    patience= 50,                        #quantas variações menores que min_delta consecutivas ele tolera antes de parar o treino
    verbose= True,
    mode= 'min'                         #mode min = minimização de perda
)

lr_monitor = LearningRateMonitor(logging_interval= 'step', log_momentum= True)

callbacks = [ckpt_callback, lr_monitor]
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

last_ckpt = ckpt_dir/"last.ckpt"

if last_ckpt.exists():
    pipeline.run(data_module, task="fit", ckpt_path= last_ckpt)
else:
    pipeline.run(data_module, task="fit")

