# -------------------- Python base --------------------
import os
from pathlib import Path

# -------------------- PyTorch & Timm --------------------
import torch
import torch.nn as nn
from torchmetrics import Accuracy, JaccardIndex, F1Score
import timm
import timm.optim
from timm.loss import BinaryCrossEntropy
from torch.optim.lr_scheduler import ReduceLROnPlateau

# -------------------- Lightning --------------------
from lightning import Trainer
from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.fabric import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint

# -------------------- Minerva --------------------
from minerva.models.nets.image.deeplabv3 import DeepLabV3Backbone, DeepLabV3, DeepLabV3PredictionHead
from minerva.models.loaders import FromPretrained
from minerva.pipelines.lightning_pipeline import SimpleLightningPipeline
from minerva.transforms.transform import TransformPipeline, Transpose, Padding

#--------------------- Locais & Custom -----------------------
from base.utils import * 
from seismic.SeismicModel import SeismicModel
from seismic.DatasetsDatamodules import *
from seismic.LinearPredHead import *
from base.ImagenetModel import ImagenetModel
from base.InformationOrganizer import FinetuningOrganizer

#----------------------------------------Organizador do finetuning----------------------
DATASET_ROOT = "/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/train"
TRAIN_ENTRIES = "/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/extras_v3/entries-TRAIN.npy"
VAL_ROOT = "/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/val"
GT_ROOT = "/petrobr/parceirosbr/home/joao.frare/workspace/spfm/sharedata/datasets/ImageNet_2012/extra_files/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"
MAT_ROOT = "/petrobr/parceirosbr/home/joao.frare/workspace/spfm/sharedata/datasets/ImageNet_2012/extra_files/ILSVRC2012_devkit_t12/data/meta.mat"
organizer = FinetuningOrganizer(data_root= '/petrobr/parceirosbr/spfm/joao.frare/logs+checkpoints_imagenet')
organizer.set_readers(DATASET_ROOT, TRAIN_ENTRIES, VAL_ROOT, GT_ROOT, MAT_ROOT)
seed_everything(organizer.args.repetition)
#----------------------------------MODELO - Transfer Learning---------------------------
num_classes = 6
learning_rate = 1e-3
num_epochs = 30
batch_size = 8
deeplab_backbone = DeepLabV3Backbone(num_classes=num_classes)

print(f'Scratch: {organizer.args.scratch} || Backbone Config: {organizer.args.backbone_freeze} || Pred_Head: {organizer.args.pred_head}')
#--------------------------------------------------------------------Importando modelo pré-treinado--------------------------------------------------------------
if not organizer.args.scratch:
    resnet50_backbone = timm.create_model('resnet50', pretrained=False, output_stride=8, num_classes= 0)
    fc = nn.Linear(2048, organizer.args.num_classes)
    train_loss_fn = BinaryCrossEntropy()
    val_loss_fn = BinaryCrossEntropy()

    pretrained_model = ImagenetModel(
        backbone= resnet50_backbone, 
        fc= fc, 
        train_loss_fn= train_loss_fn,
        val_loss_fn= val_loss_fn,
        train_metrics = {},
        val_metrics= {},
        optimizer=  None,
        optimizer_kwargs= {},
        lr_scheduler= None,
        lr_scheduler_kwargs= {},
        batch_level_transforms= None,
        num_classes=  organizer.args.num_classes,
        num_gpus= 2)

    ckpt = organizer.ckpt_dir / "best.ckpt"
    assert ckpt.exists(), f"Checkpoint não encontrado: {ckpt}"

    weighted_backbone = FromPretrained(model= pretrained_model, ckpt_path= f'{organizer.ckpt_dir}/best.ckpt', strict= False, error_on_missing_keys= False, ckpt_load_weights_only= False).backbone
    weighted_state_dict= get_state_dict(weighted_backbone)
    possible_errors = deeplab_backbone.load_state_dict(weighted_state_dict, strict= False)

    check_transfer_learning(possible_errors.missing_keys)

#-------------------------------------------------------------------Cabeça de Segmentação-------------------------------------------------------------------------
if organizer.args.pred_head == 'deeplab':
    pred_head = DeepLabV3PredictionHead(num_classes=num_classes)
else:
    #linear
    pred_head = LinearSegmentationHead(in_channels= 2048, num_classes= num_classes)

#------------------------------------------------------------------Modelo------------------------------------------------------------------------------------------
val_metrics = {
    "mIoU": JaccardIndex(
        num_classes=num_classes, average="macro", task="multiclass"
    ),
    "acc": Accuracy(num_classes=num_classes, task="multiclass"),
    "f1-weighted": F1Score(
        num_classes=num_classes, task="multiclass", average="weighted"
    ),
}
#Parametros em comum: ajustar o freeze_backbone e o freeze_layers
training_parameters = {
    'backbone': deeplab_backbone,
    'pred_head': pred_head,
    'num_classes': num_classes,
    'val_metrics': val_metrics,
    'optimizer': torch.optim.AdamW,
    'optimizer_kwargs': {
        'weight_decay': 1e-4,
        'lr': learning_rate,
    },

    "lr_scheduler": ReduceLROnPlateau,
    "lr_scheduler_kwargs": {
        "mode": "max",
        "factor": 0.5,
        "patience": 3,
        "cooldown": 1,
        "min_lr": 1e-6,
    },
}


if organizer.args.backbone_freeze == 'full_freeze':
    model = SeismicModel(
        freeze_backbone= True,
        **training_parameters
    )

elif organizer.args.backbone_freeze == 'custom_freeze':
    layers = ['conv1', 'bn1', 'layer1', 'layer2']
    model = SeismicModel(
        freeze_layers= layers,
        freeze_backbone= False,
        **training_parameters
    )

else:
    #full_finetuning
    model = SeismicModel(
        freeze_backbone= False,
        **training_parameters
    )

#----------------------------Dados - Modelagem----------------------------------------
mapping = get_dataset_mapping()
dataset_path = mapping[organizer.args.finetune_dataset]

if organizer.args.finetune_dataset == 'f3_N':
    print("Using padding of (256,704)")
    padding = Padding(256, 704)
elif organizer.args.finetune_dataset == 'seam_ai_N':
    print("Using padding of (1008,592)")
    padding = Padding(1008, 592)

transform_pipeline = TransformPipeline([
    padding,
    Transpose([2,0,1])
])

train_dataset = SeismicFullDataset(root=dataset_path, partition='train', transform=transform_pipeline)
data_module = SeismicDataModule(
    root = dataset_path,
    batch_size=batch_size,
    cap=1.0,
    drop_last=True,
    transform=transform_pipeline,
    test_transform=transform_pipeline,
    train_dataset = train_dataset,
    val_dataset = None,
    test_dataset = None,
    )

#DEBUG

batch = next(iter(data_module.val_dataloader()))
x, y = batch

print("predict split:", data_module._predict_split)

print("train size:", len(data_module.train_dataset))
print("val size:", len(data_module.val_dataset))
print("test size:", len(data_module.test_dataset))
print("predict size:", len(data_module.predict_dataset))

csv_logger = CSVLogger(organizer.finetune_log_dir, name='', version= '')
#------------------------Callbacks-------------------------------------------------------------------------
ckpt_callback = ModelCheckpoint(
    monitor= 'val_mIoU',
    mode= 'max',
    save_top_k=1,
    save_last= False,
    dirpath= organizer.finetune_ckpt_dir,
    filename= 'best',
    auto_insert_metric_name=False
)
#------------------------TRAINER---------------------------------------------------------------------------

trainer = Trainer(
    logger= csv_logger,
    max_epochs= num_epochs,
    limit_val_batches = 1.0,
    strategy= 'auto',
    devices= 1,
    check_val_every_n_epoch=1,
    callbacks= [ckpt_callback]
)

pipeline = SimpleLightningPipeline(
    model=model,
    trainer=trainer,
    log_dir=organizer.finetune_log_dir,
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
    log_dir=organizer.finetune_log_dir,
    save_run_status=True,
    seed=organizer.args.repetition,
    apply_metrics_per_sample=False,
    classification_metrics=metrics,
)
    
pipeline.run(data_module, task="evaluate", ckpt_path= organizer.finetune_ckpt_dir / 'best.ckpt')