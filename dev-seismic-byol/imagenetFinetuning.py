import argparse
from pathlib import Path
#--------------------------------------
from functions import *

# -------------------- PyTorch --------------------
from torchmetrics import Accuracy, JaccardIndex, F1Score
# -------------------- Lightning --------------------
import lightning as L
from lightning import Trainer
from lightning.fabric import seed_everything

parser = argparser.ArgumentParser(
    description= "Finetuning on Imagenet"
)

parser.add_argument(
    "per_class", type= int, description= "per_class_model used in pretrain"
)

parser.add_argument(
    "repetition", type= int, description= "repetition used"
)

parser.add_argument(
    "finetune_dataset", type= str, description= "dataset used in finetuning (seam_ai_N or f3_N)"
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
deeplab_backbone = DeepLabV3Backbone(num_classes=num_classes, pretrained= True, weigths_path= f"{pretrain_ckpt_path}/best.ckpt")

model = DeepLabV3(
    backbone=deeplab_backbone,
    learning_rate=1e-5,
    num_classes=num_classes,
    freeze_backbone=False,
)

#----------------------------Dados - Modelagem----------------------------------------
dataset_path = mapping[args.finetune_dataset]

if agrs.finetune_dataset == 'seam_ai_N':
    print("Using padding of (256,704)")
    padding = Padding(256, 704)
elif args.finetune_dataset == 'f3_N':
    print("Using padding of (1008,592)")
    padding = Padding(1008, 592)

transform_pipeline = TransformPipeline([
    padding,
    Transpose([2,0,1])
])

logger.info("Using 100% of train data")
train_dataset = SeismicFullDataset(root=data_path, partition='train', transform=transform_pipeline)
data_module = SeismicDataModule(
    root = mapping[args.finetune_dataset],
    batch_size=8,
    cap=1.0,
    drop_last=True,
    transform=transform_pipeline,
    test_transform=transform_pipeline,
    train_dataset = train_dataset,
    val_dataset = None,
    test_dataset = None,
    )

csv_logger = CSVLogger(log_dir, name=save_name, version=finetune_data)
#------------------------TRAINER----------------------------------------------------------
trainer = Trainer(
    model= model,
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
    log_dir=log_dir,
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
    log_dir=log_dir,
    save_run_status=True,
    seed=repetition,
    apply_metrics_per_sample=False,
    classification_metrics=metrics,
)
    
pipeline.run(data_module, task="evaluate")