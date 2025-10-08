from pathlib import Path

from functions import *
from functions import logger

from minerva.transforms.transform import *
from minerva.transforms.random_transform import *

from minerva.data.readers import TiffReader, PNGReader
from minerva.data.datasets import SimpleDataset

from minerva.pipelines.lightning_pipeline import SimpleLightningPipeline
from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning import Trainer
from lightning.fabric import seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


def main(
    pretrain_data,  # Where the model was pretrained
    finetune_data,  # Where the model is going to be trained
    data_path,      # Root path for the data
    num_epochs,     # Epochs to be trained
    batch_size,     
    repetition,     # Repetition and random seed
    learning_rate,  
    cap,            # Amount of samples to be used. 1.0 means 100%
    freeze,         # Freeze backbone
    ckpt_path,      # Where to save the checkpoints
    logs_path,      # Where to save the logs
    import_root_path,   # Root to import models
    gpus,           # Gpus to run       
    import_path=False,  # Flag to tell if the import path is root or complete
    full_save_name=False,   # If want to save as a specific name
    linear=False,   # FC to choose from linear or default
    steps=False,    # If num_epochs refer to steps
):

    # Set general seed**
    seed_everything(repetition)
    if full_save_name:
        save_name = full_save_name
    else:
        if isinstance(cap, float):
            save_name = (
                f"V{repetition}_pre_{pretrain_data}_train_{finetune_data}_cap_{cap*100:.0f}%"
            )
        elif isinstance(cap, int):
            save_name = (
                f"V{repetition}_pre_{pretrain_data}_train_{finetune_data}_cap_{cap}_img"
            )
        
    logger.info(f"Saving model {save_name}")
    
    # Transforms
    if finetune_data == "f3" or finetune_data == "f3_N":
        logger.info("Using padding of (256,704)")
        padding = Padding(256, 704)
    elif finetune_data == "seam_ai" or finetune_data == "seam_ai_N":
        logger.info("Using padding of (1008,592)")
        padding = Padding(1008, 592)

    # 100% dos dados
    # print(f'Número: {cap}, tipo: {type(cap)}')
    if cap == 1.0 and type(cap) == float:
        
        logger.info("Using 100% of train data")
        train_dataset = SeismicFullDataset(root=data_path, partition='train', transform=padding)
        data_module = SeismicDataModule(
            root = data_path,
            batch_size=batch_size,
            cap=cap,
            drop_last=True,
            transform=padding,
            test_transform=padding,
            train_dataset = train_dataset,
            val_dataset = None,
            test_dataset = None,
        )
        
    else:
        logger.info(f'Using {cap} samples of train data')
        data_module = SeismicDataModule(
            root = data_path,
            batch_size=batch_size,
            cap=cap,
            drop_last=False,
            transform=padding,
            test_transform=padding,
            train_dataset = None,
            val_dataset = None,
            test_dataset = None,
        )

    # Model

    model = get_model(
        pretrain_data,
        learning_rate,
        freeze,
        repetition,
        import_root_path,
        full_path=import_path, 
        linear=linear,
        finetune_data=finetune_data
    )

    log_dir = Path(logs_path) / save_name / finetune_data
    ckpt_dir = Path(ckpt_path) / save_name / finetune_data
    csv_logger = CSVLogger(log_dir, name=save_name, version=finetune_data)
    ckpt_callback = ModelCheckpoint(
        save_top_k=1, save_last=True, dirpath=ckpt_dir, mode="min", monitor="val_loss"
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=20,
        mode="min",
        # verbose=True
    )

    if steps:
        trainer = Trainer(
            accelerator="gpu",
            logger=csv_logger,
            callbacks=[ckpt_callback, early_stopping_callback],
            max_steps=num_epochs,
            strategy="auto",
            devices=gpus,
            check_val_every_n_epoch=None,
            val_check_interval=100,
        )

    else:

        trainer = Trainer(
            accelerator="gpu",
            logger=csv_logger,
            callbacks=[ckpt_callback, early_stopping_callback],
            max_epochs=num_epochs,
            strategy="auto",
            devices=gpus,
            check_val_every_n_epoch=None,
            val_check_interval=200,
        )

    pipeline = SimpleLightningPipeline(
        model=model,
        trainer=trainer,
        log_dir=log_dir,
        save_run_status=True,
    )

    pipeline.run(data_module, task="fit")
    
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
    
    
    pipeline.run(data_module, task="fit")


if __name__ == "__main__":
    main(
        pretrain_data="imagenet",
        finetune_data="f3_N",
        data_path="/petrobr/parceirosbr/home/vinicius.soares/workspace/spfm/datasets/tiff_data/f3_segmentation_N",
        num_epochs=5,
        batch_size=8,
        repetition=5,
        learning_rate=0.001,
        cap=1.0,
        freeze=True,
        ckpt_path="/home/vinicius.soares/Seismic-Byol/dev-seismic-byol/ht_logs/train/9",
        logs_path="/home/vinicius.soares/Seismic-Byol/dev-seismic-byol/ht_logs/train/9",
        import_root_path='ckpt_ht/pretrain/',
        gpus=[0],
    )
