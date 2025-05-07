from pathlib import Path

from functions import *

from minerva.transforms.transform import *
from minerva.transforms.random_transform import *

from minerva.data.readers import TiffReader, PNGReader
from minerva.data.datasets import SimpleDataset

from minerva.pipelines.lightning_pipeline import SimpleLightningPipeline
from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning import Trainer
from torchmetrics import Accuracy, JaccardIndex, F1Score
from lightning.fabric import seed_everything



def main(
    ckpt_file,
    model_name,
    finetune_data,
    pretrain_data,
    data_path,
    num_epochs,
    batch_size,
    repetition,
    ckpt_path,
    logs_path,
    gpus,
):

    # Set general seed
    seed_everything(repetition)

    # Transforms
    if finetune_data == "f3" or finetune_data == "f3_N":
        padding = Padding(256, 704)
    elif finetune_data == "seam_ai" or finetune_data == "seam_ai_N":
        padding = Padding(1008, 592)

    # Dataset

    image_path = f"{data_path}/images"
    label_path = f"{data_path}/annotations"

    train_data_reader = TiffReader(path=f"{image_path}/train")
    train_label_reader = PNGReader(path=f"{label_path}/train")

    val_data_reader = TiffReader(path=f"{image_path}/val")
    val_label_reader = PNGReader(path=f"{label_path}/val")

    test_data_reader = TiffReader(path=f"{image_path}/test")
    test_label_reader = PNGReader(path=f"{label_path}/test") 

    train_dataset = SimpleDataset(
        readers=[
            train_data_reader,
            train_label_reader,
        ],
        transforms=padding,
        return_single=False,
    )

    val_dataset = SimpleDataset(
        readers=[
            val_data_reader,
            val_label_reader,
        ],
        transforms=padding,
        return_single=False,
    )

    test_dataset = SimpleDataset(
        readers=[
            test_data_reader,
            test_label_reader,
        ],
        transforms=padding,
        return_single=False,
    ) 


    # DataModule

    data_module = CapDataModule(
        cap_train=1,
        cap_val=1,
        cap_test=1,
        seed=repetition,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle_train=True,
    )

    # Model

    model = get_eval_model(
        pretrain_data=pretrain_data,
        import_path=ckpt_file,
        learning_rate=0.001
        )

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

    log_dir = Path(logs_path) / model_name
    ckpt_dir = Path(ckpt_path) / model_name 
    logger = CSVLogger(log_dir, model_name, version=finetune_data)
    ckpt_callback = ModelCheckpoint(
        save_top_k=1, save_last=True, dirpath=ckpt_dir, mode="min", monitor="val_loss"
    )

    trainer = Trainer(
        accelerator="gpu",
        logger=logger,
        callbacks=[ckpt_callback],
        max_epochs=num_epochs,
        strategy="auto",
        devices=gpus,
        check_val_every_n_epoch=2,
    )

    pipeline = SimpleLightningPipeline(
        model=model,
        trainer=trainer,
        log_dir=log_dir,
        save_run_status=False,
        seed=repetition,
        apply_metrics_per_sample=False,
        classification_metrics=metrics,
    )

    pipeline.run(data_module, task="evaluate")

if __name__ == "__main__":
    main(
        model_name="V0_pre_seam_ai_N_train_seam_ai_N_cap_100%",
        ckpt_file="ckpt/train/0/V0_pre_seam_ai_N_train_seam_ai_N_cap_100%/seam_ai_N/epoch=3-step=560.ckpt",
        pretrain_data="seam_ai",
        finetune_data="f3",
        data_path='/workspaces/shared_data/seam_ai_datasets/seam_ai_N',
        num_epochs=20,
        batch_size=8,
        repetition=8,
        ckpt_path="./ckpt",
        logs_path="./logs",
        gpus=[0],
    )




