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


def main(
    pretrain_data,
    finetune_data,
    data_path,
    num_epochs,
    batch_size,
    repetition,
    learning_rate,
    cap,
    freeze,
    ckpt_path,
    logs_path,
    import_root_path,
    gpus,
):

    # Set general seed**
    torch.manual_seed(repetition)
    save_name = (
        f"V{repetition}_pre_{pretrain_data}_train_{finetune_data}_cap_{cap*100:.0f}%"
    )

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

    train_dataset = SimpleDataset(
        readers=[
            train_data_reader,
            train_label_reader,
        ],
        transforms=padding,
        return_single=False,
    )

    assert (
        len(train_dataset) * cap >= batch_size
    ), "Too few samples for given cap and batch size"

    val_dataset = SimpleDataset(
        readers=[
            val_data_reader,
            val_label_reader,
        ],
        transforms=padding,
        return_single=False,
    )

    # DataModule

    data_module = CapDataModule(
        cap_train=cap,
        cap_val=1,
        cap_test=1,
        seed=repetition,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle_train=True,
    )

    # Model

    model = get_model(
        pretrain_data,
        learning_rate,
        freeze,
        repetition,
        import_root_path,
    )

    log_dir = Path(logs_path) / save_name / finetune_data
    ckpt_dir = Path(ckpt_path) / save_name / finetune_data
    logger = CSVLogger(log_dir, name=save_name, version=finetune_data)
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
        save_run_status=True,
    )

    pipeline.run(data_module, task="fit")


if __name__ == "__main__":
    main(
        pretrain_data="f3",
        finetune_data="f3",
        data_path="/workspaces/shared_data/seismic/f3_segmentation",
        num_epochs=20,
        batch_size=8,
        repetition=0,
        learning_rate=0.001,
        cap=0.01,
        freeze=False,
        ckpt_path="/workspaces/Seismic-Byol/dev-seismic-byol/ckpt",
        logs_path="/workspaces/Seismic-Byol/dev-seismic-byol/logs",
        import_root_path=None,
        gpus=[0],
    )
