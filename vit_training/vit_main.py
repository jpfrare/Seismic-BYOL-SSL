from common import get_data_module
from common import get_trainer_pipeline
import torch
from minerva.models.nets.image.vit import SFM_BasePatch16_Downstream
from functools import partial
from minerva.models.loaders import FromPretrained


def main():

    print("********* Starting Main *********")

    root_data_dir = "/workspaces/shared_data/seam_ai_datasets/seam_ai/images"
    root_annotation_dir = "/workspaces/shared_data/seam_ai_datasets/seam_ai/annotations"
    img_size = (512, 512)  # Change this to the size of the images in the dataset
    model_name = "sfm-base-patch16-E100-B8"  # Model name (just identifier)
    dataset_name = "seam_ai"  # Dataset name (just identifier)
    single_channel = True  # If True, the model will be trained with single channel images (instead of 3 channels)

    log_dir = "./logs"  # Directory to save logs
    batch_size = 8  # Batch size
    seed = 42  # Seed for reproducibility
    num_epochs = 100  # Number of epochs to train
    is_debug = False  # If True, only 3 batch will be processed for 3 epochs
    accelerator = "gpu"  # CPU or GPU
    devices = 1  # Num GPUs

    data_module = get_data_module(
        root_data_dir=root_data_dir,
        root_annotation_dir=root_annotation_dir,
        img_size=img_size,
        batch_size=batch_size,
        seed=seed,
        single_channel=single_channel,
    )

    data_module.setup("fit")

    model = SFM_BasePatch16_Downstream(img_size=(512, 512), num_classes=6, in_chans=1)

    pipeline = get_trainer_pipeline(
        model=model,
        model_name=model_name,
        dataset_name=dataset_name,
        log_dir=log_dir,
        num_epochs=num_epochs,
        accelerator=accelerator,
        devices=devices,
        is_debug=is_debug,
        seed=seed,
    )

    pipeline.run(data_module, task="fit")

    print(f"Checkpoint saved at {pipeline.trainer.checkpoint_callback.last_model_path}")


if __name__ == "__main__":
    main()
