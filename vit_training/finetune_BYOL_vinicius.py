from common import get_data_module, get_trainer_pipeline
import torch
from minerva.models.ssl.byol import BYOL
from functools import partial
import os

import torchvision.models.segmentation as models
from minerva.models.nets.image.deeplabv3 import DeepLabV3Backbone
import os
from torch import nn

from minerva.models.nets.image.deeplabv3 import DeepLabV3, DeepLabV3PredictionHead
from torch import Tensor
from collections import OrderedDict


root_data_dir = "/workspaces/shared_data/seam_ai_datasets/seam_ai/images"
root_annotation_dir = "/workspaces/shared_data/seam_ai_datasets/seam_ai/annotations"

print(os.path.exists(root_data_dir))
print(os.path.exists(root_annotation_dir))

img_size = (1008, 784)  # Change this to the size of the images in the dataset
model_name = "byol"  # Model name (just identifier)
dataset_name = "seam_ai"  # Dataset name (just identifier)
single_channel = False  # If True, the model will be trained with single channel images (instead of 3 channels)

log_dir = "./logs_byol"  # Directory to save logs
batch_size = 4  # Batch size
seed = 42  # Seed for reproducibility
num_epochs = 75  # Number of epochs to train
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

print(data_module)


# Just to check if the data module is working
data_module.setup("fit")
train_batch_x, train_batch_y = next(iter(data_module.train_dataloader()))


print(train_batch_x.shape)
print(train_batch_y.shape)


# wheights_path = "/workspaces/HIAAC-KR-Dev-Container/shared_data/notebooks_e_pesos/backbones_byol/V1/V1_E300_B32_S256_both_N.pth"
wheights_path = "/workspaces/shared_data/notebooks_e_pesos/backbones_byol/V1/V1_E300_B32_S256_seam_ai.pth"
print(os.path.exists(wheights_path))

backbone = models.deeplabv3_resnet50().backbone

print(backbone.load_state_dict(torch.load(wheights_path)))


class DeepLabV3_2(DeepLabV3):

    def forward(self, x: Tensor) -> Tensor:
        x = x.float()
        input_shape = x.shape[-2:]
        h = self.backbone(x)
        if isinstance(h, OrderedDict):
            h = h["out"]
        z = self.fc(h)
        # Upscaling
        return nn.functional.interpolate(
            z, size=input_shape, mode="bilinear", align_corners=False
        )


from minerva.models.nets.image.deeplabv3 import DeepLabV3, DeepLabV3PredictionHead

pred_head = DeepLabV3PredictionHead(num_classes=6)

model = DeepLabV3_2(
    backbone=backbone,
    pred_head=pred_head,
    loss_fn=torch.nn.CrossEntropyLoss(),
    num_classes=6,
    learning_rate=0.001,
)


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
