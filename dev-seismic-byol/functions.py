import matplotlib.pyplot as plt


def plot_images(
    images,
    plot_title=None,
    subplot_titles=None,
    cmaps=None,
    filename=None,
    x_label=None,
    y_label=None,
    height=5,
    width=5,
    show=False,
):
    num_images = len(images)

    # Create a figure with subplots (1 row, num_images columns), adjusting size based on height and width parameters
    fig, axs = plt.subplots(1, num_images, figsize=(width * num_images, height))

    if num_images == 1:
        axs = [axs]

    # Set overall plot title if provided
    if plot_title is not None:
        fig.suptitle(plot_title, fontsize=12)

    # Ensure subplot_titles and cmaps are lists with correct lengths
    if subplot_titles is None:
        subplot_titles = [None] * num_images
    if cmaps is None:
        cmaps = ["gray"] * num_images

    # Plot each image in its respective subplot
    for i, (img, ax, title, cmap) in enumerate(zip(images, axs, subplot_titles, cmaps)):
        im = ax.imshow(img, cmap=cmap)

        # Set title for each subplot if provided
        if title is not None:
            ax.set_title(title)

        # Add a colorbar for each subplot
        # fig.colorbar(im, ax=ax)

        # Set x and y labels if provided
        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)

    # Adjust layout to fit titles, labels, and colorbars
    plt.tight_layout()

    # Save the figure if filename is provided
    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")
        print(f"Figure saved as '{filename}'")

    # Show the plot
    if show:
        plt.show()
    else:
        plt.close()

    return fig


import logging


def get_logger(name: str = __name__) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


logger = get_logger("minerva")


from minerva.models.ssl.byol import BYOL
from minerva.models.nets.image.deeplabv3 import DeepLabV3Backbone
from minerva.models.loaders import FromPretrained
from torchvision.models.resnet import resnet50
from minerva.models.nets.image.deeplabv3 import DeepLabV3

from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
import torchvision.models
import torch
import torchmetrics


def get_model(pretrain_data, learning_rate, freeze, repetition, root_path=None):
    num_classes = 6
    base_name = f"V{repetition}_pretrain_{pretrain_data}_In256_B32_E500"

    if root_path:
        import_path = f"{root_path}/{repetition}/{base_name}/{pretrain_data}/last.ckpt"
    else:
        import_path = (
            f"ckpt/pretrain/{repetition}/{base_name}/{pretrain_data}/last.ckpt"
        )

    seg_data = ["f3", "f3_N", "seam_ai", "seam_ai_N", "both", "both_N", "s0", "a700"]

    if pretrain_data in seg_data:
        backbone = DeepLabV3Backbone(num_classes=num_classes)
        model = BYOL(backbone=backbone, learning_rate=learning_rate)

        backbone = FromPretrained(
            model=model,
            ckpt_path=import_path,
            strict=False,
            error_on_missing_keys=False,
            # keys_to_rename={"": "backbone."},
        ).backbone

    elif pretrain_data == "imagenet":
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        backbone = resnet50(
            replace_stride_with_dilation=[False, True, True], weights=weights
        )

    elif pretrain_data == "coco":
        backbone = torch.hub.load(
            "pytorch/vision:v0.10.0", "deeplabv3_resnet50", pretrained=True
        ).backbone

    elif pretrain_data == "sup":
        logger.info("No model loaded!")
        backbone = DeepLabV3Backbone(num_classes=num_classes)

    else:
        raise KeyError("Pretrain data value wrong!")

    if freeze:
        logger.info("Freezing backbone parameters.")
        for param in backbone.parameters():
            param.requires_grad = False

    # Métricas
    iou = torchmetrics.JaccardIndex(
        task="multiclass", num_classes=num_classes, average=None
    )
    f1_score = torchmetrics.F1Score(
        task="multiclass", num_classes=num_classes, average=None
    )
    accuracy = torchmetrics.Accuracy(
        task="multiclass", num_classes=num_classes, average=None
    )

    metrics = {
        "iou": iou,
        "f1_score": f1_score,
        "accuracy": accuracy,
    }

    model = DeepLabV3(
        backbone=backbone,
        learning_rate=learning_rate,
        num_classes=num_classes,
    )

    return model


from minerva.data.data_modules.base import MinervaDataModule
from typing import Optional
import random


class CapDataModule(MinervaDataModule):
    def __init__(
        self, 
        cap_train: Optional[float] = None, 
        cap_val: Optional[float] = None, 
        cap_test: Optional[float] = None, 
        seed: Optional[int] = 42,
        drop_last: Optional[bool] = False,
        *args, 
        **kwargs    
    ):
        super().__init__(*args, **kwargs)
        self.cap_train = cap_train
        self.cap_val = cap_val
        self.cap_test = cap_test
        self.seed = seed
        self.drop_last = drop_last
        random.seed(self.seed)
        torch.manual_seed(self.seed)

    def train_dataloader(self):
        dataloader = super().train_dataloader()
        if self.cap_train is not None:
            cap_len = int(len(dataloader.dataset) * self.cap_train)
            subset, _ = torch.utils.data.random_split(
                dataloader.dataset,
                [cap_len, len(dataloader.dataset) - cap_len],
                generator=torch.Generator().manual_seed(self.seed),
            )
            return torch.utils.data.DataLoader(
                subset,
                batch_size=dataloader.batch_size,
                shuffle=True,
                num_workers=15,
                pin_memory=dataloader.pin_memory,
                drop_last=self.drop_last
            )
        return dataloader

    def val_dataloader(self):
        dataloader = super().val_dataloader()
        if self.cap_val is not None:
            cap_len = int(len(dataloader.dataset) * self.cap_val)
            subset, _ = torch.utils.data.random_split(
                dataloader.dataset,
                [cap_len, len(dataloader.dataset) - cap_len],
                generator=torch.Generator().manual_seed(self.seed),
            )
            return torch.utils.data.DataLoader(
                subset,
                batch_size=dataloader.batch_size,
                shuffle=False,
                num_workers=15,
                pin_memory=dataloader.pin_memory,
                drop_last=self.drop_last
            )
        return dataloader

    def test_dataloader(self):
        dataloader = super().test_dataloader()
        if self.cap_test is not None:
            cap_len = int(len(dataloader.dataset) * self.cap_test)
            subset, _ = torch.utils.data.random_split(
                dataloader.dataset,
                [cap_len, len(dataloader.dataset) - cap_len],
                generator=torch.Generator().manual_seed(self.seed),
            )
            return torch.utils.data.DataLoader(
                subset,
                batch_size=dataloader.batch_size,
                shuffle=False,
                num_workers=15,
                pin_memory=dataloader.pin_memory,
                drop_last=self.drop_last
            )
        return dataloader
