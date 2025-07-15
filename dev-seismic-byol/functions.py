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
import torch.nn as nn

from pathlib import Path
import re


def get_model(pretrain_data, learning_rate, freeze, repetition, root_path=None):
    num_classes = 6

    if pretrain_data == "a700":
        base_name = f"V{repetition}_pretrain_{pretrain_data}_In256_B32_E100"
    else:
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
        logger.info(f"{pretrain_data} backbone loaded")

    elif pretrain_data == "imagenet":
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        backbone = resnet50(
            replace_stride_with_dilation=[False, True, True], weights=weights
        )
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        logger.info("IMAGENET backbone loaded")

    elif pretrain_data == "coco":
        backbone = torch.hub.load(
            "pytorch/vision:v0.10.0", "deeplabv3_resnet50", pretrained=True
        ).backbone
        logger.info("COCO backbone loaded")

    elif pretrain_data == "sup":
        backbone = DeepLabV3Backbone(num_classes=num_classes)
        logger.info("No model loaded!")

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


def get_eval_model(pretrain_data, import_path, learning_rate):

    num_classes = 6

    seg_data = [
        "f3",
        "f3_N",
        "seam_ai",
        "seam_ai_N",
        "both",
        "both_N",
        "s0",
        "a700",
        "sup",
        "seg",
    ]

    if pretrain_data in seg_data:
        backbone = DeepLabV3Backbone(num_classes=num_classes)

    elif pretrain_data == "imagenet":
        backbone = resnet50(replace_stride_with_dilation=[False, True, True])
        backbone = nn.Sequential(*list(backbone.children())[:-2])

    elif pretrain_data == "coco":
        backbone = torch.hub.load(
            "pytorch/vision:v0.10.0", "deeplabv3_resnet50"
        ).backbone

    else:
        raise KeyError("Pretrain data value wrong!")

    model = DeepLabV3(
        backbone=backbone,
        learning_rate=learning_rate,
        num_classes=num_classes,
    )

    model = FromPretrained(
        model=model, ckpt_path=import_path, strict=False, error_on_missing_keys=False
    )

    return model


def extract_epoch_number(filename):
    match = re.search(r"epoch=(\d+)", filename)
    return int(match.group(1)) if match else -1


def get_models_files(base_dir="./ckpt/train", target_repetition=None):
    base_dir = Path(base_dir)
    results = []
    repetitions = (
        [str(target_repetition)]
        if target_repetition != None
        else [d.name for d in base_dir.iterdir() if d.is_dir()]
    )

    for repetition_dir in repetitions:
        rep_path = base_dir / repetition_dir
        if not rep_path.is_dir():
            continue

        for model_dir in rep_path.iterdir():
            if not model_dir.is_dir():
                continue

            match = re.match(r"V(\d+)_pre_(.+?)_train_(.+?)_cap_(.+)", model_dir.name)
            if not match:
                continue

            _, pretrain_data, train_data, cap = match.groups()

            for train_data_dir in model_dir.iterdir():
                if not train_data_dir.is_dir():
                    continue

                ckpt_files = [
                    f
                    for f in train_data_dir.iterdir()
                    if f.is_file() and f.name.startswith("epoch=")
                ]
                if ckpt_files:
                    ckpt_files.sort(
                        key=lambda f: extract_epoch_number(f.name), reverse=True
                    )
                    results.append(
                        {
                            "model_name": model_dir.name,
                            "repetition": repetition_dir,
                            "pretrain_data": pretrain_data,
                            "train_data": train_data,
                            "cap": cap,
                            "ckpt_file": str(ckpt_files[0]),
                        }
                    )

    return results


from torch.utils.data import Subset
from math import ceil, floor


def build_indices(start, end, size, sort=False, original_size=None):
    if original_size is None:
        original_size = size

    if (end - start <= 0) or (size <= 0):
        return []

    midpoint = (end + start) // 2
    r = [midpoint]

    remainder = size - 1
    left_apportion = ceil(remainder / 2)
    right_apportion = floor(remainder / 2)

    r += build_indices(start, midpoint, left_apportion, sort, original_size)
    r += build_indices(midpoint + 1, end, right_apportion, sort, original_size)

    if sort:
        r.sort()

    # Duplicar somente se for a chamada inicial e size original era 1
    if original_size == 1:
        return r * 2

    return r


class BinaryTreeSubset(Subset):
    def __init__(self, dataset, size):
        indices = build_indices(0, len(dataset), size, sort=True)
        super().__init__(dataset, indices)


from minerva.data.data_modules.base import MinervaDataModule
from typing import Optional, Union
import random


class CapDataModule(MinervaDataModule):
    def __init__(
        self,
        cap_train: Optional[Union[float, int]] = None,
        cap_val: Optional[Union[float, int]] = None,
        cap_test: Optional[Union[float, int]] = None,
        seed: Optional[int] = 42,
        drop_last: Optional[bool] = False,
        *args,
        **kwargs,
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
            if isinstance(self.cap_train, float):
                cap_len = int(len(dataloader.dataset) * self.cap_train)
                subset, _ = torch.utils.data.random_split(
                    dataloader.dataset,
                    [cap_len, len(dataloader.dataset) - cap_len],
                    generator=torch.Generator().manual_seed(self.seed),
                )
            elif isinstance(self.cap_train, int):
                subset = BinaryTreeSubset(dataloader.dataset, self.cap_train)
            else:
                raise TypeError("cap_train must be float or int.")
            return torch.utils.data.DataLoader(
                subset,
                batch_size=dataloader.batch_size,
                shuffle=True,
                num_workers=15,
                pin_memory=dataloader.pin_memory,
                drop_last=self.drop_last,
            )
        return dataloader

    def val_dataloader(self):
        dataloader = super().val_dataloader()
        if self.cap_val is not None:
            if isinstance(self.cap_val, float):
                cap_len = int(len(dataloader.dataset) * self.cap_val)
                subset, _ = torch.utils.data.random_split(
                    dataloader.dataset,
                    [cap_len, len(dataloader.dataset) - cap_len],
                    generator=torch.Generator().manual_seed(self.seed),
                )
            elif isinstance(self.cap_val, int):
                subset = BinaryTreeSubset(dataloader.dataset, self.cap_val)
            else:
                raise TypeError("cap_val must be float or int.")
            return torch.utils.data.DataLoader(
                subset,
                batch_size=dataloader.batch_size,
                shuffle=False,
                num_workers=15,
                pin_memory=dataloader.pin_memory,
                drop_last=self.drop_last,
            )
        return dataloader

    def test_dataloader(self):
        dataloader = super().test_dataloader()
        if self.cap_test is not None:
            if isinstance(self.cap_test, float):
                cap_len = int(len(dataloader.dataset) * self.cap_test)
                subset, _ = torch.utils.data.random_split(
                    dataloader.dataset,
                    [cap_len, len(dataloader.dataset) - cap_len],
                    generator=torch.Generator().manual_seed(self.seed),
                )
            elif isinstance(self.cap_test, int):
                subset = BinaryTreeSubset(dataloader.dataset, self.cap_test)
            else:
                raise TypeError("cap_test must be float or int.")
            return torch.utils.data.DataLoader(
                subset,
                batch_size=dataloader.batch_size,
                shuffle=False,
                num_workers=15,
                pin_memory=dataloader.pin_memory,
                drop_last=self.drop_last,
            )
        return dataloader
