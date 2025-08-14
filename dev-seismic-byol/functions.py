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
import torch.nn as nn

from pathlib import Path
import re

import torch
import torch.nn as nn

from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights


class LinearSegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.linear_head = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.linear_head(x)


def get_state_dict(model):
    state_dict = model.state_dict()
    renamed_state_dict = {}
    for key in state_dict.keys():
        # Replace the key prefix to match my_state_dict
        new_key = f"RN50model.{key}" if not key.startswith("RN50model.") else key
        renamed_state_dict[new_key] = state_dict[key]

    return renamed_state_dict    


def get_seg_file(root_path="/home/vinicius.soares/Seismic-Byol/dev-seismic-byol/ckpt/train", 
                 repetition=0,
                 seg_data="seam_ai_N",):
    
    folder_path = f"{root_path}/{repetition}/V{repetition}_pre_sup_train_{seg_data}_cap_100%/{seg_data}"
    
    files = [f for f in Path(folder_path).iterdir() if f.is_file() and f.name.startswith("epoch=")]
    if files:
        files.sort(key=lambda f: extract_epoch_number(f.name))
        selected_file = files[0]
        return str(selected_file.resolve())

    return None


def get_model(pretrain_data, learning_rate, freeze, repetition, root_path=None, full_path=False, linear=False, finetune_data=None):
    num_classes = 6

    if full_path: 
        import_path = full_path
    else:
        if pretrain_data == "a700":
            base_name = f"V{repetition}_pretrain_{pretrain_data}_In256_B32_E200_lr1e-05"
        else:
            base_name = f"V{repetition}_pretrain_{pretrain_data}_In256_B32_E1200_lr1e-05"

        if root_path:
            import_path = f"{root_path}/{repetition}/{base_name}/{pretrain_data}/last.ckpt"
        else:
            import_path = (
                f"ckpt/pretrain/{repetition}/{base_name}/{pretrain_data}/last.ckpt"
            )
            
    if pretrain_data == "seg":
        seg_data_mapping = {
            "f3_N": "seam_ai_N",
            "f3": "seam_ai",
            "seam_ai_N": "f3_N",
            "seam_ai": "f3",
        }
        root_path = "/home/vinicius.soares/Seismic-Byol/dev-seismic-byol/ckpt/train"
        if finetune_data in ["f3_N", "seam_ai_N", "f3", "seam_ai"]:
            import_path = get_seg_file(
                root_path=root_path,
                repetition=repetition,
                seg_data=seg_data_mapping[finetune_data],
            )
            logger.info(f"CKPT imported from:\n{import_path}")
            logger.info(f"Imported pretrain for segmentation in {seg_data_mapping[finetune_data]}")
        else:
            raise ValueError("Invalid finetune_data for pretrain_data 'seg'")

    seg_data = ["f3", "f3_N", "seam_ai", "seam_ai_N", "both", "both_N", "s0", "a700"]

    resnet50_backbone = DeepLabV3Backbone(num_classes=num_classes)

    if pretrain_data in seg_data:
        # Builds the BYOL model, loads the weights and extracts the backbone
        model = BYOL(backbone=resnet50_backbone, learning_rate=learning_rate)
        resnet50_backbone = FromPretrained(
            model=model,
            ckpt_path=import_path,
            strict=False,
            error_on_missing_keys=False,
        ).backbone
        logger.info(f"{pretrain_data} backbone loaded")

    elif pretrain_data == "imagenet":
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        imagenet_backbone = resnet50(
            replace_stride_with_dilation=[False, True, True], weights=weights
        )
        imagenet_state_dict = get_state_dict(imagenet_backbone)
        resnet50_backbone.load_state_dict(imagenet_state_dict, strict=False)
        logger.info("IMAGENET backbone loaded")
        
    elif pretrain_data == "coco":      
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        coco_backbone = deeplabv3_resnet50(weights=weights).backbone
        coco_state_dict = get_state_dict(coco_backbone)
        resnet50_backbone.load_state_dict(coco_state_dict, strict=False)
        logger.info("COCO backbone loaded")

    elif pretrain_data == "sup":
        logger.info("No model loaded!")
        
    elif pretrain_data == "seg":
        model = DeepLabV3(
            backbone=resnet50_backbone,
            learning_rate=learning_rate,
            num_classes=num_classes,
            freeze_backbone=False
        )

        resnet50_backbone = FromPretrained(
            model=model, ckpt_path=import_path, strict=False, error_on_missing_keys=False
        ).backbone
        
        logger.info("Default model loaded (seg)")
        
    elif pretrain_data == "teste":
        resnet50_backbone = deeplabv3_resnet50().backbone
        logger.info("Imported backbone from scratch loaded")
    
    else:
        raise KeyError("Pretrain data value wrong!")

    if linear:
        pred_head = LinearSegmentationHead(
            in_channels=2048,
            num_classes=num_classes
        )

        model = DeepLabV3(
            backbone=resnet50_backbone,
            pred_head=pred_head,
            learning_rate=learning_rate,
            num_classes=num_classes,
            freeze_backbone=freeze
        )
        
    else:
        model = DeepLabV3(
            backbone=resnet50_backbone,
            learning_rate=learning_rate,
            num_classes=num_classes,
            freeze_backbone=freeze
        )
    
    if freeze:
        logger.info("Freezing backbone parameters via `freeze_weights`.")

    return model


def get_eval_model(pretrain_data, import_path, learning_rate, linear=False):

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
        "coco",
        "imagenet",
        "seg"
    ]

    if pretrain_data in seg_data:
        resnet50_backbone = DeepLabV3Backbone(num_classes=num_classes)
        
    elif pretrain_data == "teste":
        resnet50_backbone = deeplabv3_resnet50().backbone
        logger.info("Imported backbone from sratch loaded")

    else:
        raise KeyError("Pretrain data value wrong!")
    
    if linear:
        pred_head = LinearSegmentationHead(
            in_channels=2048,
            num_classes=num_classes
        )

        model = DeepLabV3(
            backbone=resnet50_backbone,
            pred_head=pred_head,
            learning_rate=learning_rate,
            num_classes=num_classes,
            freeze_backbone=False
        )
        
    else:
        model = DeepLabV3(
            backbone=resnet50_backbone,
            learning_rate=learning_rate,
            num_classes=num_classes,
            freeze_backbone=False
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


from minerva.data.data_modules.base import MinervaDataModule
from minerva.data.datasets.binary_tree_subset import BinaryTreeSubset
from typing import Optional, Union, Literal
import random
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from minerva.data.datasets.base import SimpleDataset
from minerva.data.readers import TiffReader, PNGReader
import lightning as L
import os


class SeismicReducibleDataset(Dataset):

    def __init__(self, root: Path, size: int, transform = None):
        assert size > 0, f"`size` must be a positive integer, but got size = {size}"

        self.root = Path(root)

        xl_set = SimpleDataset(
            [
                TiffReader(
                    self.root / "images/train",
                    ["text", "numeric"],
                    "_",
                    [0, 1],
                    False,
                    r"xl.*",
                ),
                PNGReader(
                    self.root / "annotations/train",
                    ["text", "numeric"],
                    "_",
                    [0, 1],
                    False,
                    r"xl.*",
                ),
            ]
        )

        il_set = SimpleDataset(
            [
                TiffReader(
                    self.root / "images/train",
                    ["text", "numeric"],
                    "_",
                    [0, 1],
                    False,
                    r"il.*",
                ),
                PNGReader(
                    self.root / "annotations/train",
                    ["text", "numeric"],
                    "_",
                    [0, 1],
                    False,
                    r"il.*",
                ),
            ]
        )

        max_size = len(xl_set) + len(il_set)
        assert max_size >= size, f"There are only {max_size} samples in the dataset but got size = {size}"

        xl_size = min(size // 2, len(xl_set))
        il_size = min(size - xl_size, len(il_set))

        sets = []
        if xl_size > 0:
            sets.append(BinaryTreeSubset(xl_set, xl_size))
        if il_size > 0:
            sets.append(BinaryTreeSubset(il_set, il_size))

        self.data: Dataset = ConcatDataset(sets)
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.data[index]
        return self.transform(image), self.transform(label)

    def __len__(self):
        return len(self.data)


class SeismicFullDataset(SimpleDataset):

    def __init__(
        self,
        root: Path,
        partition: Literal["val", "test", "train"],
        transform,
    ):
        self.root = Path(root)
        super().__init__(
            [
                TiffReader(self.root / f"images/{partition}"),
                PNGReader(self.root / f"annotations/{partition}"),
            ],
            transforms=transform
        )


class SeismicDataModule(MinervaDataModule):
    def __init__(
        self,
        root: Path,
        batch_size: int = 32,
        num_workers: int = os.cpu_count(),
        cap: int = 256,
        drop_last: bool = False,
        train_dataset=None,
        val_dataset=None,
        test_dataset=None,
        transform=None,
        test_transform=None,
        *args,
        **kwargs,
    ):
        # Defina os datasets caso não tenham sido passados
        train_dataset = train_dataset or SeismicReducibleDataset(
            root=root, size=cap, transform=transform
        )
        val_dataset = val_dataset or SeismicFullDataset(
            root=root, partition="val", transform=test_transform
        )
        test_dataset = test_dataset or SeismicFullDataset(
            root=root, partition="test", transform=test_transform
        )

        super().__init__(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=drop_last,
            *args,
            **kwargs,
        )

        # Nenhuma redefinição dos dataloaders é necessária


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
                # drop_last=True,
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


import os

def get_dataset_mapping():
    
    nodename = os.uname().nodename
    
    if 'sdumont' in nodename:
        dataset_mapping = {
            'seam_ai_N':'/workspaces/shared_data/seam_ai_datasets/seam_ai_N/images',
            'seam_ai':'/workspaces/shared_data/seam_ai_datasets/seam_ai/images',
            'f3':'/workspaces/shared_data/seismic/f3_segmentation/images',
            'f3_N':'/workspaces/shared_data/seismic/f3_segmentation_N/images',
            'both':'/workspaces/shared_data/seismic/both/images',
            'both_N':'/workspaces/shared_data/seismic/both_N/images',
        }
    
    elif 'node' in nodename:
        dataset_mapping = {
            'seam_ai_N':'/workspaces/shared_data/seam_ai_datasets/seam_ai_N/images',
            'seam_ai':'/workspaces/shared_data/seam_ai_datasets/seam_ai/images',
            'f3':'/workspaces/shared_data/seismic/f3_segmentation/images',
            'f3_N':'/workspaces/shared_data/seismic/f3_segmentation_N/images',
            'both':'/workspaces/shared_data/seismic/both/images',
            'both_N':'/workspaces/shared_data/seismic/both_N/images',
        }
        
    elif 'c' in nodename:
        dataset_mapping = {
            'seam_ai_N':'/home/vinicius.soares/asml/datasets/tiff_data/seam_ai_N/images',
            'seam_ai':'/home/vinicius.soares/asml/datasets/tiff_data/seam_ai/images',
            'f3':'/home/vinicius.soares/asml/datasets/tiff_data/f3_segmentation/images',
            'f3_N':'/home/vinicius.soares/asml/datasets/tiff_data/f3_segmentation_N/images',
            'both':'/home/vinicius.soares/asml/datasets/tiff_data/both/images',
            'both_N':'/home/vinicius.soares/asml/datasets/tiff_data/both_N/images',
            'a700':'/parceirosbr/asml/datasets/a700',
        }
    else:
        raise RuntimeError(f"Unsupported nodename '{nodename}'. Unable to determine dataset mapping.")
    
    return dataset_mapping