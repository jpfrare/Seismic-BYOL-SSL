from torch.utils.data import Dataset, ConcatDataset
import lightning as L
from torch.utils.data import DataLoader
from typing import Literal, Optional, Callable
import torch
import torchvision.transforms as T
from minerva import transforms as Tm
import numpy as np
from pathlib import Path
import pandas as pd


class A700File(Dataset):

    def __init__(
        self,
        root_path: str | Path,
        partition: Literal["train", "val", "both"],
        subset: Literal["iline", "xline", "both"],
        normalization_strategy: Literal["none", "z-volume", "z-sample", "n-sample"],
        mean: Optional[float] = None,
        std: Optional[float] = None,
    ):
        self.mean = mean
        self.std = std

        root_path = Path(root_path)

        if (normalization_strategy == "z-volume") and (mean is None) and (std is None):
            raise ValueError()

        self.normalization_strategy = normalization_strategy

        if partition == "train":
            condition = lambda x: (x % 100) < 90
        elif partition == "val":
            condition = lambda x: (x % 100) >= 90
        else:
            condition = lambda _: True

        if subset != "both":
            self.files = sorted(root_path.glob(f"{subset}/**/*.npy"))
            self.files = [file for i, file in enumerate(self.files) if condition(i)]
        else:
            i_files = sorted(root_path.glob(f"iline/**/*.npy"))
            x_files = sorted(root_path.glob(f"xline/**/*.npy"))
            i_files = [file for i, file in enumerate(i_files) if condition(i)]
            x_files = [file for i, file in enumerate(x_files) if condition(i)]
            self.files = i_files + x_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        # image = torch.from_numpy(np.load(self.files[i]))
        image = np.load(self.files[i])
        if self.normalization_strategy == "z-volume":
            return (image - self.mean) / self.std
        elif self.normalization_strategy == "z-sample":
            mean = image.mean()
            std = image.std()
            return (image - mean) / std
        elif self.normalization_strategy == "n-sample":
            max = image.max()
            min = image.min()
            return (image - min) / (max - min)

        return image


class A700Dataset(Dataset):

    def __init__(
        self,
        root_path: str | Path,
        partition: Literal["train", "val", "both"],
        subset: Literal["iline", "xline", "both"],
        normalization_strategy: Literal["none", "z-volume", "z-sample", "n-sample"],
        transform: Optional[Callable] = None,
    ):
        self.transform = transform or (lambda x: x)
        root_path = Path(root_path)
        self.normalization_strategy = normalization_strategy

        self.stats = pd.read_csv(root_path / "stats.csv").set_index("directory")

        self.concatenation = ConcatDataset(
            A700File(
                root_path / index,  # type: ignore
                partition,
                subset,
                normalization_strategy,
                row["mean"],
                row["std"],
            )
            for index, row in self.stats.iterrows()
        )

    def __len__(self):
        return len(self.concatenation)

    def __getitem__(self, i):
        image: torch.Tensor = self.concatenation[i]  # type: ignore
        return self.transform(image) if self.transform else image


class A700DataModule(L.LightningDataModule):

    def __init__(
        self,
        subset: Literal["iline", "xline", "both"],
        normalization_strategy: Literal["none", "z-volume", "z-sample", "n-sample"],
        crop_size: int,
        channels: int,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        transf = Tm.TransformPipeline(
            [
                T.RandomCrop((crop_size, crop_size)),  # type: ignore
                Tm.Repeat(0, channels),
            ]
        )

        self.train_dataset = A700Dataset(
            "/parceirosbr/asml/datasets/a700",
            "train",
            subset,
            normalization_strategy,
            transf,
        )

        self.val_dataset = A700Dataset(
            "/parceirosbr/asml/datasets/a700",
            "val",
            subset,
            normalization_strategy,
            transf,
        )

    def train_dataloader(self):
        """Return the training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        """Return the validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


class A150DataModule(L.LightningDataModule):

    def __init__(
        self,
        root: str,
        subset: Literal["iline", "xline", "both"] = "both",
        transforms: Optional[Tm._Transform] = None,
        batch_size: int = 128,
        num_workers: int = 30,
        drop_last: bool = False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last

        self.train_dataset = A700Dataset(
            root,
            "train",
            subset,
            "z-sample",
            transforms,
        )

        self.val_dataset = A700Dataset(
            root,
            "val",
            subset,
            "z-sample",
            transforms,
        )

    def train_dataloader(self):
        """Return the training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=self.drop_last
        )

    def val_dataloader(self):
        """Return the validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=self.drop_last
        )