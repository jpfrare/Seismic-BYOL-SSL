from torch.utils.data import Dataset, ConcatDataset, DataLoader
import lightning as L
from typing import Literal, Optional, Callable, Union
import numpy as np
from pathlib import Path
import pandas as pd


class A700File(Dataset):
    def __init__(
        self,
        root_path: Union[str, Path],
        partition: Literal["train", "val", "both"],
        subset: Literal["iline", "xline", "both"],
        normalization_strategy: Literal["none", "z-volume", "z-sample", "n-sample"],
        mean: Optional[float] = None,
        std: Optional[float] = None,
    ):
        self.mean = mean
        self.std = std
        self.normalization_strategy = normalization_strategy

        if normalization_strategy == "z-volume" and (mean is None or std is None):
            raise ValueError("Mean and std must be provided for z-volume normalization")

        if partition == "train":
            condition = lambda x: (x % 100) < 90
        elif partition == "val":
            condition = lambda x: (x % 100) >= 90
        else:
            condition = lambda _: True

        root_path = Path(root_path)
        if subset != "both":
            self.files = sorted(root_path.glob(f"{subset}/**/*.npy"))
            self.files = [f for i, f in enumerate(self.files) if condition(i)]
        else:
            i_files = sorted(root_path.glob(f"iline/**/*.npy"))
            x_files = sorted(root_path.glob(f"xline/**/*.npy"))
            i_files = [f for i, f in enumerate(i_files) if condition(i)]
            x_files = [f for i, f in enumerate(x_files) if condition(i)]
            self.files = i_files + x_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i) -> np.ndarray:
        image = np.load(self.files[i])

        if self.normalization_strategy == "z-volume":
            image = (image - self.mean) / self.std
        elif self.normalization_strategy == "z-sample":
            image = (image - image.mean()) / image.std()
        elif self.normalization_strategy == "n-sample":
            image = (image - image.min()) / (image.max() - image.min())

        return np.expand_dims(image, axis=0)  # Add channel dimension


class A700Dataset(Dataset):
    def __init__(
        self,
        root_path: Union[str, Path],
        partition: Literal["train", "val", "both"],
        subset: Literal["iline", "xline", "both"],
        normalization_strategy: Literal["none", "z-volume", "z-sample", "n-sample"],
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        self.transform = transform or (lambda x: x)
        root_path = Path(root_path)
        
        # self.stats = pd.read_csv(root_path / "stats.csv").set_index("directory")
        
        df = pd.read_csv(root_path / "stats.csv")
        df["directory"] = df["directory"].str.replace("file_", "", regex=False).astype(int)
        self.stats = df.set_index("directory")
        
        self.concatenation = ConcatDataset([
            A700File(
                root_path / f"file_{i}",
                partition,
                subset,
                normalization_strategy,
                row["mean"],
                row["std"],
            )
            for i, row in self.stats.iterrows()
        ])

    def __len__(self):
        return len(self.concatenation)

    def __getitem__(self, i) -> np.ndarray:
        image = self.concatenation[i]
        return self.transform(image) if self.transform else image


class A700DataModule(L.LightningDataModule):
    def __init__(
        self,
        subset: Literal["iline", "xline", "both"],
        normalization_strategy: Literal["none", "z-volume", "z-sample", "n-sample"],
        crop_size: int,
        batch_size: int = 32,
        num_workers: int = 4,
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        root: str = "/parceirosbr/asml/datasets/a700"
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform or (lambda x: x[:, :crop_size, :crop_size])
        self.root = root

        self.train_dataset = A700Dataset(
            self.root, "train", subset, normalization_strategy, self.transform
        )
        self.val_dataset = A700Dataset(
            self.root, "val", subset, normalization_strategy, self.transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
