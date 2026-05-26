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
        num_workers: int = os.cpu_count() if os.cpu_count() < 24 else 24,
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
            additional_train_dataloader_kwargs={'pin_memory':True},
            additional_val_dataloader_kwargs={'pin_memory':True}, 
            additional_test_dataloader_kwargs={'pin_memory':True},
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