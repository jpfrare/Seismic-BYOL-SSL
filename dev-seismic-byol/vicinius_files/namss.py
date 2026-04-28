import numpy as np
from pathlib import Path
from typing import Optional
from minerva.data.datasets.base import SimpleDataset
from minerva.data.data_modules.base import MinervaDataModule
from minerva.data.readers.tiff_reader import TiffReader
from minerva.utils.typing import PathLike
from torch.utils.data import ConcatDataset
from minerva.transforms.transform import Repeat, CastTo, TransformPipeline, Unsqueeze


class InstanceNormalization:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply z-normalization (zero mean, unit variance) per channel.

        Parameters
        ----------
        x : np.ndarray
            Input array of shape (C, H, W). Each channel will be normalized
            independently by subtracting its mean and dividing by its standard deviation.

        Returns
        -------
        np.ndarray
            Normalized array with the same shape as the input.

        Notes
        -----
        - Mean and standard deviation are computed over the spatial dimensions (H, W)
          for each channel.
        - A small epsilon (1e-5) is added to the standard deviation to prevent division by zero.
        """
        mean = x.mean(axis=(1, 2), keepdims=True)
        std = x.std(axis=(1, 2), keepdims=True) + 1e-5
        return (x - mean) / std

    def __str__(self):
        return "InstanceNormalization()"


class NAMSSDataModule(MinervaDataModule):
    def __init__(
        self,
        root_path: PathLike,
        batch_size: int = 1,
        num_workers: int = 0,
        drop_last: bool = False,
        additional_train_dataloader_kwargs: Optional[dict] = None,
        additional_val_dataloader_kwargs: Optional[dict] = None,
        additional_test_dataloader_kwargs: Optional[dict] = None,
        shuffle_train: bool = True,
        instance_normalization: bool = True,
        transforms = None,
    ):
        """Data module for the NAMSS dataset. This creates a MinervaDataModule
        with only a training dataset by combining all subdirectories
        (e.g., train, val, test) found in the specified root_path. Each
        subdirectory is expected to contain a folder named "original" with
        TIFF image files. This data module can be used for training models on
        self-supervised tasks.

        Parameters
        ----------
        root_path : PathLike
            The root path containing subdirectories for each dataset split
            (e.g., train, val, test).
        batch_size : int, optional
            _description_, by default 1
        num_workers : int, optional
            _description_, by default 0
        batch_size : int, optional
            Default batch_size for all dataloaders, by default 1
        num_workers : int, optional
            Default num_workers for all dataloaders, by default 0
        drop_last : bool, optional
            Default drop_last for all dataloaders, by default False
        additional_train_dataloader_kwargs : Optional[dict], optional
            Override the default train dataloader kwargs, by default None
        additional_val_dataloader_kwargs : Optional[dict], optional
            Override the default val dataloader kwargs, by default None
        additional_test_dataloader_kwargs : Optional[dict], optional
            Override the default test dataloader kwargs, by default None
        shuffle_train : bool, optional
            If True, shuffle the training dataset. If False, do not shuffle the
            training dataset, by default True. By default, only the training
            dataloader is shuffled.
        instance_normalization: bool, optional
            If True, apply instance normalization to each image, by default True

        Raises
        ------
        NotADirectoryError
            If the root_path is not a directory or if any subdirectory is not a
            directory.
        """

        self.root_path = Path(root_path)
        self.datasets = []

        # Iterate over all subdirectories (train/val/test) in the root path
        for dset_path in self.root_path.iterdir():
            if not dset_path.is_dir():
                raise NotADirectoryError(f"{dset_path} is not a directory.")

            reader = TiffReader(dset_path / "original")
            
            if not transforms: 
                
                transforms = []
                # Image is 512x512, make it 1x512x512
                transforms.append(Unsqueeze(axis=0))                    
                # Repeat along channel axis to make it 3x512x512
                transforms.append(Repeat(axis=0, n_repetitions=3))      
                if instance_normalization:
                    # Apply instance normalization
                    transforms.append(InstanceNormalization())  
                # Convert to float32
                transforms.append(CastTo("float32"))  
                # Create the pipeline
                transform_pipeline = TransformPipeline(transforms)

                # Create the dataset. One reader for data, one transform pipeline.
            
            else: transform_pipeline = transforms 
            
            
            dataset = SimpleDataset(
                readers=[reader], transforms=[transform_pipeline], return_single=True
            )
            self.datasets.append(dataset)

        # Combine all datasets into one training dataset
        full_dataset = ConcatDataset(self.datasets)
        # Create the MinervaDataModule
        super().__init__(
            name="NAMSSDataModule",
            train_dataset=full_dataset,
            val_dataset=None,
            test_dataset=None,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=drop_last,
            additional_train_dataloader_kwargs=additional_train_dataloader_kwargs,
            additional_val_dataloader_kwargs=additional_val_dataloader_kwargs,
            additional_test_dataloader_kwargs=additional_test_dataloader_kwargs,
            shuffle_train=shuffle_train,
        )


def main():
    # Example usage
    data_module = NAMSSDataModule(
        "/workspaces/projects/shared_data/seismic/NAMSS/Data/NAMSS/patch_512_0",
        batch_size=4,
        num_workers=2,
    )
    single_data = data_module.train_dataset[0]
    single_batch = next(iter(data_module.train_dataloader()))

    print(data_module)
    print(f"Number of training samples (dataset)   : {len(data_module.train_dataset)}")
    print(
        f"Number of training samples (dataloader): {len(data_module.train_dataloader())}"
    )
    print(
        f"Shape of a single data sample          : {single_data.shape} (dtype: {single_data.dtype})"
    )
    print(
        f"Shape of a single batch                : {single_batch.shape} (dtype: {single_batch.dtype})"
    )


if __name__ == "__main__":
    main()