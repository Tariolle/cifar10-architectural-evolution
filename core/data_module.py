"""CIFAR-10 LightningDataModule with optional flattening for baseline models."""

from __future__ import annotations

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10


def _flatten_image(x: torch.Tensor) -> torch.Tensor:
    """[3, 32, 32] -> [3072]. Module-level function so it's picklable on Windows."""
    return x.view(-1)


class CIFAR10DataModule(pl.LightningDataModule):
    """Provides train/val dataloaders for CIFAR-10.

    Applies standard normalization (mean=0.5, std=0.5 per channel).
    When ``flatten=True``, reshapes each image from [3, 32, 32] to a
    3072-dimensional vector so it can be fed to flat models (SVM, MLP).
    """

    MEAN: tuple[float, float, float] = (0.5, 0.5, 0.5)
    STD: tuple[float, float, float] = (0.5, 0.5, 0.5)

    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 128,
        num_workers: int = 2,
        flatten: bool = False,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.flatten = flatten

        # --- Transform pipeline ---
        # PIL Image (32×32 RGB, uint8)
        #   -> ToTensor   => Tensor [3, 32, 32] float32 in [0, 1]
        #   -> Normalize  => Tensor [3, 32, 32] float32 in [-1, 1]
        #   -> (flatten)  => Tensor [3072]       float32 in [-1, 1]
        transform_list: list = [
            transforms.ToTensor(),
            transforms.Normalize(self.MEAN, self.STD),
        ]
        if self.flatten:
            transform_list.append(transforms.Lambda(_flatten_image))

        self.transform: transforms.Compose = transforms.Compose(transform_list)

        self.train_dataset: CIFAR10 | None = None
        self.val_dataset: CIFAR10 | None = None

    def setup(self, stage: str | None = None) -> None:
        """Download CIFAR-10 and create train / val splits.

        The official test set is used as the validation set (standard practice
        for CIFAR-10 benchmarks).
        """
        if stage in ("fit", None):
            self.train_dataset = CIFAR10(
                root=self.data_dir,
                train=True,
                download=True,
                transform=self.transform,
            )
            self.val_dataset = CIFAR10(
                root=self.data_dir,
                train=False,
                download=True,
                transform=self.transform,
            )

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None, "Call setup('fit') first."
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None, "Call setup('fit') first."
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )
