"""CIFAR-10 LightningDataModule with optional flattening and data augmentation."""

from __future__ import annotations

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import CIFAR10


def _flatten_image(x: torch.Tensor) -> torch.Tensor:
    """[3, 32, 32] -> [3072]. Module-level function so it's picklable on Windows."""
    return x.view(-1)


class CIFAR10DataModule(pl.LightningDataModule):
    """Provides train/val dataloaders for CIFAR-10.

    Applies standard normalization (mean=0.5, std=0.5 per channel).

    When ``flatten=True``, reshapes images to 3072-dim vectors and preloads
    into RAM (no augmentation — used by SVM, MLP).

    When ``augment=True``, applies random horizontal flip + random crop with
    4px padding (standard CIFAR-10 augmentation — used by CNN, ResNet, Swin).
    Augmented data cannot be preloaded since transforms are random per epoch.
    """

    MEAN: tuple[float, float, float] = (0.5, 0.5, 0.5)
    STD: tuple[float, float, float] = (0.5, 0.5, 0.5)

    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 128,
        num_workers: int = 2,
        flatten: bool = False,
        augment: bool = False,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.flatten = flatten
        self.augment = augment

        # --- Val/test transform (no augmentation) ---
        # PIL Image (32x32 RGB) -> [3, 32, 32] float32 in [-1, 1]
        val_transforms: list = [
            transforms.ToTensor(),
            transforms.Normalize(self.MEAN, self.STD),
        ]
        if self.flatten:
            val_transforms.append(transforms.Lambda(_flatten_image))
        self.val_transform = transforms.Compose(val_transforms)

        # --- Train transform (with optional augmentation) ---
        if self.augment:
            # RandomCrop(32, padding=4): pad 4px on each side, crop back to 32x32
            # RandomHorizontalFlip: 50% chance to flip
            self.train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(self.MEAN, self.STD),
            ])
        else:
            self.train_transform = self.val_transform

        self.train_dataset: TensorDataset | CIFAR10 | None = None
        self.val_dataset: TensorDataset | CIFAR10 | None = None

    def _preload_as_tensors(self, train: bool) -> TensorDataset:
        """Load entire split, apply transforms once, return a TensorDataset."""
        ds = CIFAR10(root=self.data_dir, train=train, download=True, transform=self.val_transform)
        xs, ys = zip(*[ds[i] for i in range(len(ds))])
        return TensorDataset(torch.stack(xs), torch.tensor(ys))

    def setup(self, stage: str | None = None) -> None:
        """Download CIFAR-10 and create train / val splits."""
        if stage in ("fit", None):
            if self.augment:
                # Augmentation requires per-epoch random transforms — can't preload
                self.train_dataset = CIFAR10(
                    root=self.data_dir, train=True, download=True,
                    transform=self.train_transform,
                )
                self.val_dataset = CIFAR10(
                    root=self.data_dir, train=False, download=True,
                    transform=self.val_transform,
                )
            else:
                # No augmentation: preload into RAM for speed
                self.train_dataset = self._preload_as_tensors(train=True)
                self.val_dataset = self._preload_as_tensors(train=False)
                self.num_workers = 0

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None, "Call setup('fit') first."
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None, "Call setup('fit') first."
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )
