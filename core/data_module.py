"""CIFAR-10 LightningDataModule with optional flattening and data augmentation."""

from __future__ import annotations

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import transforms
from torchvision.datasets import CIFAR10


def _flatten_image(x: torch.Tensor) -> torch.Tensor:
    """[3, 32, 32] -> [3072]. Module-level function so it's picklable on Windows."""
    return x.view(-1)


class _WithTeacherLogits(Dataset):
    """Wraps a dataset to also return precomputed teacher logits by index."""

    def __init__(self, dataset: Dataset, logits: torch.Tensor) -> None:
        self.dataset = dataset
        self.logits = logits

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple:
        x, y = self.dataset[idx]
        return x, y, self.logits[idx]


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
    IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
    IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 128,
        num_workers: int = 2,
        flatten: bool = False,
        augment: bool = False,
        teacher_logits_path: str | None = None,
        image_size: int | None = None,
        imagenet_norm: bool = False,
        train_subset: int | None = None,
        subset_seed: int = 0,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.flatten = flatten
        self.augment = augment
        self.teacher_logits = torch.load(teacher_logits_path, map_location="cpu", weights_only=True) if teacher_logits_path else None
        self.train_subset = train_subset
        self.subset_seed = subset_seed

        mean = self.IMAGENET_MEAN if imagenet_norm else self.MEAN
        std = self.IMAGENET_STD if imagenet_norm else self.STD
        crop_size = image_size or 32
        crop_padding = crop_size // 8  # 4px at 32, 28px at 224

        # --- Val/test transform (no augmentation) ---
        val_transforms: list = []
        if image_size:
            val_transforms.append(transforms.Resize(image_size))
        val_transforms += [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
        if self.flatten:
            val_transforms.append(transforms.Lambda(_flatten_image))
        self.val_transform = transforms.Compose(val_transforms)

        # --- Train transform (with optional augmentation) ---
        if self.augment:
            train_transforms: list = []
            if image_size:
                train_transforms.append(transforms.Resize(image_size))
            train_transforms += [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(crop_size, padding=crop_padding),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
            self.train_transform = transforms.Compose(train_transforms)
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

            # Attach precomputed teacher logits (for distillation)
            if self.teacher_logits is not None:
                self.train_dataset = _WithTeacherLogits(self.train_dataset, self.teacher_logits)

            # Subset the training set for data-scaling experiments.
            # Subset selection is seeded so different training seeds see
            # different subsets — exposing combined data + init variance.
            if self.train_subset is not None:
                gen = torch.Generator().manual_seed(self.subset_seed)
                n = len(self.train_dataset)
                indices = torch.randperm(n, generator=gen)[: self.train_subset].tolist()
                self.train_dataset = Subset(self.train_dataset, indices)

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
