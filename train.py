"""Main entry point for CIFAR-10 architectural evolution experiments.

Usage:
    python train.py                             # train from scratch, 20 epochs
    python train.py --max-epochs 50             # train from scratch, 50 epochs
    python train.py --ckpt last                 # resume from last checkpoint
    python train.py --ckpt checkpoints/svm/X.ckpt --max-epochs 40
"""

import argparse
import logging
import time
import warnings

logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*LeafSpec.*is deprecated.*")

import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

from core.data_module import CIFAR10DataModule
from core.lightning_module import CIFAR10LitModule
from models.svm import SVM


class TrainingETA(pl.Callback):
    """Prints total training ETA at the end of each epoch."""

    def on_train_start(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        self._start = time.time()
        self._start_epoch = trainer.current_epoch

    def on_train_epoch_end(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        elapsed = time.time() - self._start
        epochs_this_run = trainer.current_epoch + 1 - self._start_epoch
        epochs_left = trainer.max_epochs - (trainer.current_epoch + 1)
        eta_seconds = int(elapsed / epochs_this_run * epochs_left)
        hrs, remainder = divmod(eta_seconds, 3600)
        mins, secs = divmod(remainder, 60)
        print(f"  Epoch {trainer.current_epoch + 1}/{trainer.max_epochs} — ETA: {hrs:02d}:{mins:02d}:{secs:02d}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CIFAR-10 training")
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--ckpt", type=str, default=None,
                        help='Path to checkpoint, or "last" to resume from last.ckpt')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    # flatten=True: [B, 3, 32, 32] -> [B, 3072] for the linear SVM
    data_module = CIFAR10DataModule(
        data_dir="./data",
        batch_size=128,
        num_workers=2,
        flatten=True,
    )

    # ------------------------------------------------------------------
    # Model — Linear SVM
    # ------------------------------------------------------------------
    # Single linear layer: [B, 3072] -> [B, 10]
    # Trained with multi-class hinge loss (MultiMarginLoss)
    model = SVM(input_dim=3072, num_classes=10)
    criterion = nn.MultiMarginLoss()

    # ------------------------------------------------------------------
    # Lightning wrapper
    # ------------------------------------------------------------------
    lit_module = CIFAR10LitModule(model=model, criterion=criterion, lr=1e-3)

    # ------------------------------------------------------------------
    # TensorBoard logger  (events written to ./logs/cifar10/)
    # ------------------------------------------------------------------
    logger = TensorBoardLogger(save_dir="./logs", name="cifar10")

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    checkpoint_cb = ModelCheckpoint(
        dirpath="./checkpoints/svm",
        filename="svm-{epoch:02d}-{val/acc:.3f}",
        monitor="val/acc",
        mode="max",
        save_last=True,
    )

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=[checkpoint_cb, RichProgressBar(), TrainingETA()],
        accelerator="auto",
        devices=1,
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    trainer.fit(lit_module, datamodule=data_module, ckpt_path=args.ckpt)


if __name__ == "__main__":
    main()
