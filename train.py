"""Main entry point for CIFAR-10 architectural evolution experiments.

Usage:
    python train.py --model svm                     # train SVM from scratch
    python train.py --model mlp --max-epochs 200    # train MLP for 200 epochs
    python train.py --model svm --ckpt last         # resume SVM from last checkpoint
    python train.py --model mlp --ckpt last --max-epochs 300
"""

import argparse
import logging
import time
import warnings

logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message=".*LeafSpec.*is deprecated.*")

import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

from core.data_module import CIFAR10DataModule
from core.lightning_module import CIFAR10LitModule
from models.svm import SVM
from models.mlp import MLP

# ======================================================================
# Model registry — maps CLI name to (model, criterion, flatten flag)
# ======================================================================
MODELS: dict[str, dict] = {
    "svm": {
        "model": lambda: SVM(input_dim=3072, num_classes=10),
        "criterion": nn.MultiMarginLoss,
        "flatten": True,
    },
    "mlp": {
        "model": lambda: MLP(input_dim=3072, num_classes=10),
        "criterion": nn.CrossEntropyLoss,
        "flatten": True,
    },
}


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
    parser.add_argument("--model", type=str, required=True, choices=MODELS.keys(),
                        help="Model architecture to train")
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--ckpt", type=str, default=None,
                        help='Path to checkpoint, or "last" to resume from last.ckpt')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MODELS[args.model]

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    data_module = CIFAR10DataModule(
        data_dir="./data",
        batch_size=128,
        num_workers=2,
        flatten=config["flatten"],
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = config["model"]()
    criterion = config["criterion"]()

    # ------------------------------------------------------------------
    # Lightning wrapper
    # ------------------------------------------------------------------
    lit_module = CIFAR10LitModule(model=model, criterion=criterion, lr=1e-3)

    # ------------------------------------------------------------------
    # TensorBoard logger  (events written to ./logs/<model>/)
    # ------------------------------------------------------------------
    logger = TensorBoardLogger(save_dir="./logs", name=args.model)

    # ------------------------------------------------------------------
    # Checkpointing  (saved to ./checkpoints/<model>/)
    # ------------------------------------------------------------------
    checkpoint_cb = ModelCheckpoint(
        dirpath=f"./checkpoints/{args.model}",
        filename=f"{args.model}-{{epoch:02d}}-{{val/acc:.3f}}",
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
