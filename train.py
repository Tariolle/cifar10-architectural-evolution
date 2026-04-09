"""Main entry point for CIFAR-10 architectural evolution experiments.

Sanity-check usage:
    python train.py

Trains a dummy linear layer for 2 epochs to validate the full pipeline
(data loading, training loop, TensorBoard logging).
"""

import logging
import warnings

logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*LeafSpec.*is deprecated.*")

import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from core.data_module import CIFAR10DataModule
from core.lightning_module import CIFAR10LitModule


def main() -> None:
    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    # flatten=True: each image is reshaped from [3, 32, 32] to [3072]
    # so the dummy linear layer can consume it directly.
    data_module = CIFAR10DataModule(
        data_dir="./data",
        batch_size=128,
        num_workers=2,
        flatten=True,
    )

    # ------------------------------------------------------------------
    # Model (placeholder)
    # ------------------------------------------------------------------
    # A single Linear layer: [B, 3072] -> [B, 10]
    # This will be replaced by real architectures (SVM, MLP, CNN, etc.)
    model = nn.Linear(in_features=3072, out_features=10)

    # ------------------------------------------------------------------
    # Lightning wrapper
    # ------------------------------------------------------------------
    lit_module = CIFAR10LitModule(model=model, lr=1e-3)

    # ------------------------------------------------------------------
    # TensorBoard logger  (events written to ./logs/cifar10/)
    # ------------------------------------------------------------------
    logger = TensorBoardLogger(save_dir="./logs", name="cifar10")

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = Trainer(
        max_epochs=2,
        logger=logger,
        accelerator="auto",
        devices=1,
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    trainer.fit(lit_module, datamodule=data_module)


if __name__ == "__main__":
    main()
