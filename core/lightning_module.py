"""Universal LightningModule wrapper for CIFAR-10 classification."""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy


class CIFAR10LitModule(pl.LightningModule):
    """Wraps any ``nn.Module`` and trains it on CIFAR-10.

    Handles CrossEntropyLoss, Top-1 accuracy (via TorchMetrics), and
    automatic TensorBoard logging of both metrics.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module | None = None,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__()
        self.model = torch.compile(model)
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = criterion or nn.CrossEntropyLoss()

        # Top-1 accuracy trackers (one per phase to avoid metric leakage)
        self.train_acc = MulticlassAccuracy(num_classes=10, top_k=1)
        self.val_acc = MulticlassAccuracy(num_classes=10, top_k=1)

        # Persist lr (but not the model graph) inside hparams
        self.save_hyperparameters(ignore=["model", "criterion"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the wrapped model.

        Args:
            x: Input tensor.
               Flat models  (SVM, MLP):            [B, 3072]
               Conv models  (CNN, ResNet, Swin):    [B, 3, 32, 32]

        Returns:
            Logits tensor of shape [B, 10].
        """
        return self.model(x)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        # x: [B, D] or [B, C, H, W]   y: [B] int64 labels in {0..9}
        logits: torch.Tensor = self(x)  # [B, 10]
        loss: torch.Tensor = self.criterion(logits, y)

        self.train_acc(logits, y)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, y = batch
        logits: torch.Tensor = self(x)  # [B, 10]
        loss: torch.Tensor = self.criterion(logits, y)

        self.val_acc(logits, y)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
