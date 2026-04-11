"""Knowledge distillation LightningModule.

Trains a student model using a combination of:
    - Hard-label loss: CrossEntropy(student_logits, ground_truth)
    - Soft-label loss: KL(student_soft, teacher_soft) * T²

Combined loss = α * soft_loss + (1 - α) * hard_loss

Teacher logits are precomputed and provided via the dataloader
(no teacher model in GPU memory during training).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy


class DistillationLitModule(pl.LightningModule):
    """Wraps a student model for knowledge distillation with precomputed teacher logits.

    The dataloader provides (image, label, teacher_logits) triples.
    """

    def __init__(
        self,
        student: nn.Module,
        temperature: float = 4.0,
        alpha: float = 0.7,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        warmup_epochs: int = 0,
        flatten_for_student: bool = False,
    ) -> None:
        super().__init__()
        self.student = torch.compile(student)

        self.temperature = temperature
        self.alpha = alpha
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.flatten_for_student = flatten_for_student

        self.ce_criterion = nn.CrossEntropyLoss()
        self.train_acc = MulticlassAccuracy(num_classes=10, top_k=1)
        self.val_acc = MulticlassAccuracy(num_classes=10, top_k=1)

        self.save_hyperparameters(ignore=["student"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.flatten_for_student:
            x = x.flatten(1)
        return self.student(x)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y, teacher_logits = batch  # teacher_logits: precomputed [B, 10]

        student_input = x.flatten(1) if self.flatten_for_student else x
        student_logits = self.student(student_input)

        # Hard-label loss
        ce_loss = self.ce_criterion(student_logits, y)

        # Soft-label loss: KL divergence with temperature scaling
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        kl_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (self.temperature ** 2)

        loss = self.alpha * kl_loss + (1 - self.alpha) * ce_loss

        self.train_acc(student_logits, y)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/ce_loss", ce_loss, on_step=False, on_epoch=True)
        self.log("train/kl_loss", kl_loss, on_step=False, on_epoch=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        # Validation batches are 2-tuples (no teacher logits needed)
        x, y = batch[0], batch[1]
        student_input = x.flatten(1) if self.flatten_for_student else x
        student_logits = self.student(student_input)

        loss = self.ce_criterion(student_logits, y)
        self.val_acc(student_logits, y)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(
            self.student.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )

        if self.warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-3, total_iters=self.warmup_epochs,
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs - self.warmup_epochs,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[self.warmup_epochs],
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs,
            )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
