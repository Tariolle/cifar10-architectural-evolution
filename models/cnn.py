"""Vanilla CNN for CIFAR-10 classification."""

import torch
import torch.nn as nn


class CNN(nn.Module):
    """Three-block CNN with BatchNorm and MaxPool.

    First architecture in the evolution that exploits spatial structure.
    Introduces local connectivity (kernels), parameter sharing, and
    translation equivariance — all absent from SVM and MLP.

    Pipeline:
        [B, 3, 32, 32] -> Conv block ×3 (channels: 64 → 128 → 256)
                        -> AdaptiveAvgPool -> [B, 256]
                        -> FC(256, 128) -> Dropout -> FC(128, 10)
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: [B, 3, 32, 32] -> [B, 64, 16, 16]
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 2: [B, 64, 16, 16] -> [B, 128, 8, 8]
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 3: [B, 128, 8, 8] -> [B, 256, 4, 4]
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B, 256, 1, 1]
            nn.Flatten(),             # [B, 256]
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
