"""Lightweight CNN for knowledge distillation experiments (~56K params).

Same 3-block pattern as CNN but with much smaller channel counts (24→48→96
vs 64→128→256). Designed as a distillation student to test whether a tiny
conv net can absorb a teacher's knowledge.

Pipeline:
    [B, 3, 32, 32] -> Conv block x3 (channels: 24 → 48 → 96)
                    -> AdaptiveAvgPool -> [B, 96]
                    -> FC(96, 32) -> Dropout -> FC(32, 10)
"""

import torch
import torch.nn as nn


class TinyCNN(nn.Module):
    """Three-block CNN with ~56K params — 7x smaller than CNN (406K)."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: [B, 3, 32, 32] -> [B, 24, 16, 16]
            nn.Conv2d(3, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 2: [B, 24, 16, 16] -> [B, 48, 8, 8]
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 3: [B, 48, 8, 8] -> [B, 96, 4, 4]
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B, 96, 1, 1]
            nn.Flatten(),             # [B, 96]
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
