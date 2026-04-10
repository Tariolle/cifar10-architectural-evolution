"""Multi-Layer Perceptron for CIFAR-10 classification."""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Three hidden-layer MLP with ReLU and Dropout.

    Learns its own feature representations (unlike the SVM baseline which
    uses fixed random projections). No BatchNorm — that came with CNNs.

    Pipeline:
        [B, 3072] -> 1024 -> ReLU -> Dropout
                  -> 512  -> ReLU -> Dropout
                  -> 256  -> ReLU -> Dropout
                  -> [B, 10]
    """

    def __init__(
        self,
        input_dim: int = 3072,
        num_classes: int = 10,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
