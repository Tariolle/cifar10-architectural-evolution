"""RBF Kernel SVM via Random Fourier Features (Rahimi & Recht, 2007)."""

import math

import torch
import torch.nn as nn


class SVM(nn.Module):
    """Approximate RBF kernel SVM using Random Fourier Features.

    The RFF layer maps inputs into a random feature space that approximates
    the RBF kernel, followed by a linear classifier trained with hinge loss.
    The random projection weights are fixed (not learned).

    Pipeline:
        [B, 3072] -> RFF projection -> [B, rff_dim] -> Linear -> [B, 10]

    Args:
        input_dim: Dimensionality of flattened input (3072 for CIFAR-10).
        num_classes: Number of output classes (10 for CIFAR-10).
        rff_dim: Number of random Fourier features.
        gamma: RBF kernel bandwidth parameter. Controls the "reach" of
               each training example. Default 1/input_dim follows
               scikit-learn's convention.
    """

    def __init__(
        self,
        input_dim: int = 3072,
        num_classes: int = 10,
        rff_dim: int = 4096,
        gamma: float | None = None,
    ) -> None:
        super().__init__()
        gamma = gamma or (1.0 / input_dim)

        # Fixed random projection: W ~ N(0, 2*gamma), b ~ Uniform(0, 2*pi)
        # These are NOT learned — they approximate the RBF kernel.
        self.register_buffer("W", torch.randn(input_dim, rff_dim) * math.sqrt(2.0 * gamma))
        self.register_buffer("b", torch.rand(rff_dim) * (2.0 * math.pi))

        # Scale factor for the RFF mapping: sqrt(2 / D)
        self.register_buffer("scale", torch.tensor(math.sqrt(2.0 / rff_dim)))

        # Trainable linear classifier on top of the random features
        self.linear = nn.Linear(rff_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3072]
        z = x @ self.W + self.b  # [B, rff_dim]
        z = torch.cos(z) * self.scale  # [B, rff_dim] — RFF mapping
        return self.linear(z)  # [B, 10]
