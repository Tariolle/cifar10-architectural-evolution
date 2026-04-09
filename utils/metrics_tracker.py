"""Utility for counting model parameters and FLOPs via fvcore."""

import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis


def count_parameters(model: nn.Module) -> int:
    """Return the total number of *trainable* parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model: nn.Module, sample_input: torch.Tensor) -> int:
    """Return total FLOPs for a single forward pass.

    Args:
        model: The module to profile.
        sample_input: A representative input tensor (batch size 1 recommended).

    Returns:
        Total FLOPs as an integer.
    """
    flops = FlopCountAnalysis(model, sample_input)
    return int(flops.total())


def model_summary(model: nn.Module, sample_input: torch.Tensor) -> dict[str, int]:
    """Return a dict containing ``params`` and ``flops``."""
    return {
        "params": count_parameters(model),
        "flops": count_flops(model, sample_input),
    }
