"""Efficiency benchmark: FLOPs, params, and inference time for all models."""

import logging
import time
import warnings

logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
logging.getLogger("fvcore").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*LeafSpec.*is deprecated.*")

import torch
import torch.nn as nn

from models.svm import SVM
from models.mlp import MLP
from models.cnn import CNN
from models.resnet import ResNet20
from models.swin import SwinTransformer
from models.hybrid import HybridCNNTransformer
from models.pretrained import pretrained_resnet18, pretrained_swin_t
from utils.metrics_tracker import count_parameters, count_flops

MODELS = {
    "SVM": (SVM(input_dim=3072, num_classes=10), torch.randn(1, 3072)),
    "MLP": (MLP(input_dim=3072, num_classes=10), torch.randn(1, 3072)),
    "CNN": (CNN(num_classes=10), torch.randn(1, 3, 32, 32)),
    "ResNet-20": (ResNet20(num_classes=10), torch.randn(1, 3, 32, 32)),
    "Swin": (SwinTransformer(num_classes=10), torch.randn(1, 3, 32, 32)),
    "Hybrid": (HybridCNNTransformer(num_classes=10), torch.randn(1, 3, 32, 32)),
    # Pretrained models operate at their native 224x224 resolution
    "ResNet-18*": (pretrained_resnet18(num_classes=10), torch.randn(1, 3, 224, 224)),
    "Swin-T*": (pretrained_swin_t(num_classes=10), torch.randn(1, 3, 224, 224)),
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WARMUP_RUNS = 50
TIMED_RUNS = 200
BATCH_SIZE = 128


def benchmark_inference(model: nn.Module, sample: torch.Tensor) -> float:
    """Return average inference time in ms for a single batch."""
    batch = sample.expand(BATCH_SIZE, *sample.shape[1:]).to(DEVICE)
    model = model.to(DEVICE).eval()

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
        for _ in range(WARMUP_RUNS):
            model(batch)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(TIMED_RUNS):
            model(batch)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    return (elapsed / TIMED_RUNS) * 1000  # ms per batch


def format_flops(flops: int) -> str:
    if flops >= 1e9:
        return f"{flops / 1e9:.2f}G"
    return f"{flops / 1e6:.1f}M"


def main() -> None:
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}, Warmup: {WARMUP_RUNS}, Timed: {TIMED_RUNS}\n")

    header = f"{'Model':<12} {'Params':>10} {'FLOPs':>10} {'ms/batch':>10} {'Acc':>8} {'Acc/MFLOP':>10}"
    print(header)
    print("-" * len(header))

    # Best val accuracies from training
    accuracies = {
        "SVM": 49.5, "MLP": 58.7, "CNN": 87.3, "ResNet-20": 89.9, "Swin": 86.6, "Hybrid": 90.4,
        "ResNet-18*": 96.6, "Swin-T*": 97.4,
    }

    for name, (model, sample) in MODELS.items():
        params = count_parameters(model)
        flops = count_flops(model, sample)
        ms = benchmark_inference(model, sample)
        acc = accuracies[name]
        acc_per_mflop = acc / (flops / 1e6) if flops > 0 else float("inf")

        print(
            f"{name:<12} {params:>10,} {format_flops(flops):>10} "
            f"{ms:>9.1f}ms {acc:>7.1f}% {acc_per_mflop:>9.2f}"
        )


if __name__ == "__main__":
    main()
