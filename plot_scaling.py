"""Produce the data-scaling curve from TensorBoard event files.

Reads logs/{model}_n{N}_s{seed}/ for model in {resnet, swin, hybrid},
N in {1K..50K}, seed in {0,1,2}. Takes best val/acc per run, aggregates
to mean ± std across seeds, and writes a PNG to assets/.

Usage: python plot_scaling.py
"""

from __future__ import annotations

import glob
import os
import re
import statistics
from collections import defaultdict

import matplotlib.pyplot as plt
from tbparse import SummaryReader

MODELS = ["resnet", "swin", "hybrid"]
LABELS = {"resnet": "ResNet-20", "swin": "Swin", "hybrid": "Hybrid"}
COLORS = {"resnet": "#1f77b4", "swin": "#d62728", "hybrid": "#2ca02c"}
MARKERS = {"resnet": "o", "swin": "s", "hybrid": "^"}


def collect_results() -> dict[str, dict[int, list[float]]]:
    """results[model][N] = [best_val_acc_seed0, best_val_acc_seed1, ...] in percent."""
    results: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for d in sorted(glob.glob("logs/*_n*_s*")):
        m = re.match(r"(\w+)_n(\d+)_s(\d+)", os.path.basename(d))
        if not m:
            continue
        model, size, _ = m.group(1), int(m.group(2)), int(m.group(3))
        if model not in MODELS:
            continue
        df = SummaryReader(d).scalars
        va = df[df["tag"] == "val/acc"]["value"]
        if va.empty:
            continue
        results[model][size].append(100.0 * va.max())
    return {m: dict(sorted(results[m].items())) for m in MODELS if m in results}


def plot_scaling_curve(results: dict[str, dict[int, list[float]]], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    for model in MODELS:
        if model not in results:
            continue
        sizes = sorted(results[model])
        means = [statistics.mean(results[model][n]) for n in sizes]
        stds = [statistics.stdev(results[model][n]) if len(results[model][n]) > 1 else 0.0 for n in sizes]

        ax.errorbar(
            sizes, means, yerr=stds,
            label=LABELS[model], color=COLORS[model], marker=MARKERS[model],
            markersize=7, linewidth=2, capsize=4, capthick=1.5,
        )

    ax.set_xscale("log")
    ax.set_xticks([1000, 5000, 10000, 25000, 50000])
    ax.set_xticklabels(["1K", "5K", "10K", "25K", "50K"])
    ax.set_xlabel("Training samples", fontsize=11)
    ax.set_ylabel("Validation accuracy (%)", fontsize=11)
    ax.set_title("CIFAR-10 data scaling (3 seeds, best val/acc)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.set_ylim(35, 95)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


def plot_scaling_curve_band(results: dict[str, dict[int, list[float]]], out_path: str) -> None:
    """Shaded ±1σ confidence band version of the main scaling curve."""
    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    for model in MODELS:
        if model not in results:
            continue
        sizes = sorted(results[model])
        means = [statistics.mean(results[model][n]) for n in sizes]
        stds = [statistics.stdev(results[model][n]) if len(results[model][n]) > 1 else 0.0 for n in sizes]
        lo = [m - s for m, s in zip(means, stds)]
        hi = [m + s for m, s in zip(means, stds)]

        ax.fill_between(sizes, lo, hi, color=COLORS[model], alpha=0.25, linewidth=0)
        ax.plot(
            sizes, means,
            label=LABELS[model], color=COLORS[model], marker=MARKERS[model],
            markersize=7, linewidth=2,
        )

    ax.set_xscale("log")
    ax.set_xticks([1000, 5000, 10000, 25000, 50000])
    ax.set_xticklabels(["1K", "5K", "10K", "25K", "50K"])
    ax.set_xlabel("Training samples", fontsize=11)
    ax.set_ylabel("Validation accuracy (%)", fontsize=11)
    ax.set_title("CIFAR-10 data scaling (3 seeds, mean ± 1σ)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.set_ylim(30, 95)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


def main() -> None:
    results = collect_results()
    os.makedirs("assets", exist_ok=True)
    plot_scaling_curve(results, "assets/data_scaling.png")
    plot_scaling_curve_band(results, "assets/data_scaling_band.png")


if __name__ == "__main__":
    main()
