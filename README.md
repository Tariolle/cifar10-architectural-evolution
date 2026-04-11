# CIFAR-10 Architectural Evolution

A comparative study of five model paradigms on CIFAR-10, tracing the evolution from classical linear classifiers to modern vision transformers. All models are trained from scratch under identical conditions (same data pipeline, AdamW optimizer, cosine annealing LR) with architecture-appropriate regularization.

## Results

| Model | Params | Val Acc | Train Acc | Overfit Gap | Epochs |
|-------|--------|---------|-----------|-------------|--------|
| SVM (RBF Kernel) | 41K | **49.5%** | 56.5% | 7.0% | 200 |
| MLP (3-layer FC) | 3.8M | **58.7%** | 91.1% | 32.4% | 95* |
| CNN (3-block Conv) | 406K | **87.3%** | 92.4% | 5.2% | 100 |
| ResNet-20 | 272K | **89.9%** | 99.0% | 9.6% | 123* |
| Swin Transformer | 5.4M | **86.6%** | 96.7% | 10.1% | 117* |

\* Early stopped (patience=15 on val/acc).

### Key findings

1. **SVM → MLP (+9.2%)**: 93x more parameters buy marginal gains. Without spatial awareness, fully connected layers memorize rather than generalize (32% overfit gap).
2. **MLP → CNN (+28.6%)**: The biggest jump. Spatial inductive bias (local connectivity, weight sharing) achieves 87% with 10x fewer params than the MLP.
3. **CNN → ResNet (+2.6%)**: Skip connections enable deeper, more efficient learning with 33% fewer params. Close to the original paper's 91.25% (gap likely due to AdamW vs SGD).
4. **ResNet → Swin (-3.3%)**: The surprise. 20x more parameters and a more expressive architecture *loses* to ResNet. On 50K 32x32 images, there isn't enough data for transformers to learn the spatial structure that convolutions get for free. With aggressive augmentation (RandAugment, CutMix) or pretraining, Swin reaches 90–97% on CIFAR-10 — but under a fair, uniform training recipe, convolutions win.

### Per-model details

<details>
<summary>SVM (RBF Kernel via Random Fourier Features) — ~200 epochs</summary>

- Approximates RBF kernel SVM via fixed random projections (Rahimi & Recht, 2007). Only the linear classifier is trainable.
- **49.5% matches the literature** for kernel SVM on raw CIFAR-10 pixels (expected: 50–55%).
- The ceiling is the representation: fixed random projections cannot capture spatial structure.
</details>

<details>
<summary>MLP (3-layer FC) — early stopped at 95 epochs</summary>

- 58.7% exceeds typical MLP benchmarks (literature: 53–57%), likely due to modern training recipe (AdamW, cosine annealing, weight decay + dropout).
- 32% train/val gap shows severe overfitting — the model memorizes but can't generalize.
- The ceiling is the architecture: fully connected layers treat each pixel independently.
</details>

<details>
<summary>CNN (3-block Conv + BatchNorm) — 100 epochs, with data augmentation</summary>

- Standard CIFAR-10 augmentation (random horizontal flip + random crop with 4px padding) for all conv-based models.
- +5.1% over no-augmentation baseline (82.2% → 87.3%) — same model, same params, just more diverse training data.
- The ceiling is depth: stacking more conv layers causes vanishing gradients without skip connections.
</details>

<details>
<summary>ResNet-20 — early stopped at 123 epochs, with data augmentation</summary>

- Architecture follows He et al. (2015): 3 stages of {16, 32, 64} channels, 3 blocks per stage.
- +2.6% over CNN with 33% fewer params — skip connections enable more efficient learning, not just deeper networks.
- 89.9% is close to the original paper's 91.25%. The gap is likely due to AdamW vs SGD with momentum.
</details>

<details>
<summary>Swin Transformer (CIFAR-10 adapted) — early stopped at ~117 epochs, with data augmentation</summary>

- CIFAR-10 adapted Swin-Tiny: patch_size=2, window_size=4, embed_dim=64, depths=[2,2,6], 3 stages (16x16 → 8x8 → 4x4).
- Transformer-specific training: 10-epoch linear LR warmup, weight_decay=0.05 (vs 1e-4 for CNNs), stochastic depth 0.1.
- 86.6% with 20x more params than ResNet — transformers lack the spatial inductive bias that makes convolutions efficient on small images.
- This matches independent benchmarks: Swin from scratch with basic augmentation reaches 86–90% on CIFAR-10. The 90%+ results in the literature use RandAugment, CutMix, or pretraining.
</details>

## Project Structure

```
├── core/
│   ├── data_module.py         # CIFAR-10 data loading and preprocessing
│   └── lightning_module.py    # Universal training wrapper
├── models/                    # Architecture definitions
├── utils/
│   └── metrics_tracker.py     # Parameter and FLOP counting (fvcore)
└── train.py                   # Entry point
```

## Setup

```bash
conda create -n cifar10-evo python=3.11 -y
conda activate cifar10-evo
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

## Usage

```bash
python train.py --model svm                         # train from scratch
python train.py --model cnn --max-epochs 200        # custom epoch count
python train.py --model resnet --ckpt last           # resume from last checkpoint
```

TensorBoard logs are written to `./logs/`. To visualize:

```bash
tensorboard --logdir logs
```

## Tech Stack

PyTorch, PyTorch Lightning, TorchMetrics, TensorBoard, fvcore
