# CIFAR-10 Architectural Evolution

A comparative study of five model paradigms on CIFAR-10, tracing the evolution from classical linear classifiers to modern vision transformers.

## Models

| Model | Paradigm | Status |
|-------|----------|--------|
| SVM (RBF Kernel) | Baseline classifier via Random Fourier Features | **49.5%** |
| MLP | Fully connected | **58.7%** |
| CNN | Convolutional | **82.2%** |
| ResNet | Residual connections | Planned |
| Swin Transformer | Shifted-window attention | Planned |

Each model is trained under identical conditions (same data pipeline, optimizer, and metrics) to ensure a fair comparison.

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
python train.py --model svm                         # train SVM from scratch
python train.py --model mlp --max-epochs 200        # train MLP for 200 epochs
python train.py --model cnn --ckpt last             # resume CNN from last checkpoint
python train.py --model mlp --ckpt last --max-epochs 300
```

TensorBoard logs are written to `./logs/`. To visualize:

```bash
tensorboard --logdir logs
```

## Results

### SVM (RBF Kernel via Random Fourier Features) — ~200 epochs to converge

| Metric | Start | Final |
| -------- | ------- | ------- |
| Val accuracy | ~42% | **49.5%** |
| Val loss | ~0.39 | 0.319 |
| Train accuracy | ~42% | 56.5% |
| Train loss | ~0.39 | 0.243 |
| Overfitting gap | ~0% | 7% |
| Trainable params | | 40,970 |

- Plateaus around epoch ~100, marginal gains after.
- 7% train/val gap indicates mild overfitting — the fixed RFF features can't generalize beyond ~50%.
- **49.5% matches the literature** for RBF kernel SVM on raw CIFAR-10 pixels (expected: 50–55%).
- The ceiling is the representation: fixed random projections cannot capture spatial structure in images.

### MLP (3-layer FC) — early stopped at 95 epochs

| Metric | Start | Final |
| -------- | ------- | ------- |
| Val accuracy | ~46% | **58.7%** |
| Val loss | ~1.56 | 1.844 |
| Train accuracy | ~37% | 91.1% |
| Train loss | ~1.76 | 0.254 |
| Overfitting gap | ~0% | 32.4% |
| Trainable params | | 3,805,450 |

- Early stopping triggered (patience=15 on val/acc) — val accuracy plateaued around epoch 80.
- 32% train/val gap shows severe overfitting: the model memorizes training data but can't generalize.
- **58.7% exceeds typical MLP benchmarks** (literature: 53–57%), likely due to modern training recipe (AdamW, cosine annealing, weight decay + dropout).
- 93x more parameters than SVM (+9.2% accuracy) — diminishing returns without spatial inductive bias. The ceiling is the architecture: fully connected layers treat each pixel independently.

### CNN (3-block Conv + BatchNorm) — 50 epochs

| Metric | Start | Final |
| -------- | ------- | ------- |
| Val accuracy | 53.8% | **82.2%** |
| Val loss | 1.270 | 0.801 |
| Train accuracy | 45.3% | 99.7% |
| Train loss | 1.483 | 0.017 |
| Overfitting gap | ~0% | 17.5% |
| Trainable params | | 405,898 |

- Best val accuracy at epoch 50 — model was still improving, more epochs or data augmentation could push higher.
- 17.5% train/val gap shows significant overfitting (train acc 99.7%).
- **82.2% with 10x fewer params than MLP** (406K vs 3.8M) — spatial inductive bias (local connectivity, parameter sharing, translation equivariance) dominates brute-force fully connected layers.
- The ceiling is now depth: stacking more conv layers causes vanishing gradients without skip connections.

## Tech Stack

- **PyTorch** — model definitions
- **PyTorch Lightning** — training loop abstraction
- **TorchMetrics** — accuracy tracking
- **TensorBoard** — experiment visualization
- **fvcore** — FLOPs and parameter counting
