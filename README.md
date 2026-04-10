# CIFAR-10 Architectural Evolution

A comparative study of five model paradigms on CIFAR-10, tracing the evolution from classical linear classifiers to modern vision transformers.

## Models

| Model | Paradigm | Status |
|-------|----------|--------|
| SVM (RBF Kernel) | Baseline classifier via Random Fourier Features | **49.5%** |
| MLP | Fully connected | **58.7%** |
| CNN | Convolutional | **87.3%** |
| ResNet-20 | Residual connections | **89.9%** |
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

### CNN (3-block Conv + BatchNorm) — 100 epochs, with data augmentation

| Metric | Start | Final |
| -------- | ------- | ------- |
| Val accuracy | 49.7% | **87.3%** |
| Val loss | 1.371 | 0.412 |
| Train accuracy | 40.7% | 92.4% |
| Train loss | 1.595 | 0.221 |
| Overfitting gap | ~0% | 5.2% |
| Trainable params | | 405,898 |

- Standard CIFAR-10 augmentation (random horizontal flip + random crop with 4px padding) added for all conv models.
- **+5.1% over no-augmentation run** (82.2% → 87.3%) — same model, same params, just more diverse training data.
- Overfitting crushed from 17.5% to 5.2% — augmentation prevents the model from memorizing the training set.
- **87.3% with 10x fewer params than MLP** (406K vs 3.8M) — spatial inductive bias dominates brute-force fully connected layers.
- The ceiling is now depth: stacking more conv layers causes vanishing gradients without skip connections.

### ResNet-20 — early stopped at 123 epochs, with data augmentation

| Metric | Start | Final |
| -------- | ------- | ------- |
| Val accuracy | 57.1% | **89.9%** |
| Val loss | 1.180 | 0.540 |
| Train accuracy | 46.9% | 99.0% |
| Train loss | 1.443 | 0.030 |
| Overfitting gap | ~0% | 9.6% |
| Trainable params | | 272,474 |

- Early stopping triggered at epoch 123 (best at 107, patience 15).
- **+2.6% over CNN with 33% fewer params** (272K vs 406K) — skip connections enable more efficient learning, not just deeper networks.
- 9.6% overfitting gap — higher than CNN (5.2%) because ResNet's deeper architecture has more capacity to memorize.
- **89.9% is close to the original paper's 91.25%** — the gap is likely due to Adam vs SGD with momentum (ResNets historically prefer SGD).

## Tech Stack

- **PyTorch** — model definitions
- **PyTorch Lightning** — training loop abstraction
- **TorchMetrics** — accuracy tracking
- **TensorBoard** — experiment visualization
- **fvcore** — FLOPs and parameter counting
