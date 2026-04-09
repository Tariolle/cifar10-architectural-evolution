# CIFAR-10 Architectural Evolution

A comparative study of five model paradigms on CIFAR-10, tracing the evolution from classical linear classifiers to modern vision transformers.

## Models

| Model | Paradigm | Status |
|-------|----------|--------|
| SVM (RBF Kernel) | Baseline classifier via Random Fourier Features | **49.5%** |
| MLP | Fully connected | Planned |
| CNN | Convolutional | Planned |
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
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Usage

```bash
python train.py                                     # train from scratch, 20 epochs
python train.py --max-epochs 50                     # custom epoch count
python train.py --ckpt last                         # resume from last checkpoint
python train.py --ckpt checkpoints/svm/X.ckpt --max-epochs 100
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

## Tech Stack

- **PyTorch** — model definitions
- **PyTorch Lightning** — training loop abstraction
- **TorchMetrics** — accuracy tracking
- **TensorBoard** — experiment visualization
- **fvcore** — FLOPs and parameter counting
