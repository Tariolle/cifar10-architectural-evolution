# CIFAR-10 Architectural Evolution

A comparative study of five model paradigms on CIFAR-10, tracing the evolution from classical linear classifiers to modern vision transformers.

## Models

| Model | Paradigm | Status |
|-------|----------|--------|
| SVM (RBF Kernel) | Baseline classifier via Random Fourier Features | Training |
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

## Tech Stack

- **PyTorch** — model definitions
- **PyTorch Lightning** — training loop abstraction
- **TorchMetrics** — accuracy tracking
- **TensorBoard** — experiment visualization
- **fvcore** — FLOPs and parameter counting
