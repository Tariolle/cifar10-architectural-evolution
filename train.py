"""Main entry point for CIFAR-10 architectural evolution experiments.

Usage:
    python train.py --model svm                     # train SVM from scratch
    python train.py --model mlp --max-epochs 200    # train MLP for 200 epochs
    python train.py --model svm --ckpt last         # resume SVM from last checkpoint
    python train.py --model mlp --ckpt last --max-epochs 300
"""

import argparse
import logging
import time
import warnings

logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message=".*LeafSpec.*is deprecated.*")
warnings.filterwarnings("ignore", message=".*does not have many workers.*")

import torch
import torch.nn as nn

torch.backends.cudnn.benchmark = True
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

from core.data_module import CIFAR10DataModule
from core.distillation_module import DistillationLitModule
from core.lightning_module import CIFAR10LitModule
from models.svm import SVM
from models.mlp import MLP
from models.cnn import CNN
from models.tiny_cnn import TinyCNN
from models.resnet import ResNet20
from models.swin import SwinTransformer
from models.hybrid import HybridCNNTransformer
from models.pretrained import pretrained_resnet18, pretrained_swin_t

# ======================================================================
# Model registry — maps CLI name to (model, criterion, flatten flag)
# ======================================================================
MODELS: dict[str, dict] = {
    "svm": {
        "model": lambda: SVM(input_dim=3072, num_classes=10),
        "criterion": nn.MultiMarginLoss,
        "flatten": True,
    },
    "mlp": {
        "model": lambda: MLP(input_dim=3072, num_classes=10),
        "criterion": nn.CrossEntropyLoss,
        "flatten": True,
        "weight_decay": 1e-4,
    },
    "cnn": {
        "model": lambda: CNN(num_classes=10),
        "criterion": nn.CrossEntropyLoss,
        "flatten": False,
        "weight_decay": 1e-4,
        "augment": True,
    },
    "tiny-cnn": {
        "model": lambda: TinyCNN(num_classes=10),
        "criterion": nn.CrossEntropyLoss,
        "flatten": False,
        "weight_decay": 1e-4,
        "augment": True,
    },
    "resnet": {
        "model": lambda: ResNet20(num_classes=10),
        "criterion": nn.CrossEntropyLoss,
        "flatten": False,
        "weight_decay": 1e-4,
        "augment": True,
    },
    "swin": {
        "model": lambda: SwinTransformer(num_classes=10),
        "criterion": nn.CrossEntropyLoss,
        "flatten": False,
        "weight_decay": 0.05,
        "augment": True,
        "warmup_epochs": 10,
    },
    "hybrid": {
        "model": lambda: HybridCNNTransformer(num_classes=10),
        "criterion": nn.CrossEntropyLoss,
        "flatten": False,
        "weight_decay": 0.02,
        "augment": True,
        "warmup_epochs": 5,
    },
    # ------------------------------------------------------------------
    # Pretrained (ImageNet transfer learning)
    # ------------------------------------------------------------------
    "pretrained-resnet18": {
        "model": lambda: pretrained_resnet18(num_classes=10),
        "criterion": nn.CrossEntropyLoss,
        "flatten": False,
        "weight_decay": 1e-4,
        "augment": True,
        "warmup_epochs": 3,
        "lr": 1e-4,
        "image_size": 224,
        "imagenet_norm": True,
    },
    "pretrained-swin-t": {
        "model": lambda: pretrained_swin_t(num_classes=10),
        "criterion": nn.CrossEntropyLoss,
        "flatten": False,
        "weight_decay": 0.05,
        "augment": True,
        "warmup_epochs": 5,
        "lr": 1e-4,
        "batch_size": 64,
        "image_size": 224,
        "imagenet_norm": True,
    },
    # ------------------------------------------------------------------
    # Distillation students (teacher: ResNet-20)
    # ------------------------------------------------------------------
    "distill-mlp": {
        "model": lambda: MLP(input_dim=3072, num_classes=10),
        "criterion": nn.CrossEntropyLoss,
        "flatten": False,
        "weight_decay": 1e-4,
        "distill": True,
        "flatten_for_student": True,
    },
    "distill-cnn": {
        "model": lambda: CNN(num_classes=10),
        "criterion": nn.CrossEntropyLoss,
        "flatten": False,
        "weight_decay": 1e-4,
        "augment": True,
        "distill": True,
    },
    "distill-tiny-cnn": {
        "model": lambda: TinyCNN(num_classes=10),
        "criterion": nn.CrossEntropyLoss,
        "flatten": False,
        "weight_decay": 1e-4,
        "augment": True,
        "distill": True,
    },
}


class TrainingETA(pl.Callback):
    """Prints total training ETA at the end of each epoch."""

    def on_train_start(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        self._start = time.time()
        self._start_epoch = trainer.current_epoch

    def on_train_epoch_end(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        elapsed = time.time() - self._start
        epochs_this_run = trainer.current_epoch + 1 - self._start_epoch
        epochs_left = trainer.max_epochs - (trainer.current_epoch + 1)
        eta_seconds = int(elapsed / epochs_this_run * epochs_left)
        hrs, remainder = divmod(eta_seconds, 3600)
        mins, secs = divmod(remainder, 60)
        print(f"  Epoch {trainer.current_epoch + 1}/{trainer.max_epochs} — ETA: {hrs:02d}:{mins:02d}:{secs:02d}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CIFAR-10 training")
    parser.add_argument("--model", type=str, choices=MODELS.keys(),
                        help="Model architecture to train")
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="AdamW weight decay (default: per-model or 0)")
    parser.add_argument("--ckpt", type=str, default=None,
                        help='Path to checkpoint, or "last" to resume from last.ckpt')
    parser.add_argument("--precompute-teacher", type=str, default=None, metavar="CKPT",
                        help="Precompute teacher logits from checkpoint and exit")
    parser.add_argument("--teacher-logits", type=str, default="./checkpoints/resnet/teacher_logits.pt",
                        help="Path to precomputed teacher logits (default: resnet)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (drives init, batching, and subset choice)")
    parser.add_argument("--train-subset", type=int, default=None, metavar="N",
                        help="Train on a random subset of N samples (data-scaling experiments)")
    parser.add_argument("--patience", type=int, default=15,
                        help="EarlyStopping patience on val/acc (default: 15)")
    return parser.parse_args()


def precompute_teacher_logits(ckpt_path: str, save_path: str) -> None:
    """Run teacher on all 50K training images, save logits to disk."""
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torchvision.datasets import CIFAR10

    # Load teacher
    teacher = ResNet20(num_classes=10)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = {}
    for k, v in checkpoint["state_dict"].items():
        if k.startswith("model._orig_mod."):
            state_dict[k.removeprefix("model._orig_mod.")] = v
        elif k.startswith("model."):
            state_dict[k.removeprefix("model.")] = v
    teacher.load_state_dict(state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = teacher.to(device).eval()

    # Load training set (no augmentation, same normalization as training)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=2)

    all_logits = []
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        for images, _ in loader:
            logits = teacher(images.to(device))
            all_logits.append(logits.cpu())

    logits_tensor = torch.cat(all_logits)  # [50000, 10]
    torch.save(logits_tensor, save_path)
    print(f"Saved teacher logits {tuple(logits_tensor.shape)} to {save_path}")


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Precompute teacher logits and exit
    # ------------------------------------------------------------------
    if args.precompute_teacher:
        import os
        save_path = args.teacher_logits
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        precompute_teacher_logits(args.precompute_teacher, save_path)
        return

    if not args.model:
        parser = argparse.ArgumentParser()
        parser.error("--model is required for training")

    if args.seed is not None:
        pl.seed_everything(args.seed, workers=True)

    config = MODELS[args.model]
    is_distill = config.get("distill", False)

    # Unique run name when running sweeps (subset / seed).
    # Keeps defaults bit-identical to pre-sweep behavior.
    if args.train_subset is not None or args.seed is not None:
        subset_tag = args.train_subset if args.train_subset is not None else "full"
        seed_tag = args.seed if args.seed is not None else 0
        run_name = f"{args.model}_n{subset_tag}_s{seed_tag}"
    else:
        run_name = args.model

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    batch_size = config.get("batch_size", 128)
    data_module = CIFAR10DataModule(
        data_dir="./data",
        batch_size=batch_size,
        num_workers=2,
        flatten=config["flatten"],
        augment=config.get("augment", False),
        teacher_logits_path=args.teacher_logits if is_distill else None,
        image_size=config.get("image_size"),
        imagenet_norm=config.get("imagenet_norm", False),
        train_subset=args.train_subset,
        subset_seed=args.seed if args.seed is not None else 0,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = config["model"]()
    criterion = config["criterion"]()

    # ------------------------------------------------------------------
    # Lightning wrapper
    # ------------------------------------------------------------------
    weight_decay = args.weight_decay if args.weight_decay is not None else config.get("weight_decay", 0.0)
    warmup_epochs = config.get("warmup_epochs", 0)
    lr = config.get("lr", 1e-3)

    if is_distill:
        lit_module = DistillationLitModule(
            student=model, lr=lr,
            weight_decay=weight_decay, warmup_epochs=warmup_epochs,
            flatten_for_student=config.get("flatten_for_student", False),
        )
    else:
        lit_module = CIFAR10LitModule(
            model=model, criterion=criterion, lr=lr,
            weight_decay=weight_decay, warmup_epochs=warmup_epochs,
        )

    # ------------------------------------------------------------------
    # TensorBoard logger  (events written to ./logs/<run_name>/)
    # ------------------------------------------------------------------
    logger = TensorBoardLogger(save_dir="./logs", name=run_name)

    # ------------------------------------------------------------------
    # Checkpointing  (saved to ./checkpoints/<run_name>/)
    # ------------------------------------------------------------------
    checkpoint_cb = ModelCheckpoint(
        dirpath=f"./checkpoints/{run_name}",
        filename=f"{run_name}-{{epoch:02d}}-{{val/acc:.3f}}",
        monitor="val/acc",
        mode="max",
        save_last=True,
    )

    early_stop_cb = EarlyStopping(
        monitor="val/acc",
        patience=args.patience,
        mode="max",
    )

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=[checkpoint_cb, early_stop_cb, RichProgressBar(), TrainingETA()],
        accelerator="auto",
        devices=1,
        precision="16-mixed",
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    trainer.fit(lit_module, datamodule=data_module, ckpt_path=args.ckpt)


if __name__ == "__main__":
    main()
