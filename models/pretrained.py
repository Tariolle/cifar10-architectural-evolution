"""Pretrained ImageNet models fine-tuned for CIFAR-10 (Phase 9).

Loads torchvision pretrained weights and replaces the classification head
with a 10-class linear layer. All backbone weights are kept trainable
(full fine-tuning).

ResNet-18:  ~11.2M params
Swin-Tiny:  ~27.5M params
"""

import torch.nn as nn
from torchvision.models import resnet18, swin_t, ResNet18_Weights, Swin_T_Weights


def pretrained_resnet18(num_classes: int = 10) -> nn.Module:
    """ResNet-18 with ImageNet-pretrained weights, head replaced for CIFAR-10."""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def pretrained_swin_t(num_classes: int = 10) -> nn.Module:
    """Swin-Tiny with ImageNet-pretrained weights, head replaced for CIFAR-10."""
    model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model
