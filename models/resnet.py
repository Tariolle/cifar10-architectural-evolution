"""ResNet-20 for CIFAR-10 (He et al., 2015).

Architecture follows the original paper's CIFAR-10 variant:
- No 7x7 conv or maxpool stem (unlike ImageNet ResNets)
- 3 stages of 3x3 convs with {16, 32, 64} channels
- 3 residual blocks per stage (6n+2 = 20 layers for n=3)
- Global average pool -> single linear layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Pre-activation-free residual block: conv -> BN -> ReLU -> conv -> BN + skip."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut for dimension mismatch (channel increase or spatial downsample)
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet20(nn.Module):
    """ResNet-20 for CIFAR-10: 3 stages x 3 blocks, channels {16, 32, 64}.

    Pipeline:
        [B, 3, 32, 32] -> conv(16) -> BN -> ReLU
                        -> stage1: 3 blocks, 16ch, 32x32
                        -> stage2: 3 blocks, 32ch, 16x16
                        -> stage3: 3 blocks, 64ch, 8x8
                        -> global avg pool -> [B, 64] -> FC -> [B, 10]
    """

    def __init__(self, num_classes: int = 10, num_blocks: int = 3) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.stage1 = self._make_stage(16, 16, num_blocks, stride=1)
        self.stage2 = self._make_stage(16, 32, num_blocks, stride=2)
        self.stage3 = self._make_stage(32, 64, num_blocks, stride=2)

        self.fc = nn.Linear(64, num_classes)

    @staticmethod
    def _make_stage(in_ch: int, out_ch: int, num_blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(in_ch, out_ch, stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, 3, 32, 32] -> [B, 16, 32, 32]
        out = F.relu(self.bn1(self.conv1(x)))
        # Stages: [B,16,32,32] -> [B,16,32,32] -> [B,32,16,16] -> [B,64,8,8]
        out = self.stage3(self.stage2(self.stage1(out)))
        # Global average pool: [B, 64, 8, 8] -> [B, 64]
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)
