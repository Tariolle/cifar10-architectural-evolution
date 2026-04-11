"""Hybrid CNN-Transformer for CIFAR-10 (Phase 8).

Combines convolutional early stages (ResNet-style BasicBlocks with GELU) for
local feature extraction with Swin Transformer late stages for global reasoning.

Pipeline:
    [B, 3, 32, 32] -> Conv stem(64) + BN + GELU   -> [B, 64, 32, 32]
                    -> 2x BasicBlock(64, GELU)      -> [B, 64, 32, 32]
                    -> BasicBlock(64, s=2, GELU)    -> [B, 64, 16, 16]
                    -> reshape + LayerNorm          -> [B, 256, 64]
                    -> Swin Stage 1 (2 blk, 16x16)  -> PatchMerge -> [B, 64, 128]
                    -> Swin Stage 2 (4 blk, 8x8)    -> [B, 64, 128]
                    -> LayerNorm -> pool -> FC       -> [B, 10]
"""

import torch
import torch.nn as nn

from models.resnet import BasicBlock
from models.swin import SwinStage


class HybridCNNTransformer(nn.Module):
    """Conv early stages + Swin attention late stages (~1.2M params)."""

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        conv_dim: int = 64,
        num_conv_blocks: int = 2,
        swin_depths: tuple[int, ...] = (2, 4),
        swin_heads: tuple[int, ...] = (2, 4),
        window_size: int = 4,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()

        # --- Conv backbone: local feature extraction ---
        self.conv_stem = nn.Sequential(
            nn.Conv2d(in_channels, conv_dim, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(conv_dim),
            nn.GELU(),
        )

        conv_blocks = [BasicBlock(conv_dim, conv_dim, act_layer=nn.GELU)
                       for _ in range(num_conv_blocks)]
        conv_blocks.append(BasicBlock(conv_dim, conv_dim, stride=2, act_layer=nn.GELU))
        self.conv_blocks = nn.Sequential(*conv_blocks)

        # --- Transition: conv features [B,C,H,W] -> transformer tokens [B,N,C] ---
        self.transition_norm = nn.LayerNorm(conv_dim)

        # --- Transformer backbone: global reasoning ---
        total_blocks = sum(swin_depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        self.swin_stages = nn.ModuleList()
        dim = conv_dim
        resolution = 16  # after stride-2 downsample from 32x32
        block_idx = 0
        for i, (depth, heads) in enumerate(zip(swin_depths, swin_heads)):
            is_last = (i == len(swin_depths) - 1)
            stage = SwinStage(
                dim=dim,
                depth=depth,
                num_heads=heads,
                window_size=window_size,
                resolution=resolution,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[block_idx:block_idx + depth],
                downsample=not is_last,
            )
            self.swin_stages.append(stage)
            block_idx += depth
            if not is_last:
                dim *= 2
                resolution //= 2

        # --- Classification head ---
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv backbone: [B, 3, 32, 32] -> [B, 64, 16, 16]
        x = self.conv_stem(x)
        x = self.conv_blocks(x)

        # Transition: [B, 64, 16, 16] -> [B, 256, 64]
        x = x.flatten(2).transpose(1, 2)
        x = self.transition_norm(x)

        # Transformer backbone
        for stage in self.swin_stages:
            x = stage(x)

        # Head: [B, N, C] -> [B, C] -> [B, 10]
        x = self.norm(x).mean(dim=1)
        return self.head(x)
