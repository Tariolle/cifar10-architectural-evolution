"""Swin Transformer for CIFAR-10 classification (adapted from Liu et al., 2021).

Standard Swin-Tiny targets 224x224 with patch_size=4, window_size=7, embed_dim=96.
For CIFAR-10's 32x32 images we shrink everything:

    patch_size=2    -> 16x16 = 256 tokens (vs 56x56 = 3136)
    window_size=4   -> divides 16, 8, and 4 evenly
    3 stages        -> 16x16 -> 8x8 -> 4x4 (vs 4 stages down to 7x7)
    embed_dim=64    -> channels 64 -> 128 -> 256
    depths=[2,2,6]  -> 10 transformer blocks total

Pipeline:
    [B, 3, 32, 32] -> PatchEmbed(2x2)         -> [B, 256, 64]
                    -> Stage 1 (2 blocks, 16x16) -> PatchMerge -> [B, 64, 128]
                    -> Stage 2 (2 blocks, 8x8)   -> PatchMerge -> [B, 16, 256]
                    -> Stage 3 (2 blocks, 4x4)   -> [B, 16, 256]
                    -> LayerNorm -> mean pool -> Linear -> [B, 10]

At 4x4 resolution, window_size=4 covers the full map — effectively global attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Window helpers
# ---------------------------------------------------------------------------

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition feature map into non-overlapping windows.

    Args:
        x: [B, H, W, C]
    Returns:
        windows: [num_windows * B, window_size, window_size, C]
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """Reverse window_partition.

    Args:
        windows: [num_windows * B, window_size, window_size, C]
    Returns:
        x: [B, H, W, C]
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


# ---------------------------------------------------------------------------
# Drop path (stochastic depth)
# ---------------------------------------------------------------------------

class DropPath(nn.Module):
    """Drop entire residual branches during training (stochastic depth)."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        # Per-sample mask: [B, 1, 1] for [B, L, C] input
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, dtype=x.dtype, device=x.device).add_(keep).floor_()
        return x / keep * mask


# ---------------------------------------------------------------------------
# MLP block
# ---------------------------------------------------------------------------

class Mlp(nn.Module):
    """Two-layer MLP with GELU: Linear -> GELU -> Drop -> Linear -> Drop."""

    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


# ---------------------------------------------------------------------------
# Patch embedding
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """Convert image to patch tokens via strided convolution.

    [B, 3, 32, 32] -> Conv2d(patch_size, stride=patch_size) -> [B, H*W, embed_dim]
    """

    def __init__(self, in_channels: int = 3, embed_dim: int = 64, patch_size: int = 2) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, 3, 32, 32] -> [B, C, H, W] -> [B, H*W, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Patch merging (spatial downsample 2x, channel upsample 2x)
# ---------------------------------------------------------------------------

class PatchMerging(nn.Module):
    """Merge 2x2 neighboring patches: halve resolution, double channels.

    [B, H*W, C] -> [B, (H/2)*(W/2), 2*C]
    """

    def __init__(self, dim: int, resolution: int) -> None:
        super().__init__()
        self.resolution = resolution
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, C = x.shape
        H = W = self.resolution
        x = x.view(B, H, W, C)

        # Gather 2x2 neighbors along channel dim
        x0 = x[:, 0::2, 0::2, :]  # top-left
        x1 = x[:, 1::2, 0::2, :]  # bottom-left
        x2 = x[:, 0::2, 1::2, :]  # top-right
        x3 = x[:, 1::2, 1::2, :]  # bottom-right
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # [B, H/2, W/2, 4C]

        x = x.view(B, -1, 4 * C)
        return self.reduction(self.norm(x))


# ---------------------------------------------------------------------------
# Window attention with relative position bias
# ---------------------------------------------------------------------------

class WindowAttention(nn.Module):
    """Multi-head self-attention within a local window.

    Each window has window_size*window_size tokens. Attention is computed
    independently per window. Relative position bias is added to attention
    scores before softmax (learnable, per head).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Relative position bias table: (2*ws-1) * (2*ws-1) entries, one per head
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Precompute relative position index for each token pair in a window
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size), indexing="ij"
        ))  # [2, ws, ws]
        coords_flat = coords.view(2, -1)  # [2, ws*ws]
        # Pairwise relative positions: [2, ws*ws, ws*ws]
        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [ws*ws, ws*ws, 2]
        relative_coords[:, :, 0] += window_size - 1  # shift to non-negative
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1  # row stride
        self.register_buffer("relative_position_index", relative_coords.sum(-1))  # [ws*ws, ws*ws]

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x:    [nW*B, N, C]  where N = window_size^2
            mask: [nW, N, N] or None
        Returns:
            [nW*B, N, C]
        """
        BW, N, C = x.shape
        qkv = self.qkv(x).reshape(BW, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each [BW, heads, N, head_dim]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [BW, heads, N, N]

        # Add relative position bias
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(N, N, self.num_heads).permute(2, 0, 1)  # [heads, N, N]
        attn = attn + bias.unsqueeze(0)

        # Apply shifted-window mask (different per window, same across batch)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(BW // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)  # [1, nW, 1, N, N]
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(BW, N, C)
        return self.proj_drop(self.proj(x))


# ---------------------------------------------------------------------------
# Swin Transformer block
# ---------------------------------------------------------------------------

class SwinTransformerBlock(nn.Module):
    """One Swin block: LN -> W-MSA or SW-MSA -> residual -> LN -> MLP -> residual.

    Even-indexed blocks use regular windows (shift_size=0).
    Odd-indexed blocks use shifted windows (shift_size=window_size//2).
    At the smallest resolution where window_size >= resolution, shift is disabled.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        resolution: int,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.resolution = resolution
        self.window_size = window_size
        # No shifting when window covers the entire feature map
        self.shift_size = shift_size if resolution > window_size else 0

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, num_heads, window_size,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

        # Precompute attention mask for shifted windows
        if self.shift_size > 0:
            H = W = resolution
            img_mask = torch.zeros(1, H, W, 1)
            h_slices = (
                slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None),
            )
            w_slices = (
                slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, window_size)  # [nW, ws, ws, 1]
            mask_windows = mask_windows.view(-1, window_size * window_size)  # [nW, ws*ws]
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, N, N]
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
            self.register_buffer("attn_mask", attn_mask)
        else:
            self.attn_mask = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        H = W = self.resolution

        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # Partition into windows -> attention -> merge back
        x = window_partition(x, self.window_size)            # [nW*B, ws, ws, C]
        x = x.view(-1, self.window_size * self.window_size, C)  # [nW*B, N, C]
        x = self.attn(x, mask=self.attn_mask)                # [nW*B, N, C]
        x = x.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(x, self.window_size, H, W)       # [B, H, W, C]

        # Reverse shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x.view(B, L, C)

        # Residual connections
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Swin stage (N blocks + optional downsample)
# ---------------------------------------------------------------------------

class SwinStage(nn.Module):
    """A sequence of Swin blocks at a given resolution, with optional PatchMerging."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        resolution: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: list[float] | float = 0.0,
        downsample: bool = True,
    ) -> None:
        super().__init__()
        dpr = drop_path if isinstance(drop_path, list) else [drop_path] * depth

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                resolution=resolution,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=dpr[i],
            )
            for i in range(depth)
        ])
        self.downsample = PatchMerging(dim, resolution) if downsample else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


# ---------------------------------------------------------------------------
# Top-level Swin Transformer
# ---------------------------------------------------------------------------

class SwinTransformer(nn.Module):
    """Swin Transformer adapted for CIFAR-10 (32x32 input).

    Defaults produce a ~5.4M parameter model:
        embed_dim=64, depths=[2,2,6], heads=[2,4,8], window_size=4

    Pipeline:
        [B, 3, 32, 32] -> PatchEmbed -> 3 stages -> LN -> pool -> [B, 10]
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        embed_dim: int = 64,
        depths: tuple[int, ...] = (2, 2, 6),
        num_heads: tuple[int, ...] = (2, 4, 8),
        window_size: int = 4,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        patch_size: int = 2,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)

        # Stochastic depth: linearly increase from 0 to drop_path_rate
        total_blocks = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        # Build stages
        resolution = 32 // patch_size  # 16
        self.stages = nn.ModuleList()
        cursor = 0
        for i, (depth, heads) in enumerate(zip(depths, num_heads)):
            dim = embed_dim * (2 ** i)  # 64, 128, 256
            self.stages.append(SwinStage(
                dim=dim,
                depth=depth,
                num_heads=heads,
                window_size=window_size,
                resolution=resolution,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cursor:cursor + depth],
                downsample=(i < len(depths) - 1),  # no downsample after last stage
            ))
            cursor += depth
            if i < len(depths) - 1:
                resolution //= 2  # 16 -> 8 -> 4

        self.norm = nn.LayerNorm(embed_dim * (2 ** (len(depths) - 1)))  # 256
        self.head = nn.Linear(embed_dim * (2 ** (len(depths) - 1)), num_classes)

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
        # [B, 3, 32, 32] -> [B, 256, 64]
        x = self.patch_embed(x)
        # Stages: [B,256,64] -> [B,64,128] -> [B,16,256]
        for stage in self.stages:
            x = stage(x)
        # [B, 16, 256] -> pool -> [B, 256] -> [B, 10]
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)
