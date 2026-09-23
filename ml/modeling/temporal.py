"""Radiation encoder and packed, block-diagonal causal temporal attention."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class Residual3D(nn.Module):
    def __init__(self, channels: int, output: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(channels, output, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, output),
            nn.SiLU(),
            nn.Conv3d(output, output, 3, padding=1, bias=False),
            nn.GroupNorm(8, output),
        )
        self.skip = nn.Conv3d(channels, output, 1, stride=2, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return F.silu(self.body(x) + self.skip(x))


class RadiationEncoder(nn.Module):
    def __init__(self, output_dim: int = 256) -> None:
        super().__init__()
        self.network = nn.Sequential(
            Residual3D(1, 16),
            Residual3D(16, 32),
            Residual3D(32, 64),
            Residual3D(64, 128),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(128, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


def causal_allowed_mask(lesion_ids: Tensor) -> Tensor:
    """SDPA boolean True means allowed: same lesion and key at/before query."""
    index = torch.arange(len(lesion_ids), device=lesion_ids.device)
    return (lesion_ids[:, None] == lesion_ids[None, :]) & (
        index[:, None] >= index[None, :]
    )


class CausalBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.proj = nn.Linear(dim, dim * 3), nn.Linear(dim, dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
        )
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, allowed: Tensor) -> Tensor:
        n, dim = x.shape
        q, k, v = (
            self.qkv(self.norm1(x))
            .reshape(n, 3, self.heads, dim // self.heads)
            .permute(1, 2, 0, 3)
            .unbind(0)
        )
        attended = (
            F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=allowed,
                dropout_p=self.dropout if self.training else 0.0,
            )
            .transpose(0, 1)
            .reshape(n, dim)
        )
        x = x + self.residual_dropout(self.proj(attended))
        return x + self.residual_dropout(self.ffn(self.norm2(x)))


class PackedCausalTransformer(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        heads: int = 8,
        layers: int = 3,
        dropout: float = 0.1,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList(
            [CausalBlock(dim, heads, dropout) for _ in range(layers)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x: Tensor, lesion_ids: Tensor, positions: Tensor) -> Tensor:
        frequency = torch.exp(
            torch.arange(0, self.dim, 2, device=x.device, dtype=torch.float32)
            * (-math.log(10000.0) / self.dim)
        )
        angles = positions.float()[:, None] * frequency
        encoding = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1)
        x = x + encoding.to(x.dtype)
        allowed = causal_allowed_mask(lesion_ids)
        for layer in self.layers:
            x = layer(x, allowed)
        return self.head(self.norm(x))


def relative_to_absolute_pos(pos: Tensor, step_x: float, step_y: float) -> Tensor:
    pos = pos.sigmoid()
    h, w = pos.shape[1:3]

    anchor_x = torch.arange(w, dtype=torch.float32, device=pos.device) * step_x
    anchor_y = torch.arange(h, dtype=torch.float32, device=pos.device) * step_y

    absolute_x = pos[..., 0] * step_x + anchor_x
    absolute_y = pos[..., 1] * step_y + anchor_y.unsqueeze(1)
    return torch.stack((absolute_x, absolute_y), dim=-1)
