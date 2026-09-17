"""Checkpoint-compatible Dinov3D-Neuro inference backbone.

Architecture follows the published model card and tensor schema. Three-axis
RoPE extends the documented axial DINOv3 formulation. Upstream 3D code was not
publicly reachable when implemented: numerical parity remains unverified.
"""

import math
from collections.abc import Sequence
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


REPO_ID = "huggingbrain/Dinov3d-Neuro"
REVISION = "8f1355d6ba4159c1d6fdc2e1452212c0ab029481"
WEIGHTS_FILE = "eval/training_137499/teacher_checkpoint.pth"


class AxialRoPE3D(nn.Module):
    def __init__(self, head_dim: int = 66) -> None:
        super().__init__()
        self.register_buffer(
            "periods", 100.0 ** (torch.arange(head_dim // 6) / (head_dim // 6))
        )

    def forward(self, shape: Sequence[int]) -> tuple[Tensor, Tensor]:
        # Match the published bf16 positional-coordinate configuration.
        coords = [
            2
            * (torch.arange(n, device=self.periods.device, dtype=torch.bfloat16) + 0.5)
            / n
            - 1
            for n in shape
        ]
        grid = torch.stack(torch.meshgrid(*coords, indexing="ij"), dim=-1).reshape(
            -1, 3
        )
        angles = (
            (2 * math.pi * grid[..., None] / self.periods.to(torch.bfloat16))
            .flatten(1)
            .repeat(1, 2)
        )
        return angles.sin(), angles.cos()


def rotate_patches(x: Tensor, rope: tuple[Tensor, Tensor]) -> Tensor:
    sin, cos = rope
    patches = x[:, :, 1:].to(sin.dtype)
    first, second = patches.chunk(2, dim=-1)
    rotated = patches * cos + torch.cat((-second, first), dim=-1) * sin
    return torch.cat((x[:, :, :1], rotated.to(x.dtype)), dim=2)


class DinoAttention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, rope: tuple[Tensor, Tensor]) -> Tensor:
        b, n, d = x.shape
        q, k, v = (
            self.qkv(x)
            .reshape(b, n, 3, self.heads, d // self.heads)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        q, k = rotate_patches(q, rope), rotate_patches(k, rope)
        return self.proj(
            F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, d)
        )


class LayerScale(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), 1e-5))

    def forward(self, x: Tensor) -> Tensor:
        return self.gamma * x


class DinoMLP(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(dim, 4 * dim), nn.Linear(4 * dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class DinoBlock(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.norm1, self.norm2 = (
            nn.LayerNorm(dim, eps=1e-6),
            nn.LayerNorm(dim, eps=1e-6),
        )
        self.attn, self.mlp = DinoAttention(dim, heads), DinoMLP(dim)
        self.ls1, self.ls2 = LayerScale(dim), LayerScale(dim)

    def forward(self, x: Tensor, rope: tuple[Tensor, Tensor]) -> Tensor:
        x = x + self.ls1(self.attn(self.norm1(x), rope))
        return x + self.ls2(self.mlp(self.norm2(x)))


class PatchEmbed3D(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv3d(1, embed_dim, kernel_size=16, stride=16)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class DinoV3D(nn.Module):
    embed_dim = 792

    def __init__(self, gradient_checkpointing: bool = False) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.mask_token = nn.Parameter(
            torch.zeros(1, self.embed_dim), requires_grad=False
        )
        self.patch_embed = PatchEmbed3D(self.embed_dim)
        self.rope_embed = AxialRoPE3D()
        self.blocks = nn.ModuleList([DinoBlock(self.embed_dim, 12) for _ in range(12)])
        self.norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.gradient_checkpointing = gradient_checkpointing

    def forward_features(self, x: Tensor) -> dict[str, Tensor]:
        x = self.patch_embed.proj(x)
        rope = self.rope_embed(x.shape[2:])
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        for block in self.blocks:
            x = (
                checkpoint(block, x, rope, use_reentrant=False)
                if self.gradient_checkpointing and self.training
                else block(x, rope)
            )
        x = self.norm(x)
        return {"x_norm_clstoken": x[:, 0], "x_norm_patchtokens": x[:, 1:]}

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_features(x)["x_norm_clstoken"]

    def load_pretrained(
        self, checkpoint_path: str | Path | None = None, revision: str = REVISION
    ) -> str:
        path = (
            Path(checkpoint_path)
            if checkpoint_path
            else Path(hf_hub_download(REPO_ID, WEIGHTS_FILE, revision=revision))
        )
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        teacher = payload["teacher"]
        state = {
            k.removeprefix("backbone."): v
            for k, v in teacher.items()
            if k.startswith("backbone.")
        }
        # SSL projection heads are intentionally excluded; all backbone tensors must match.
        self.load_state_dict(state, strict=True)
        return str(path)
