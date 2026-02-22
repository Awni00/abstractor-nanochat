"""Shared attention helpers used by GPT and abstractor attention modules."""

import torch
import torch.nn.functional as F


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to a (B, T, H, D) tensor."""
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=3)


def qk_norm(x: torch.Tensor) -> torch.Tensor:
    """Functional RMSNorm with no learnable params for attention Q/K tensors."""
    return F.rms_norm(x, (x.size(-1),))
