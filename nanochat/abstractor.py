"""
Production Hadamard dual-attention implementation.

This module intentionally keeps a minimal, GPT-compatible surface.
Legacy experimental implementations are preserved verbatim in
`nanochat/abstractor_reference.py`.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from nanochat.attention_utils import apply_rotary_emb, qk_norm
from nanochat.flash_attention import flash_attn


class NullSymbolRetriever(nn.Module):
    """Placeholder retriever for API compatibility."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        return None


class HadamardDualAttention(nn.Module):
    """
    Fused dual attention with shared attention score computation.

    The standard self-attention value branch and Hadamard relational branch
    both use the same Q/K attention path. A single flash-attention call runs
    over concatenated values `[v_sa, xk_rel]`, then outputs are split and
    combined back into a single residual stream.

    TODO: add KV-cache support for this module.
    TODO: add optional structured debug/intermediate cache for interpretability.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head

        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0

        proportion = float(config.dual_rel_head_proportion)
        assert 0.0 <= proportion <= 1.0, "dual_rel_head_proportion must be in [0, 1]"
        self.n_heads_ra = max(0, min(self.n_head, int(round(self.n_head * proportion))))
        self.n_heads_sa = self.n_head - self.n_heads_ra

        # Shared Q/K attention path.
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)

        # Branch values: standard SA values and relational xk_rel values.
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_xk_rel = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)

        # Hadamard relational branch query projection (RA heads only).
        self.c_q_rel = None
        if self.n_heads_ra > 0:
            self.c_q_rel = nn.Linear(self.n_embd, self.n_heads_ra * self.head_dim, bias=False)

        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

        self.residual_gate_channels = config.residual_gate_channels
        assert self.residual_gate_channels > 0
        assert self.residual_gate_channels <= self.n_embd

        self.use_layer_residual = self._has_residual_layer(
            layer_idx,
            config.n_layer,
            config.use_residual_augmentation,
            config.residual_stride,
        )

        # Shared residual source (from GPT value embeddings), separate gates per branch.
        self.sa_residual_gate = None
        self.ra_residual_gate = None
        if self.use_layer_residual and self.n_heads_sa > 0:
            self.sa_residual_gate = nn.Linear(self.residual_gate_channels, self.n_kv_head, bias=False)
        if self.use_layer_residual and self.n_heads_ra > 0:
            self.ra_residual_gate = nn.Linear(self.residual_gate_channels, self.n_kv_head, bias=False)

    @staticmethod
    def _has_residual_layer(
        layer_idx: int,
        n_layer: int,
        use_residual_augmentation: bool,
        residual_stride: int,
    ) -> bool:
        if not use_residual_augmentation:
            return False
        assert residual_stride >= 1, "residual_stride must be >= 1"
        if layer_idx == n_layer - 1:
            return True
        return layer_idx % residual_stride == (n_layer - 1) % residual_stride

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        # Tensor shape legend:
        # B=batch size, T=sequence length, C=model width, Hq=query heads, Hkv=kv heads, D=head_dim.
        # x: [B, T, C]
        bsz, seqlen, _ = x.size()

        q = self.c_q(x).view(bsz, seqlen, self.n_head, self.head_dim)  # [B, T, Hq, D]
        k = self.c_k(x).view(bsz, seqlen, self.n_kv_head, self.head_dim)  # [B, T, Hkv, D]
        v_sa = self.c_v(x).view(bsz, seqlen, self.n_kv_head, self.head_dim)  # [B, T, Hkv, D]
        xk_rel = self.c_xk_rel(x).view(bsz, seqlen, self.n_kv_head, self.head_dim)  # [B, T, Hkv, D]

        # Shared residual source with branch-specific gates.
        # ve (if present): [B, T, Hkv * D]
        if ve is not None:
            ve = ve.view(bsz, seqlen, self.n_kv_head, self.head_dim)  # [B, T, Hkv, D]
            ve = ve.to(v_sa.dtype)  # [B, T, Hkv, D]
            if self.sa_residual_gate is not None:
                sa_gate = 2 * torch.sigmoid(self.sa_residual_gate(x[..., : self.residual_gate_channels]))  # [B, T, Hkv]
                v_sa = v_sa + sa_gate.unsqueeze(-1) * ve  # [B, T, Hkv, D]
            if self.ra_residual_gate is not None:
                ra_gate = 2 * torch.sigmoid(self.ra_residual_gate(x[..., : self.residual_gate_channels]))  # [B, T, Hkv]
                xk_rel = xk_rel + ra_gate.unsqueeze(-1) * ve  # [B, T, Hkv, D]

        cos, sin = cos_sin  # cos: [T, D/2], sin: [T, D/2]
        q = apply_rotary_emb(q, cos, sin)  # [B, T, Hq, D]
        k = apply_rotary_emb(k, cos, sin)  # [B, T, Hkv, D]
        q = qk_norm(q)  # [B, T, Hq, D]
        k = qk_norm(k)  # [B, T, Hkv, D]

        # One fused attention call over concatenated value channels.
        v_cat = torch.cat([v_sa, xk_rel], dim=-1)  # [B, T, Hkv, 2D]
        if kv_cache is None:
            y_cat = flash_attn.flash_attn_func(q, k, v_cat, causal=True, window_size=window_size)  # [B, T, Hq, 2D]
        else:
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            expected_v_dim = 2 * self.head_dim
            assert k_cache.shape[-1] == self.head_dim, (
                f"Expected dual attention k_cache last dim={self.head_dim}, got {k_cache.shape[-1]}"
            )
            assert v_cache.shape[-1] == expected_v_dim, (
                f"Expected dual attention v_cache last dim={expected_v_dim}, got {v_cache.shape[-1]}. "
                f"Construct KVCache with v_head_dim={expected_v_dim} for hadamard_dual."
            )
            y_cat = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v_cat,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )  # [B, T, Hq, 2D]
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(seqlen)

        y_sa_all, y_rel_all = y_cat.split(self.head_dim, dim=-1)  # each [B, T, Hq, D]

        branch_outputs = []
        if self.n_heads_sa > 0:
            branch_outputs.append(y_sa_all[:, :, : self.n_heads_sa, :])  # [B, T, Hsa, D]

        if self.n_heads_ra > 0:
            assert self.c_q_rel is not None
            q_rel = self.c_q_rel(x).view(bsz, seqlen, self.n_heads_ra, self.head_dim)  # [B, T, Hra, D]
            y_rel_ra = y_rel_all[:, :, self.n_heads_sa :, :]  # [B, T, Hra, D]
            branch_outputs.append(q_rel * y_rel_ra)  # [B, T, Hra, D]

        y = branch_outputs[0] if len(branch_outputs) == 1 else torch.cat(branch_outputs, dim=2)  # [B, T, Hq, D]
        y = y.contiguous().view(bsz, seqlen, self.n_embd)  # [B, T, C]
        y = self.c_proj(y)  # [B, T, C]
        return y


class DualAttention(nn.Module):
    """Public GPT-facing wrapper; currently backed by HadamardDualAttention."""

    def __init__(self, config, layer_idx: int, symbol_retriever: Optional[nn.Module] = None):
        super().__init__()
        self.symbol_retriever = symbol_retriever if symbol_retriever is not None else NullSymbolRetriever()
        self.impl = HadamardDualAttention(config, layer_idx)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        # Symbol retrieval is intentionally not part of the v1 production surface.
        return self.impl(x, ve, cos_sin, window_size, kv_cache)


__all__ = ["NullSymbolRetriever", "HadamardDualAttention", "DualAttention"]
