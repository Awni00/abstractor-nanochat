"""
Legacy abstractor implementation snapshot.

This file preserves the full pre-refactor code as reference only.
The production implementation now lives in `nanochat/abstractor.py`.
"""

from typing import Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from nanochat.flash_attention import flash_attn


class NullSymbolRetriever(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        return None

class SymbolicAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_symbols: int,
        dropout: float = 0.0,
        scale: Optional[float] = None,
        trainable_symbols: bool = True,
        query_bias: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_symbols = n_symbols
        self.dropout = dropout
        self.scale = scale
        self.trainable_symbols = trainable_symbols
        self.query_bias = query_bias

        self.q_proj = Linear(self.d_model, self.d_model, bias=self.query_bias)
        self.template_features = nn.Parameter(
            trunc_normal_init_(torch.empty(self.n_symbols, self.d_model), std=1.0 / (self.d_model**0.5)),
        )
        self.symbol_library = nn.Parameter(
            trunc_normal_init_(torch.empty(self.n_symbols, self.d_model), std=1.0 / (self.d_model**0.5)),
            requires_grad=trainable_symbols
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.size()

        query = self.q_proj(x)
        query = query.view(batch_size, seq_len, self.n_heads, dim // self.n_heads).transpose(1, 2)

        key = self.template_features.view(self.n_symbols, self.n_heads, self.d_model // self.n_heads).transpose(0, 1)
        key = key.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(query.dtype)

        value = self.symbol_library.view(self.n_symbols, self.n_heads, self.d_model // self.n_heads).transpose(0, 1)
        value = value.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(query.dtype)

        retrieved_symbols = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, scale=self.scale, dropout_p=self.dropout, attn_mask=None, is_causal=False
        )
        retrieved_symbols = retrieved_symbols.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)

        return retrieved_symbols


class PositionalSymbolRetriever(nn.Module):
    def __init__(self, d_model: int, max_length: int, sinusoidal: bool = False):
        super().__init__()
        self.symbol_dim = d_model
        self.max_length = max_length
        self.sinusoidal = sinusoidal
        self.symbol_library = nn.Embedding(self.max_length, self.symbol_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        batch_size, seq_len, _ = x.size()

        pos = torch.arange(0, seq_len, dtype=torch.long, device=device)
        retrieved_symbols = self.symbol_library(pos).unsqueeze(0).repeat(batch_size, 1, 1)
        retrieved_symbols = retrieved_symbols.to(x.dtype)
        return retrieved_symbols


class PositionRelativeSymbolRetrieverLegacy(nn.Module):
    """Legacy version of PositionRelativeSymbolRetriever for compatibility."""
    def __init__(self, d_model: int, max_rel_pos: int):
        super().__init__()
        self.symbol_dim = d_model
        self.max_rel_pos = max_rel_pos
        self.rel_pos_enc = RelativePositionalEncoding(dim=d_model, max_rel_pos=max_rel_pos)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[1]
        device = x.device
        return self.rel_pos_enc(length).to(device).to(x.dtype)

class PositionRelativeSymbolRetriever(nn.Module):
    """
    Stores only the 2 * max_rel_pos + 1 embeddings needed for relative positions.
    """

    def __init__(self, d_model: int, max_rel_pos: int):
        super().__init__()
        self.symbol_dim = d_model
        self.max_rel_pos = max_rel_pos
        self.rel_pos_embeddings = nn.Parameter(torch.empty(max_rel_pos * 2 + 1, d_model))
        nn.init.xavier_uniform_(self.rel_pos_embeddings)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        embeddings = self.rel_pos_embeddings.to(device=x.device, dtype=x.dtype)
        return embeddings


class RelationalSymbolicAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        rel_n_heads: int,
        symbolic_attn_n_heads: int,
        n_symbols: int,
        nbhd_delta: int,
        causal_nbhd: bool = True,
        include_self: bool = False,
        normalize_rels: bool = True,
        dropout: float = 0.0,
        rel_scale: Optional[float] = None,
        symbolic_attn_scale: Optional[float] = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.rel_n_heads = rel_n_heads
        self.symbolic_attn_n_heads = symbolic_attn_n_heads
        self.n_symbols = n_symbols
        self.nbhd_delta = nbhd_delta
        self.causal_nbhd = causal_nbhd
        self.dropout = dropout
        self.rel_scale = rel_scale if rel_scale is not None else (d_model // rel_n_heads) ** -0.5
        self.symbolic_attn_scale = symbolic_attn_scale
        self.include_self = include_self
        self.normalize_rels = normalize_rels

        self.nbhd_rel_dim = self._compute_nbhd_rel_dim(rel_n_heads, nbhd_delta, causal_nbhd, include_self)

        self.symbolic_attention = SymbolicAttention(d_model, symbolic_attn_n_heads, n_symbols, dropout, symbolic_attn_scale)
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.model_dim_proj = Linear(self.nbhd_rel_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.size()

        query = self.q_proj(x)
        key = self.k_proj(x)

        query = query.view(batch_size, seq_len, self.rel_n_heads, self.d_model // self.rel_n_heads).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.rel_n_heads, self.d_model // self.rel_n_heads).transpose(1, 2)

        if self.causal_nbhd:
            neighbor_mask = self.compute_causal_neighbor_mask(seq_len, self.nbhd_delta, self.include_self)
        else:
            neighbor_mask = self.compute_neighbor_mask(seq_len, self.nbhd_delta, self.include_self)

        neighborhood_keys = key[:, :, neighbor_mask]

        neighbor_rel_tensor = torch.einsum("bhid,bhijd->bhij", query, neighborhood_keys)

        if self.normalize_rels:
            neighbor_rel_tensor = torch.softmax(neighbor_rel_tensor * self.rel_scale, dim=-1)

        neighbor_rel_tensor = neighbor_rel_tensor.permute(0, 2, 3, 1)
        neighbor_rel_tensor = neighbor_rel_tensor.contiguous().view(batch_size, -1, self.nbhd_rel_dim)
        neighbor_rel_tensor = self.model_dim_proj(neighbor_rel_tensor)

        retrieved_symbols = self.symbolic_attention(neighbor_rel_tensor)

        return retrieved_symbols

    def _compute_nbhd_rel_dim(self, rel_n_heads, nbhd_delta, causal_nbhd, include_self):
        if causal_nbhd:
            if include_self:
                return rel_n_heads * (nbhd_delta + 1)
            return rel_n_heads * nbhd_delta

        if include_self:
            return rel_n_heads * (2 * nbhd_delta + 1)
        return rel_n_heads * (2 * nbhd_delta)

    @staticmethod
    def compute_neighbor_mask(n, delta, include_self=True):
        sequence = torch.arange(n).unsqueeze(1)
        if include_self:
            neighborhood = torch.arange(-delta, delta + 1).unsqueeze(0)
        else:
            neighborhood = torch.concat([torch.arange(-delta, 0), torch.arange(1, delta + 1)]).unsqueeze(0)

        mask = sequence + neighborhood
        mask = mask.clamp(0, n - 1)
        return mask

    @staticmethod
    def compute_causal_neighbor_mask(n, delta, include_self=False):
        sequence = torch.arange(n).unsqueeze(1)
        if include_self:
            neighborhood = torch.arange(delta + 1).unsqueeze(0)
        else:
            neighborhood = torch.arange(1, delta + 1).unsqueeze(0)

        mask = sequence - neighborhood
        mask = mask.clamp(0, n - 1)
        return mask


class Attention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        key_dim: Optional[int] = None,
        n_kv_heads: Optional[int] = None,
        add_bias_kv: bool = False,
        add_bias_out: bool = False,
        total_n_heads: Optional[int] = None,
        disable_flash_attention: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.dropout = dropout
        self.add_bias_kv = add_bias_kv
        self.add_bias_out = add_bias_out
        self.total_n_heads = n_heads if total_n_heads is None else total_n_heads
        self.disable_flash_attention = disable_flash_attention

        self.key_dim = key_dim if key_dim is not None else self.d_model // self.total_n_heads
        self.n_rep_kv = self.n_heads // self.n_kv_heads
        self.head_dim = self.d_model // self.total_n_heads
        assert self.n_heads % self.n_kv_heads == 0
        assert self.n_rep_kv * self.n_kv_heads == self.n_heads
        assert self.total_n_heads * self.head_dim == self.d_model

        self.attn_scale = 1 / math.sqrt(self.head_dim)

        self.wq = Linear(self.d_model, self.n_heads * self.key_dim, bias=False)
        self.wk = Linear(self.d_model, self.n_kv_heads * self.key_dim, bias=self.add_bias_kv)
        self.wv = Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=self.add_bias_kv)
        self.wo = Linear(self.n_heads * self.head_dim, self.n_heads * self.head_dim, bias=self.add_bias_out)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, qseqlen, _ = query.shape
        bsz, kseqlen, _ = key.shape
        bsz, vseqlen, _ = value.shape
        assert kseqlen == vseqlen

        need_weights = need_weights or self.disable_flash_attention

        xq, xk, xv = self.wq(query), self.wk(key), self.wv(value)
        xq = xq.view(bsz, qseqlen, self.n_heads, self.key_dim)
        xk = xk.view(bsz, kseqlen, self.n_kv_heads, self.key_dim)
        xv = xv.view(bsz, vseqlen, self.n_kv_heads, self.head_dim)

        if freqs_cos is not None and freqs_sin is not None:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        if self.n_rep_kv != 1:
            xk = repeat_kv(xk, self.n_rep_kv)
            xv = repeat_kv(xv, self.n_rep_kv)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        if not need_weights:
            output = torch.nn.functional.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal,
                scale=self.attn_scale,
            )
            scores = None
        else:
            assert not (attn_mask is not None and is_causal)
            if is_causal and attn_mask is None:
                attn_mask = compute_causal_mask(qseqlen, device=xq.device)

            scores = torch.matmul(xq, xk.transpose(2, 3)) * self.attn_scale

            if attn_mask is not None:
                attn_mask_ = torch.zeros(qseqlen, kseqlen, dtype=xq.dtype, device=xq.device).masked_fill(
                    attn_mask.logical_not(), float("-inf")
                )
                scores = scores + attn_mask_

            scores = torch.nn.functional.softmax(scores, dim=-1)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)

        output = output.transpose(1, 2).contiguous().view(bsz, qseqlen, -1)
        output = self.wo(output)
        output = self.resid_dropout(output)

        return output, scores


def get_activation_function(name: str):
    activation_dict = {
        "relu": nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "gelu": nn.GELU(approximate="tanh"),
        "silu": nn.SiLU(),
        "softmax": nn.Softmax(dim=-1),
        "identity": nn.Identity(),
    }
    if name in activation_dict:
        return activation_dict[name]
    raise ValueError(f"Activation function {name} not found in {list(activation_dict.keys())}")


class RelationalAttention(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_relations: Optional[int] = None,
        dropout: float = 0.0,
        key_dim: Optional[int] = None,
        n_kv_heads: Optional[int] = None,
        rel_activation: str = "identity",
        rel_proj_dim: Optional[int] = None,
        add_bias_kv: bool = False,
        add_bias_out: bool = False,
        total_n_heads: Optional[int] = None,
        symmetric_rels: bool = False,
        use_relative_positional_symbols: bool = False,
        use_flash=True, # New parameter to enable FlashAttention when possible
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_relations = n_relations if n_relations is not None else n_heads
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.rel_activation = rel_activation
        self.rel_activation_ = get_activation_function(rel_activation)
        self.symmetric_rels = symmetric_rels
        self.dropout = dropout
        self.add_bias_kv = add_bias_kv
        self.add_bias_out = add_bias_out
        self.use_relative_positional_symbols = use_relative_positional_symbols
        if self.use_relative_positional_symbols:
            raise ValueError("Use RelationalAttentionWithPositionRelativeSymbols instead for relative positional symbols.")

        self.use_flash = use_flash
        if self.use_flash and rel_activation != "identity":
            print("Warning: FlashAttention can only be used with 'identity' rel_activation. Disabling FlashAttention.")
            self.use_flash = False

        self.total_n_heads = n_heads if total_n_heads is None else total_n_heads
        self.head_dim = self.d_model // self.total_n_heads
        self.n_rep_kv = self.n_heads // self.n_kv_heads
        self.key_dim = key_dim if key_dim is not None else self.head_dim
        self.rel_proj_dim = rel_proj_dim if rel_proj_dim is not None else (self.head_dim * self.n_heads) // self.n_relations

        assert self.n_heads % self.n_kv_heads == 0
        assert self.n_rep_kv * self.n_kv_heads == self.n_heads
        assert self.total_n_heads * self.head_dim == self.d_model
        assert self.rel_proj_dim * self.n_relations == self.head_dim * self.n_heads

        self.attn_scale = 1 / math.sqrt(self.head_dim)
        self.rel_scale = 1 / math.sqrt(self.rel_proj_dim)

        self.wq_attn = Linear(self.d_model, self.n_heads * self.key_dim, bias=False)
        self.wk_attn = Linear(self.d_model, self.n_kv_heads * self.key_dim, bias=self.add_bias_kv)

        self.wq_rel = Linear(self.d_model, self.n_relations * self.rel_proj_dim, bias=False)
        if self.symmetric_rels:
            self.wk_rel = self.wq_rel
        else:
            self.wk_rel = Linear(self.d_model, self.n_relations * self.rel_proj_dim, bias=False)

        self.wr = nn.Parameter(torch.empty(self.n_heads, self.head_dim, self.n_relations))
        torch.nn.init.kaiming_uniform_(self.wr, a=math.sqrt(5))
        self.wv = Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=self.add_bias_kv)
        self.wo = Linear(self.n_heads * self.head_dim, self.n_heads * self.head_dim, bias=self.add_bias_out)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

    def forward(
        self,
        x: torch.Tensor,
        symbols: torch.Tensor,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seqlen, _ = x.shape

        # Attention Q/K projections
        xq_attn, xk_attn = self.wq_attn(x), self.wk_attn(x)
        xq_attn = xq_attn.view(bsz, seqlen, self.n_heads, self.key_dim)
        xk_attn = xk_attn.view(bsz, seqlen, self.n_kv_heads, self.key_dim)

        if freqs_cos is not None and freqs_sin is not None:
            xq_attn, xk_attn = apply_rotary_emb(xq_attn, xk_attn, freqs_cos, freqs_sin)

        if self.n_rep_kv != 1:
            xk_attn = repeat_kv(xk_attn, self.n_rep_kv)

        # Determine if we can use optimized paths
        # FlashAttention (External) usually requires head_dim == rel_proj_dim
        use_flash_identity_opt = (
            self.rel_activation == "identity"
            and self.head_dim == self.rel_proj_dim
            and HAS_FLASH_ATTN
            and flash_attn_v3 is not None
            and x.dtype in [torch.bfloat16, torch.float16]
            and attn_mask is None
        )

        # SDPA Path is more flexible and supports mismatched dimensions
        use_sdpa_identity_opt = (
            self.rel_activation == "identity"
            and not use_flash_identity_opt # Only use if Flash isn't already taking it
            # SDPA is available in any modern torch
            and attn_mask is None
        )

        # Relation Q/K projections
        xq_rel, xk_rel = self.wq_rel(x), self.wk_rel(x)
        xq_rel = xq_rel.view(bsz, seqlen, self.n_relations, self.rel_proj_dim)
        xk_rel = xk_rel.view(bsz, seqlen, self.n_relations, self.rel_proj_dim)

        # --- Relations Branch ---
        attn_scores = None
        relations = None

        if self.use_flash and not (use_flash_identity_opt or use_sdpa_identity_opt):
            raise ValueError(
                "Module's attribute 'use_flash' is True but conditions for using FlashAttention are not met."
            )

        if use_flash_identity_opt and self.use_flash:
            # FLASH PATH (O(L) memory, strict dims)
            # ... (Rest of existing expansion logic) ...
            q_flash = xq_attn.unsqueeze(3).expand(bsz, seqlen, self.n_heads, self.n_relations, self.key_dim)
            q_flash = q_flash.reshape(bsz, seqlen, self.n_heads * self.n_relations, self.key_dim)
            k_flash = xk_attn.unsqueeze(3).expand(bsz, seqlen, self.n_heads, self.n_relations, self.key_dim)
            k_flash = k_flash.reshape(bsz, seqlen, self.n_heads * self.n_relations, self.key_dim)
            v_flash = xk_rel.unsqueeze(2).expand(bsz, seqlen, self.n_heads, self.n_relations, self.rel_proj_dim)
            v_flash = v_flash.reshape(bsz, seqlen, self.n_heads * self.n_relations, self.rel_proj_dim)

            z = flash_attn_v3(q_flash, k_flash, v_flash, causal=is_causal, softmax_scale=self.attn_scale)
            z = z.view(bsz, seqlen, self.n_heads, self.n_relations, self.rel_proj_dim)
            m = (xq_rel.unsqueeze(2) * z).sum(dim=-1) * self.rel_scale
            attended_relations = torch.einsum("blhr,hdr->blhd", m, self.wr.to(m.dtype))

        elif use_sdpa_identity_opt and self.use_flash:
            # SDPA PATH (O(L) memory, flexible dims)
            # Repeat/Expand logic same as Flash but for SDPA format (B, H, L, D)

            # Prepare Q: [B, H*Nr, L, D]
            q_sdpa = xq_attn.transpose(1, 2).unsqueeze(2).expand(bsz, self.n_heads, self.n_relations, seqlen, self.key_dim)
            q_sdpa = q_sdpa.reshape(bsz, self.n_heads * self.n_relations, seqlen, self.key_dim)

            # Prepare K: [B, H*Nr, L, D]
            k_sdpa = xk_attn.transpose(1, 2).unsqueeze(2).expand(bsz, self.n_heads, self.n_relations, seqlen, self.key_dim)
            k_sdpa = k_sdpa.reshape(bsz, self.n_heads * self.n_relations, seqlen, self.key_dim)

            # Prepare V (xk_rel): [B, H*Nr, L, D_rel]
            v_sdpa = xk_rel.transpose(1, 2).unsqueeze(1).expand(bsz, self.n_heads, self.n_relations, seqlen, self.rel_proj_dim)
            v_sdpa = v_sdpa.reshape(bsz, self.n_heads * self.n_relations, seqlen, self.rel_proj_dim)

            # SDPA handles mismatched D_attn vs D_rel automatically
            z = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa,
                is_causal=is_causal,
                scale=self.attn_scale,
                dropout_p=self.dropout if self.training else 0.0
            )

            # Reshape back to [B, H, Nr, L, D_rel]
            z = z.view(bsz, self.n_heads, self.n_relations, seqlen, self.rel_proj_dim)

            # Compute M = Q^R . Z
            # xq_rel: [B, L, Nr, D_rel] -> [B, 1, Nr, L, D_rel]
            xq_rel_t = xq_rel.permute(0, 2, 1, 3).unsqueeze(1)

            # m shape: [B, H, Nr, L]
            m = (xq_rel_t * z).sum(dim=-1) * self.rel_scale

            # Project: [B, H, Nr, L] -> [B, L, H, D]
            # Use same einsum: "blhr,hdr->blhd"
            # m was [B, H, Nr, L] -> transpose to [B, L, H, Nr]
            m = m.permute(0, 3, 1, 2)
            attended_relations = torch.einsum("blhr,hdr->blhd", m, self.wr.to(m.dtype))

        else:
            # STANDARD / FALLBACK PATH
            xq_attn_t = xq_attn.transpose(1, 2)
            xk_attn_t = xk_attn.transpose(1, 2)

            attn_scores = torch.matmul(xq_attn_t, xk_attn_t.transpose(2, 3)) * self.attn_scale
            if is_causal:
                 mask = compute_causal_mask(seqlen, device=xq_attn.device)
                 attn_scores = attn_scores.masked_fill(mask.logical_not(), float("-inf"))
            if attn_mask is not None:
                attn_mask_ = torch.zeros(seqlen, seqlen, dtype=xq_attn.dtype, device=xq_attn.device).masked_fill(
                    attn_mask.logical_not(), float("-inf")
                )
                attn_scores = attn_scores + attn_mask_

            attn_scores = nn.functional.softmax(attn_scores, dim=-1)
            attn_scores = self.attn_dropout(attn_scores)

            # xq_rel: [B, L, Nr, D] -> [B, Nr, L, D]
            xq_rel_t = xq_rel.transpose(1, 2)
            xk_rel_t = xk_rel.transpose(1, 2)

            # [B, Nr, L, D] @ [B, Nr, D, L] -> [B, Nr, L, L]
            relations = torch.matmul(xq_rel_t, xk_rel_t.transpose(2, 3)) * self.rel_scale
            relations = self.rel_activation_(relations)
            # (b nr i j) -> (b i j nr)
            relations = relations.permute(0, 2, 3, 1).contiguous()

            # attended_relations = sum_j (attn_scores[b,h,i,j] * relations[b,i,j,nr])
            # einsum "bhij,bijr->bihr"
            attended_relations_intermediate = torch.einsum("bhij,bijr->bihr", attn_scores, relations)
            attended_relations = torch.einsum("bihr,hdr->bihd", attended_relations_intermediate, self.wr.to(attended_relations_intermediate.dtype))

            # Match shape of optimized path [B, L, H, D]
            # einsum output is [B, L, H, D] (i=L)
            pass

        # --- Symbols Branch ---
        sv = self.wv(symbols)
        if self.use_relative_positional_symbols:
            assert symbols.shape[0] == symbols.shape[1] == seqlen
            sv = sv.view(seqlen, seqlen, self.n_kv_heads, self.head_dim)
            if use_flash_identity_opt:
                # We need scores for this case. Recompute them if we took the optimized path.
                # Or just fallback entirely?
                # The constraints check for use_flash_identity_opt relies on standard behavior.
                # If use_relative_positional_symbols is True, we probably shouldn't use the opt
                # unless we want to compute scores just for this.
                # Given 'rel_activation="identity"' constraint, it's independent of symbol type.
                # BUT we need scores.
                # So if use_relative_positional_symbols, we MUST compute scores.
                # This negates the memory benefit.
                # So we should probably disable use_flash_identity_opt if use_relative_positional_symbols is True.
                # But let's handle it gracefully: recompute scores here if needed.
                xq_attn_t = xq_attn.transpose(1, 2)
                xk_attn_t = xk_attn.transpose(1, 2)
                attn_scores = torch.matmul(xq_attn_t, xk_attn_t.transpose(2, 3)) * self.attn_scale
                if is_causal:
                     mask = compute_causal_mask(seqlen, device=xq_attn.device)
                     attn_scores = attn_scores.masked_fill(mask.logical_not(), float("-inf"))
                attn_scores = nn.functional.softmax(attn_scores, dim=-1)
                attn_scores = self.attn_dropout(attn_scores)

            # einsum "bhij,ijhd->bihd"
            attended_symbols = torch.einsum("bhij,ijhd->bihd", attn_scores, sv)

        else:
            # Standard symbols
            sv = self.wv(symbols).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
            if self.n_rep_kv != 1:
                sv = repeat_kv(sv.transpose(1, 2), self.n_rep_kv).transpose(1, 2)

            if use_flash_identity_opt:
                # Use FlashAttention for symbols too!
                attended_symbols = flash_attn_v3(
                    xq_attn, xk_attn, sv,
                    causal=is_causal,
                    softmax_scale=self.attn_scale
                )
                # Output: [B, L, H, D]
            elif use_sdpa_identity_opt:
                # Use SDPA for symbols too!
                attended_symbols = F.scaled_dot_product_attention(
                    xq_attn.transpose(1, 2),
                    xk_attn.transpose(1, 2),
                    sv.transpose(1, 2),
                    is_causal=is_causal,
                    scale=self.attn_scale,
                    dropout_p=self.dropout if self.training else 0.0
                )
                attended_symbols = attended_symbols.transpose(1, 2)
            else:
                # Use bmm or einsum (standard path)
                sv_t = sv.transpose(1, 2) # [B, H, L, D]
                # attn_scores [B, H, L, L] @ sv_t [B, H, L, D] -> [B, H, L, D]
                attended_symbols = torch.matmul(attn_scores, sv_t)
                attended_symbols = attended_symbols.transpose(1, 2)

        output = attended_symbols + attended_relations

        # Use reshape instead of rearrange (handles non-contiguous)
        output = output.reshape(bsz, seqlen, self.n_heads * self.head_dim)
        output = self.wo(output)
        output = self.resid_dropout(output)

        return output, attn_scores, relations


class RelationalAttentionWithPositionRelativeSymbols(nn.Module):
    """
    Relational attention that uses position-relative symbols (more efficient than implementation in RelationalAttention).

    This module matches the computation of RelationalAttention(use_relative_positional_symbols=True)
    but is faster and more memory-efficient by avoiding the expansion to [L, L, D] for relative symbols.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_rel_pos: int,
        n_relations: Optional[int] = None,
        dropout: float = 0.0,
        key_dim: Optional[int] = None,
        n_kv_heads: Optional[int] = None,
        rel_activation: str = "identity",
        rel_proj_dim: Optional[int] = None,
        add_bias_kv: bool = False,
        add_bias_out: bool = False,
        total_n_heads: Optional[int] = None,
        symmetric_rels: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_relations = n_relations if n_relations is not None else n_heads
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.rel_activation = rel_activation
        self.rel_activation_ = get_activation_function(rel_activation)
        self.symmetric_rels = symmetric_rels
        self.dropout = dropout
        self.add_bias_kv = add_bias_kv
        self.add_bias_out = add_bias_out
        self.max_rel_pos = max_rel_pos

        self.total_n_heads = n_heads if total_n_heads is None else total_n_heads
        self.head_dim = self.d_model // self.total_n_heads
        self.n_rep_kv = self.n_heads // self.n_kv_heads
        self.key_dim = key_dim if key_dim is not None else self.head_dim
        self.rel_proj_dim = rel_proj_dim if rel_proj_dim is not None else (self.head_dim * self.n_heads) // self.n_relations

        assert self.n_heads % self.n_kv_heads == 0
        assert self.n_rep_kv * self.n_kv_heads == self.n_heads
        assert self.total_n_heads * self.head_dim == self.d_model
        assert self.rel_proj_dim * self.n_relations == self.head_dim * self.n_heads

        self.attn_scale = 1 / math.sqrt(self.head_dim)
        self.rel_scale = 1 / math.sqrt(self.rel_proj_dim)

        self.wq_attn = Linear(self.d_model, self.n_heads * self.key_dim, bias=False)
        self.wk_attn = Linear(self.d_model, self.n_kv_heads * self.key_dim, bias=self.add_bias_kv)

        self.wq_rel = Linear(self.d_model, self.n_relations * self.rel_proj_dim, bias=False)
        if self.symmetric_rels:
            self.wk_rel = self.wq_rel
        else:
            self.wk_rel = Linear(self.d_model, self.n_relations * self.rel_proj_dim, bias=False)

        self.wr = nn.Parameter(torch.empty(self.n_heads, self.head_dim, self.n_relations))
        torch.nn.init.kaiming_uniform_(self.wr, a=math.sqrt(5))
        self.wv = Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=self.add_bias_kv)
        self.wo = Linear(self.n_heads * self.head_dim, self.n_heads * self.head_dim, bias=self.add_bias_out)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        self.symbol_retriever = PositionRelativeSymbolRetriever(d_model, max_rel_pos)

    def _get_rel_position_indices(self, seqlen: int, device: torch.device) -> torch.Tensor:
        """Get or compute cached relative position indices matrix."""
        cache_attr = '_cached_rel_pos'
        if not hasattr(self, cache_attr) or getattr(self, cache_attr).shape[0] < seqlen or getattr(self, cache_attr).device != device:
            row_idx = torch.arange(seqlen, device=device)
            col_idx = torch.arange(seqlen, device=device)
            rel_positions = col_idx.unsqueeze(0) - row_idx.unsqueeze(1)  # [L, L]
            # Clamp and shift to [0, 2*max_rel_pos]
            rel_positions = rel_positions.clamp(-self.max_rel_pos, self.max_rel_pos) + self.max_rel_pos
            self.register_buffer(cache_attr, rel_positions, persistent=False)
        return getattr(self, cache_attr)[:seqlen, :seqlen]

    def _attend_relative_symbols(
        self, attn_scores: torch.Tensor, projected_rel_symbols: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute attention-weighted sum of relative symbols using vectorized gather.

        Args:
            attn_scores: [bsz, n_heads, seqlen, seqlen] attention weights
            projected_rel_symbols: [2*max_rel_pos+1, n_heads, head_dim] projected symbol embeddings

        Returns:
            [bsz, seqlen, n_heads, head_dim] attended symbol values
        """
        bsz, n_heads, seqlen, _ = attn_scores.shape

        # Get relative position indices: rel_idx[i, j] = clip(j - i, -max, max) + max_rel_pos
        rel_positions = self._get_rel_position_indices(seqlen, attn_scores.device)  # [L, L]

        # Gather symbols for each (i, j) position: [L, L] -> [L, L, n_heads, head_dim]
        gathered = projected_rel_symbols[rel_positions]  # [L, L, nh, hd]

        # Weighted sum: output[b, i, h, d] = sum_j attn[b, h, i, j] * gathered[i, j, h, d]
        output = torch.einsum("bhij,ijhd->bihd", attn_scores, gathered)

        return output

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seqlen, _ = x.shape

        xq_attn, xk_attn = self.wq_attn(x), self.wk_attn(x)
        xq_attn = rearrange(xq_attn, "b l (nh hd) -> b l nh hd", nh=self.n_heads)
        xk_attn = rearrange(xk_attn, "b l (nh hd) -> b l nh hd", nh=self.n_kv_heads)

        if freqs_cos is not None and freqs_sin is not None:
            xq_attn, xk_attn = apply_rotary_emb(xq_attn, xk_attn, freqs_cos, freqs_sin)

        if self.n_rep_kv != 1:
            xk_attn = repeat_kv(xk_attn, self.n_rep_kv)

        xq_attn = xq_attn.transpose(1, 2)
        xk_attn = xk_attn.transpose(1, 2)

        assert not (attn_mask is not None and is_causal)
        if is_causal and attn_mask is None:
            attn_mask = compute_causal_mask(seqlen, device=xq_attn.device)

        attn_scores = torch.matmul(xq_attn, xk_attn.transpose(2, 3)) * self.attn_scale

        if attn_mask is not None:
            attn_mask_ = torch.zeros(seqlen, seqlen, dtype=xq_attn.dtype, device=xq_attn.device).masked_fill(
                attn_mask.logical_not(), float("-inf")
            )
            attn_scores = attn_scores + attn_mask_

        attn_scores = nn.functional.softmax(attn_scores, dim=-1)
        attn_scores = self.attn_dropout(attn_scores)

        xq_rel, xk_rel = self.wq_rel(x), self.wk_rel(x)
        xq_rel = rearrange(xq_rel, "b l (nr rd) -> b l nr rd", nr=self.n_relations)
        xk_rel = rearrange(xk_rel, "b l (nr rd) -> b l nr rd", nr=self.n_relations)

        rel_embeddings = self.symbol_retriever(x)
        sv = self.wv(rel_embeddings)
        sv = rearrange(sv, "r (nh hd) -> r nh hd", nh=self.n_kv_heads)

        if self.n_rep_kv != 1:
            sv = repeat_kv(sv.unsqueeze(0), self.n_rep_kv).squeeze(0)

        xq_rel = xq_rel.transpose(1, 2)
        xk_rel = xk_rel.transpose(1, 2)

        relations = torch.matmul(xq_rel, xk_rel.transpose(2, 3)) * self.rel_scale
        relations = self.rel_activation_(relations)
        relations = rearrange(relations, "b nr i j -> b i j nr")

        attended_symbols = self._attend_relative_symbols(attn_scores, sv)
        wr = self.wr.to(attended_symbols.dtype)
        attended_relations = torch.einsum("bhij,bijr->bihr", attn_scores, relations)
        attended_relations = torch.einsum("bihr,hdr->bihd", attended_relations, wr)
        output = attended_symbols + attended_relations

        output = rearrange(output, "b l nh hd -> b l (nh hd)")
        output = self.wo(output)
        output = self.resid_dropout(output)

        return output, attn_scores, relations


# Optimized variant of HadamardRelationalAttention.
# Replaces the legacy implementation.


class HadamardRelationalAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        key_dim: Optional[int] = None,
        n_kv_heads: Optional[int] = None,
        rel_activation: str = "identity",
        add_bias_kv: bool = False,
        add_bias_out: bool = False,
        total_n_heads: Optional[int] = None,
        symmetric_rels: bool = False,
        identity_rels: bool = False,
        use_relative_positional_symbols: bool = False,
        disable_symbols=False,
        **kwargs
    ):
        """
        An implementation of Hadamard Relational Attention.

        This is a variant of relational attention that uses Hadamard (element-wise) products
        to compute relations between objects. It supports optimized computation paths using
        Flash Attention or scaled dot-product attention when the relation activation is identity.

        The module supports symmetric relations, identity relation projections, position-relative
        symbolic embeddings, multi-query attention/grouped query attention, and control over
        total number of heads (for use with "dual attention").

        Parameters
        ----------
        d_model : int
            model dimension
        n_heads : int
            number of attention heads (query heads if n_kv_heads is set)
        dropout : float, optional
            dropout rate. By default 0.0
        key_dim : int, optional
            dimension of keys. If None, key_dim = head_dim. By default None
        n_kv_heads : int, optional
            number of key/value heads. used to implement multi-query attention or grouped query attention.
            n_kv_heads=1 corresponds to MQA, n_kv_heads > 1 corresponds to grouped query attention.
            n_kv_heads=n_heads is standard MHA. uses MHA when None. By default None
        rel_activation : str, optional
            name of activation function applied to relations. When 'identity', uses optimized
            Flash Attention or SDPA paths. By default 'identity'.
        add_bias_kv : bool, optional
            whether to use bias in key/value projections, by default False
        add_bias_out : bool, optional
            whether to use bias in out projection, by default False
        total_n_heads : int, optional
            total number of heads in dual attention (if using dual attention).
            used to ensure that concat(A, E) is of dimension d_model after concatenation.
            hence, output dimension is (d_model // total_heads) * n_heads.
            if None, total_heads = n_heads and output dimension is d_model
        symmetric_rels : bool, optional
            whether to use symmetric relations (shares wq_rel and wk_rel parameters), by default False
        identity_rels : bool, optional
            whether to use identity projections for relations (no learned projection), by default False
        use_relative_positional_symbols : bool, optional
            whether to use relative positional symbols of shape [len, len, dim], by default False
        disable_symbols : bool, optional
            whether to disable symbol computation entirely, by default False
        **kwargs
            additional keyword arguments (ignored)
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_relations = n_heads
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.rel_activation = rel_activation
        self.rel_activation_ = get_activation_function(rel_activation)
        self.symmetric_rels = symmetric_rels
        self.dropout = dropout
        self.add_bias_kv = add_bias_kv
        self.add_bias_out = add_bias_out
        self.identity_rels = identity_rels
        self.use_relative_positional_symbols = use_relative_positional_symbols

        self.total_n_heads = n_heads if total_n_heads is None else total_n_heads
        self.head_dim = self.d_model // self.total_n_heads
        self.n_rep_kv = self.n_heads // self.n_kv_heads
        self.key_dim = key_dim if key_dim is not None else self.head_dim
        self.disable_symbols = disable_symbols

        assert self.n_heads % self.n_kv_heads == 0
        assert self.n_rep_kv * self.n_kv_heads == self.n_heads
        assert self.total_n_heads * self.head_dim == self.d_model

        self.attn_scale = 1 / math.sqrt(self.head_dim)

        self.wq_attn = Linear(self.d_model, self.n_heads * self.key_dim, bias=False)
        self.wk_attn = Linear(self.d_model, self.n_kv_heads * self.key_dim, bias=self.add_bias_kv)

        if self.identity_rels:
            self.wq_rel = nn.Identity()
            self.wk_rel = nn.Identity()
        else:
            self.wq_rel = Linear(self.d_model, self.n_heads * self.key_dim, bias=False)
            if self.symmetric_rels:
                self.wk_rel = self.wq_rel
            else:
                self.wk_rel = Linear(self.d_model, self.n_kv_heads * self.key_dim, bias=False)

        if not self.disable_symbols:
            self.wv = Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=self.add_bias_kv)
        self.wo = Linear(self.n_heads * self.head_dim, self.n_heads * self.head_dim, bias=self.add_bias_out)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

    def forward(
        self,
        x: torch.Tensor,
        symbols: Optional[torch.Tensor] = None,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass for HadamardRelationalAttention.

        Shape notation:
            B = batch size, L = sequence length, D = key_dim (= head_dim by default)
            H = n_heads, Hkv = n_kv_heads

        Args:
            x: Input tensor of shape (B, L, d_model)
            symbols: Optional symbol tensor of shape (B, L, d_model)

        Returns:
            output: (B, L, H * D)
            attn_scores: (B, H, L, L) if materialized, else None
            relations: (B, L, L, H, D) if non-identity activation, else None
        """
        bsz, seqlen, _ = x.shape  # x: (B, L, d_model)

        # === Attention Branch ===
        xq_attn = self.wq_attn(x)  # (B, L, H * D)
        xk_attn = self.wk_attn(x)  # (B, L, Hkv * D)
        xq_attn = xq_attn.view(bsz, seqlen, self.n_heads, self.key_dim)     # (B, L, H, D)
        xk_attn = xk_attn.view(bsz, seqlen, self.n_kv_heads, self.key_dim)  # (B, L, Hkv, D)

        if freqs_cos is not None and freqs_sin is not None:
            xq_attn, xk_attn = apply_rotary_emb(xq_attn, xk_attn, freqs_cos, freqs_sin)

        # FlashAttn expects (B, L, H, D), SDPA prefers (B, H, L, D)
        use_flash = (
            HAS_FLASH_ATTN
            and flash_attn_v3 is not None
            and x.dtype in [torch.bfloat16, torch.float16]
            and attn_mask is None
        )

        if self.n_rep_kv != 1:
            xk_attn_rep = repeat_kv(xk_attn, self.n_rep_kv)  # (B, L, H, D)
        else:
            xk_attn_rep = xk_attn  # (B, L, Hkv, D) = (B, L, H, D) when Hkv = H

        # === Relations Branch ===
        if self.identity_rels:
            xq_rel = x.view(bsz, seqlen, self.n_relations, self.key_dim)  # (B, L, H, D)
            xk_rel = x.view(bsz, seqlen, self.n_relations, self.key_dim)  # (B, L, H, D)
        else:
            xq_rel = self.wq_rel(x).view(bsz, seqlen, self.n_relations, self.key_dim)  # (B, L, H, D)
            xk_rel = self.wk_rel(x).view(bsz, seqlen, self.n_kv_heads, self.key_dim)   # (B, L, Hkv, D)

            if self.n_rep_kv != 1:
                 xk_rel = repeat_kv(xk_rel, self.n_rep_kv)  # (B, L, H, D)

        attn_scores = None
        relations = None

        # === Computation ===

        if self.rel_activation == "identity":
            # OPTIMIZED PATH: Use Flash Attention or SDPA (no materialized attention scores)

            # 1. Compute attended relations
            if use_flash:
                # Flash Attention path: expects (B, L, H, D)
                attended_xk_rel = flash_attn_v3(xq_attn, xk_attn, xk_rel, causal=is_causal)  # (B, L, H, D)
                attended_relations = xq_rel * attended_xk_rel       # (B, L, H, D)
                attended_relations = attended_relations.transpose(1, 2)  # (B, H, L, D)
            else:
                # SDPA path: expects (B, H, L, D)
                xq_attn_t = xq_attn.transpose(1, 2)      # (B, H, L, D)
                xk_attn_t = xk_attn_rep.transpose(1, 2)  # (B, H, L, D)
                xk_rel_t = xk_rel.transpose(1, 2)        # (B, H, L, D)

                attended_xk_rel = torch.nn.functional.scaled_dot_product_attention(
                    query=xq_attn_t, key=xk_attn_t, value=xk_rel_t,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=is_causal
                )  # (B, H, L, D)
                attended_relations = xq_rel.transpose(1, 2) * attended_xk_rel  # (B, H, L, D)

            # 2. Compute attended symbols
            if symbols is not None:
                if self.disable_symbols:
                    raise ValueError(f"{self.disable_symbols=} but forward pass called with symbols not None.")

                if self.use_relative_positional_symbols:
                    # Relative positional symbols: must materialize attention scores
                    xq_attn_t = xq_attn.transpose(1, 2)      # (B, H, L, D)
                    xk_attn_t = xk_attn_rep.transpose(1, 2)  # (B, H, L, D)

                    attn_scores = torch.matmul(xq_attn_t, xk_attn_t.transpose(2, 3)) * self.attn_scale  # (B, H, L, L)
                    if is_causal:
                        mask = compute_causal_mask(seqlen, device=xq_attn.device)
                        attn_scores = attn_scores.masked_fill(mask.logical_not(), float("-inf"))
                    if attn_mask is not None:
                         attn_scores = attn_scores.masked_fill(attn_mask.logical_not(), float("-inf"))
                    attn_scores = torch.nn.functional.softmax(attn_scores, dim=-1)  # (B, H, L, L)
                    attn_scores = self.attn_dropout(attn_scores)

                    sv = self.wv(symbols).view(seqlen, seqlen, self.n_kv_heads, self.head_dim)  # (L, L, Hkv, D)
                    attended_symbols = torch.einsum("bhij,ijhd->bihd", attn_scores, sv)  # (B, L, H, D)
                    attended_symbols = attended_symbols.permute(0, 2, 1, 3)  # (B, H, L, D)

                else:
                    # Standard symbols
                    sv = self.wv(symbols).view(bsz, seqlen, self.n_kv_heads, self.head_dim)  # (B, L, Hkv, D)
                    if use_flash:
                        attended_symbols = flash_attn_v3(xq_attn, xk_attn, sv, causal=is_causal)  # (B, L, H, D)
                        attended_symbols = attended_symbols.transpose(1, 2)  # (B, H, L, D)
                    else:
                        if self.n_rep_kv != 1:
                            sv = repeat_kv(sv, self.n_rep_kv)  # (B, L, H, D)
                        sv_t = sv.transpose(1, 2)  # (B, H, L, D)

                        attended_symbols = torch.nn.functional.scaled_dot_product_attention(
                            query=xq_attn.transpose(1, 2),
                            key=xk_attn_rep.transpose(1, 2),
                            value=sv_t,
                            attn_mask=attn_mask,
                            dropout_p=self.dropout if self.training else 0.0,
                            is_causal=is_causal
                        )  # (B, H, L, D)

                output = attended_symbols + attended_relations  # (B, H, L, D)
            else:
                output = attended_relations  # (B, H, L, D)

        else:
            # FALLBACK PATH: Non-identity activation (must materialize attention scores and relations)
            xq_attn_t = xq_attn.transpose(1, 2)      # (B, H, L, D)
            xk_attn_t = xk_attn_rep.transpose(1, 2)  # (B, H, L, D)

            # Compute attention scores
            attn_scores = torch.matmul(xq_attn_t, xk_attn_t.transpose(2, 3)) * self.attn_scale  # (B, H, L, L)
            if is_causal:
                 mask = compute_causal_mask(seqlen, device=xq_attn.device)
                 attn_scores = attn_scores.masked_fill(mask.logical_not(), float("-inf"))
            if attn_mask is not None:
                attn_scores = attn_scores.masked_fill(attn_mask.logical_not(), float("-inf"))
            attn_scores = torch.nn.functional.softmax(attn_scores, dim=-1)  # (B, H, L, L)
            attn_scores = self.attn_dropout(attn_scores)

            # Compute pairwise relations with non-identity activation
            # xq_rel: (B, L, H, D), xk_rel: (B, L, H, D)
            relations = torch.einsum("bihd,bjhd->bijhd", xq_rel, xk_rel)  # (B, L, L, H, D)
            relations = self.rel_activation_(relations)  # (B, L, L, H, D)

            # Attend over relations
            attended_relations = torch.einsum("bhij,bijhd->bihd", attn_scores, relations)  # (B, L, H, D)
            attended_relations = attended_relations.permute(0, 2, 1, 3)  # (B, H, L, D)

            if symbols is not None:
                sv = self.wv(symbols)  # (B, L, Hkv * D)
                if self.use_relative_positional_symbols:
                    sv = sv.view(seqlen, seqlen, self.n_kv_heads, self.head_dim)  # (L, L, Hkv, D)
                    attended_symbols = torch.einsum("bhij,ijhd->bihd", attn_scores, sv)  # (B, L, H, D)
                    attended_symbols = attended_symbols.permute(0, 2, 1, 3)  # (B, H, L, D)
                else:
                    sv = sv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)  # (B, L, Hkv, D)
                    if self.n_rep_kv != 1:
                        sv = repeat_kv(sv, self.n_rep_kv)  # (B, L, H, D)
                    sv_t = sv.transpose(1, 2)  # (B, H, L, D)
                    attended_symbols = torch.matmul(attn_scores, sv_t)  # (B, H, L, D)

                output = attended_symbols + attended_relations  # (B, H, L, D)
            else:
                output = attended_relations  # (B, H, L, D)

        # === Final Projection ===
        # output: (B, H, L, D)
        output = output.transpose(1, 2).contiguous()  # (B, L, H, D)
        output = output.view(bsz, seqlen, self.n_heads * self.head_dim)  # (B, L, H * D)
        output = self.wo(output)  # (B, L, H * D)
        output = self.resid_dropout(output)

        return output, attn_scores, relations


class RelationalCrossAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        n_kv_heads: Optional[int] = None,
        activation: str = "softmax",
        add_bias_kv: bool = False,
        add_bias_out: bool = False,
        total_n_heads: Optional[int] = None,
        use_relative_positional_symbols: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.activation = activation
        self.activation_ = get_activation_function(activation)
        self.dropout = dropout
        self.add_bias_kv = add_bias_kv
        self.add_bias_out = add_bias_out
        self.total_n_heads = n_heads if total_n_heads is None else total_n_heads
        self.use_relative_positional_symbols = use_relative_positional_symbols

        self.n_rep_kv = self.n_heads // self.n_kv_heads
        self.head_dim = self.d_model // self.total_n_heads
        assert self.n_heads % self.n_kv_heads == 0
        assert self.n_rep_kv * self.n_kv_heads == self.n_heads
        assert self.total_n_heads * self.head_dim == self.d_model

        self.attn_scale = 1 / math.sqrt(self.head_dim)

        self.wq = Linear(self.d_model, self.n_heads * self.head_dim, bias=False)
        self.wk = Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=self.add_bias_kv)
        self.wv = Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=self.add_bias_kv)
        self.wo = Linear(self.n_heads * self.head_dim, self.n_heads * self.head_dim, bias=self.add_bias_out)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

    def forward(
        self,
        x: torch.Tensor,
        symbols: torch.Tensor,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seqlen, _ = x.shape

        xq, xk, sv = self.wq(x), self.wk(x), self.wv(symbols)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        if self.use_relative_positional_symbols:
            assert symbols.shape[0] == symbols.shape[1] == seqlen
            sv = sv.view(seqlen, seqlen, self.n_kv_heads, self.head_dim)
        else:
            sv = sv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        if freqs_cos is not None and freqs_sin is not None:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        if self.n_rep_kv != 1:
            xk = repeat_kv(xk, self.n_rep_kv)
            sv = repeat_kv(sv, self.n_rep_kv)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)

        assert not (attn_mask is not None and is_causal)
        if is_causal and attn_mask is None:
            attn_mask = compute_causal_mask(seqlen, device=xq.device)

        scores = torch.matmul(xq, xk.transpose(2, 3)) * self.attn_scale

        if attn_mask is not None and self.activation == "softmax":
            attn_mask_ = torch.zeros(seqlen, seqlen, dtype=xq.dtype, device=xq.device).masked_fill(
                attn_mask.logical_not(), float("-inf")
            )
            scores = scores + attn_mask_

        scores = self.activation_(scores)

        if attn_mask is not None and self.activation != "softmax":
            scores = scores * attn_mask

        scores = self.attn_dropout(scores)

        if not self.use_relative_positional_symbols:
            sv = sv.transpose(1, 2)
            output = torch.matmul(scores, sv)
            output = output.transpose(1, 2)
        else:
            output = torch.einsum("bhij,ijhd->bihd", scores, sv)

        output = output.contiguous().view(bsz, seqlen, -1)
        output = self.wo(output)
        output = self.resid_dropout(output)

        return output, scores


class DisentangledRelationalCrossAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        n_kv_heads: Optional[int] = None,
        rel_activation: str = "identity",
        add_bias_kv: bool = False,
        add_bias_out: bool = False,
        total_n_heads: Optional[int] = None,
        use_relative_positional_symbols: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.rel_activation = rel_activation
        self.rel_activation_ = get_activation_function(rel_activation)
        self.dropout = dropout
        self.add_bias_kv = add_bias_kv
        self.add_bias_out = add_bias_out
        self.total_n_heads = n_heads if total_n_heads is None else total_n_heads
        self.use_relative_positional_symbols = use_relative_positional_symbols

        self.n_rep_kv = self.n_heads // self.n_kv_heads
        self.head_dim = self.d_model // self.total_n_heads
        assert self.n_heads % self.n_kv_heads == 0
        assert self.n_rep_kv * self.n_kv_heads == self.n_heads
        assert self.total_n_heads * self.head_dim == self.d_model

        self.attn_scale = 1 / math.sqrt(self.head_dim)

        self.wq_attn = Linear(self.d_model, self.n_heads * self.head_dim, bias=False)
        self.wk_attn = Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=self.add_bias_kv)

        self.wq_rel = Linear(self.d_model, self.n_heads * self.head_dim, bias=False)
        self.wk_rel = Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=self.add_bias_kv)

        self.wv = Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=self.add_bias_kv)
        self.wo = Linear(self.n_heads * self.head_dim, self.n_heads * self.head_dim, bias=self.add_bias_out)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

    def forward(
        self,
        x: torch.Tensor,
        symbols: torch.Tensor,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seqlen, _ = x.shape

        xq_attn, xk_attn = self.wq_attn(x), self.wk_attn(x)
        xq_attn = xq_attn.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk_attn = xk_attn.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq_rel, xk_rel = self.wq_rel(x), self.wk_rel(x)
        xq_rel = xq_rel.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk_rel = xk_rel.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        sv = self.wv(symbols)

        if self.use_relative_positional_symbols:
            assert symbols.shape[0] == symbols.shape[1] == seqlen
            sv = sv.view(seqlen, seqlen, self.n_kv_heads, self.head_dim)
        else:
            sv = sv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        if freqs_cos is not None and freqs_sin is not None:
            xq_attn, xk_attn = apply_rotary_emb(xq_attn, xk_attn, freqs_cos, freqs_sin)

        if self.n_rep_kv != 1:
            xk_attn = repeat_kv(xk_attn, self.n_rep_kv)
            sv = repeat_kv(sv, self.n_rep_kv)

        xq_attn = xq_attn.transpose(1, 2)
        xk_attn = xk_attn.transpose(1, 2)
        xq_rel = xq_rel.transpose(1, 2)
        xk_rel = xk_rel.transpose(1, 2)

        assert not (attn_mask is not None and is_causal)
        if is_causal and attn_mask is None:
            attn_mask = compute_causal_mask(seqlen, device=xq_attn.device)

        attn_scores = torch.matmul(xq_attn, xk_attn.transpose(2, 3)) * self.attn_scale

        if attn_mask is not None:
            attn_mask_ = torch.zeros(seqlen, seqlen, dtype=xq_attn.dtype, device=xq_attn.device).masked_fill(
                attn_mask.logical_not(), float("-inf")
            )
            attn_scores = attn_scores + attn_mask_

        attn_scores = nn.functional.softmax(attn_scores, dim=-1)

        rel_scores = torch.matmul(xq_rel, xk_rel.transpose(2, 3)) * self.attn_scale
        rel_scores = self.rel_activation_(rel_scores)

        rca_scores = attn_scores * rel_scores
        rca_scores = self.attn_dropout(rca_scores)

        if not self.use_relative_positional_symbols:
            sv = sv.transpose(1, 2)
            output = torch.matmul(rca_scores, sv)
            output = output.transpose(1, 2)
        else:
            output = torch.einsum("bhij,ijhd->bihd", rca_scores, sv)

        output = output.contiguous().view(bsz, seqlen, -1)
        output = self.wo(output)
        output = self.resid_dropout(output)

        return output, attn_scores, rel_scores


def create_relational_attention(
    ra_type: str,
    d_model: int,
    n_heads: int,
    total_n_heads: int,
    dropout: float,
    ra_kwargs: dict,
    symbol_retriever: Optional[nn.Module] = None,
) -> nn.Module:
    kwargs = ra_kwargs.copy()
    if ra_type == "relational_attention":
        if isinstance(symbol_retriever, PositionRelativeSymbolRetriever):
            kwargs.pop("use_relative_positional_symbols", None)
            return RelationalAttentionWithPositionRelativeSymbols(
                d_model=d_model,
                n_heads=n_heads,
                max_rel_pos=symbol_retriever.max_rel_pos,
                total_n_heads=total_n_heads,
                dropout=dropout,
                **kwargs,
            )
        return RelationalAttention(
            d_model=d_model,
            n_heads=n_heads,
            total_n_heads=total_n_heads,
            dropout=dropout,
            **kwargs,
        )
    elif ra_type == "hadamard_relational_attention":
        return HadamardRelationalAttention(
            d_model=d_model,
            n_heads=n_heads,
            total_n_heads=total_n_heads,
            dropout=dropout,
            **kwargs,
        )
    elif ra_type == "rca":
        return RelationalCrossAttention(
            d_model=d_model,
            n_heads=n_heads,
            total_n_heads=total_n_heads,
            dropout=dropout,
            **kwargs,
        )
    elif ra_type == "disrca":
        return DisentangledRelationalCrossAttention(
            d_model=d_model,
            n_heads=n_heads,
            total_n_heads=total_n_heads,
            dropout=dropout,
            **kwargs,
        )
    else:
        raise ValueError(f"Invalid relational attention type: {ra_type}")


class DualAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads_sa: int,
        n_heads_ra: int,
        dropout: float = 0.,
        sa_kwargs: Optional[dict] = None,
        ra_kwargs: Optional[dict] = None,
        share_attn_params: bool = False,
        ra_type: str = "relational_attention",
        causal: bool = False,
        symbol_retriever: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads_sa = n_heads_sa
        self.n_heads_ra = n_heads_ra
        self.dropout = dropout
        self.sa_kwargs = sa_kwargs if sa_kwargs is not None else {}
        self.ra_kwargs = ra_kwargs if ra_kwargs is not None else {}
        self.ra_type = ra_type
        self.share_attn_params = share_attn_params
        self.causal = causal
        self.symbol_retriever = symbol_retriever
        if symbol_retriever is None:
            self.symbol_retriever = NullSymbolRetriever()
        self.use_relative_positional_symbols = isinstance(self.symbol_retriever, PositionRelativeSymbolRetriever)
        self.ra_kwargs["use_relative_positional_symbols"] = self.use_relative_positional_symbols
        if isinstance(self.symbol_retriever, NullSymbolRetriever):
            self.ra_kwargs['disable_symbols'] = True

        if self.share_attn_params and n_heads_sa != n_heads_ra:
            raise ValueError("Number of heads in self-attention and relational attention must be the same if sharing attention parameters")

        self.use_self_attn = n_heads_sa > 0
        self.use_rel_attn = n_heads_ra > 0
        self.total_n_heads = n_heads_sa + n_heads_ra

        if not (self.use_self_attn or self.use_rel_attn):
            raise ValueError("At least one of self-attention or relational attention must be used")

        if self.use_self_attn:
            self.self_attention = Attention(
                d_model=d_model,
                n_heads=n_heads_sa,
                total_n_heads=self.total_n_heads,
                dropout=dropout,
                **self.sa_kwargs,
            )

        if self.use_rel_attn:
            self.relational_attention = create_relational_attention(
                ra_type=ra_type,
                d_model=d_model,
                n_heads=n_heads_ra,
                total_n_heads=self.total_n_heads,
                dropout=dropout,
                ra_kwargs=self.ra_kwargs,
                symbol_retriever=symbol_retriever,
            )

        if self.share_attn_params and self.use_self_attn and self.use_rel_attn:
            self.self_attention.wq = self.relational_attention.wq_attn
            self.self_attention.wk = self.relational_attention.wk_attn

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: Optional[bool] = None,
        cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if is_causal is None and attn_mask is None:
            is_causal = self.causal
            assert self.causal is not None, "Failed to set causal flag in DualAttention"

        freqs_cos, freqs_sin = cos_sin if cos_sin is not None else (None, None)

        symbols = None
        if self.use_rel_attn:
            if self.symbol_retriever is None:
                raise ValueError("symbol_retriever must be provided when using relational attention.")
            symbols = self.symbol_retriever(x)

        if self.use_self_attn:
            self_attn_out, self_attn_scores = self.self_attention(
                query=x,
                key=x,
                value=x,
                freqs_cos=freqs_cos,
                freqs_sin=freqs_sin,
                attn_mask=attn_mask,
                is_causal=is_causal,
                need_weights=need_weights,
            )

        if self.use_rel_attn:
            if isinstance(self.relational_attention, RelationalAttentionWithPositionRelativeSymbols):
                rel_attn_out, *rel_attn_scores = self.relational_attention(
                    x,
                    freqs_cos=freqs_cos,
                    freqs_sin=freqs_sin,
                    attn_mask=attn_mask,
                    is_causal=is_causal,
                )
            else:
                rel_attn_out, *rel_attn_scores = self.relational_attention(
                    x,
                    symbols,
                    attn_mask=attn_mask,
                    is_causal=is_causal,
                    freqs_cos=freqs_cos,
                    freqs_sin=freqs_sin,
                )

        if self.use_rel_attn and self.use_self_attn:
            out = torch.concat((self_attn_out, rel_attn_out), dim=-1)
        elif self.use_rel_attn:
            out = rel_attn_out
            self_attn_scores = None
        else:
            out = self_attn_out
            rel_attn_scores = None

        return out, self_attn_scores, rel_attn_scores
    
def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0):
    # NOTE: PyTorch nn.init.trunc_normal_ is not mathematically correct, the std dev is not actually the std dev of initialized tensor
    # This function is a PyTorch version of jax truncated normal init (default init method in flax)
    # https://github.com/jax-ml/jax/blob/main/jax/_src/random.py#L807-L848
    # https://github.com/jax-ml/jax/blob/main/jax/_src/nn/initializers.py#L162-L199

    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2

            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * lower ** 2)
            pdf_l = c * math.exp(-0.5 * upper ** 2)
            comp_std = std / math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)

            tensor.uniform_(a, b)
            tensor.erfinv_()
            tensor.mul_(sqrt2 * comp_std)
            tensor.clip_(lower * comp_std, upper * comp_std)

    return tensor
