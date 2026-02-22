# Hadamard Dual-Attention Implementation Notes

This note documents the current production implementation of dual attention and related integration points.

## Scope and Goals
- Keep the production surface minimal and aligned with `nanochat/gpt.py` style.
- Support optional dual attention via config while preserving default/backward-compatible standard GPT behavior.
- Keep only one relational variant in production (`hadamard_dual`) for v1.
- Preserve the full pre-refactor implementation as reference only.

## File Layout
- `nanochat/abstractor.py`: production implementation (minimal API).
- `nanochat/abstractor_reference.py`: legacy implementation snapshot (not used by production path).
- `nanochat/attention_utils.py`: shared helpers for RoPE application and Q/K normalization.

## Production API (`nanochat/abstractor.py`)
Exports are intentionally limited to:
- `NullSymbolRetriever`
- `HadamardDualAttention`
- `DualAttention`

`DualAttention` is the GPT-facing entry point and currently wraps `HadamardDualAttention`.

## Core Design Choices
- Single fused attention score path:
  - Q/K are shared across both branches.
  - SA values (`v`) and RA values (`xk_rel`) are concatenated and passed through one flash-attention call.
  - Outputs are split back into SA/RA streams.
- Hadamard RA branch:
  - RA output uses `q_rel * y_rel` (elementwise).
  - Only hadamard relational variant is supported in production v1.
- GQA support retained:
  - `n_kv_head <= n_head` and `n_head % n_kv_head == 0` are required.
- Head partitioning:
  - `dual_rel_head_proportion` controls RA/SA split.
  - `n_heads_ra = clamp(round(n_head * proportion), 0, n_head)`.
  - Endpoints `0.0` (all SA) and `1.0` (all RA) are valid.

## Residual Augmentation Policy
A single policy is shared between standard and dual modes:
- Config fields:
  - `use_residual_augmentation`
  - `residual_stride`
  - `residual_gate_channels`
- Scheduling rule:
  - If enabled, apply on stride-aligned layers.
  - Final transformer layer is always included.
- In dual mode:
  - Shared residual source (`ve`, value embedding table path).
  - Separate learned gates per branch (SA and RA).

## GPT Integration
- `GPTConfig` adds:
  - `attention_impl` (`standard` or `hadamard_dual`)
  - `dual_rel_head_proportion`
  - residual augmentation fields above
- `Block` attention selection:
  - `standard` -> `CausalSelfAttention`
  - `hadamard_dual` -> `DualAttention`
- Defaults preserve legacy behavior (`attention_impl="standard"`).

## KV Cache Status (Important)
- Dual mode KV cache is intentionally not implemented in v1.
- Behavior is fail-fast:
  - `GPT.forward(...)` raises `NotImplementedError` when `attention_impl="hadamard_dual"` and `kv_cache` is provided.
- Implication:
  - Teacher-forced training forward is supported.
  - Inference paths that depend on KV cache must not use dual mode yet.

## Serialization and Pipeline Compatibility
- `scripts/base_train.py` includes CLI flags for all new attention/residual settings.
- `scripts/chat_sft.py` saves full `model.config.__dict__`.
- `scripts/chat_rl.py` already saved full config.
- `nanochat/checkpoint_manager.py` patches missing new config keys for old checkpoints.

## Tests Added
See `tests/test_dual_attention.py` for:
- Config patch/default checks.
- Head split behavior (`0.0`, `0.5`, `1.0`).
- Dual forward shape checks.
- GQA shape/compatibility checks in dual mode.
- Residual scheduling checks.
- KV behavior checks (standard accepts cache, dual rejects).

## Known Limitations
- Only hadamard dual attention is available in production.
- Symbol retriever functionality is not part of v1 production behavior (`NullSymbolRetriever` placeholder).
- No structured debug/intermediate-state return API for attention internals.
- No dual-mode KV-cache support.

## TODOs / Future Work
1. Implement KV-cache support for `HadamardDualAttention` and integrate with decode/inference paths.
2. Add optional debug/intermediate cache API at GPT level for interpretability analysis.
3. Revisit symbol retriever integration once production requirements are clear.
4. Expand test coverage for longer-window/sliding-window decode behavior once dual KV cache exists.
5. Consider a performance pass (kernel-level and memory-layout profiling) after functional parity goals are complete.

## Handoff Notes
- If changing attention math, keep `nanochat/attention_utils.py` as the single helper source for RoPE/QK normalization to avoid drift between implementations.
- Keep `attention_impl="standard"` as default unless dual mode is fully feature-complete for all runtime paths.
- Preserve backward compatibility by updating checkpoint config patching whenever new config keys are introduced.
