import torch
import pytest

from nanochat.abstractor import HadamardDualAttention
from nanochat.checkpoint_manager import _patch_missing_config_keys
from nanochat.engine import KVCache
from nanochat.gpt import GPT, GPTConfig, has_residual_layer


def test_config_defaults_and_patch_missing_keys():
    cfg = {
        "sequence_len": 64,
        "vocab_size": 256,
        "n_layer": 2,
        "n_head": 2,
        "n_kv_head": 2,
        "n_embd": 32,
        "window_pattern": "L",
    }
    _patch_missing_config_keys(cfg)

    assert cfg["attention_impl"] == "standard"
    assert cfg["dual_rel_head_proportion"] == 0.5
    assert cfg["use_residual_augmentation"] is True
    assert cfg["residual_stride"] == 2
    assert cfg["residual_gate_channels"] == 32

    default_cfg = GPTConfig()
    assert default_cfg.attention_impl == "standard"


def test_hadamard_head_split_from_proportion():
    cfg = GPTConfig(
        sequence_len=32,
        vocab_size=128,
        n_layer=2,
        n_head=6,
        n_kv_head=3,
        n_embd=96,
        attention_impl="hadamard_dual",
        dual_rel_head_proportion=0.0,
    )
    attn = HadamardDualAttention(cfg, layer_idx=0)
    assert attn.n_heads_ra == 0
    assert attn.n_heads_sa == 6

    cfg.dual_rel_head_proportion = 0.5
    attn = HadamardDualAttention(cfg, layer_idx=0)
    assert attn.n_heads_ra == 3
    assert attn.n_heads_sa == 3

    cfg.dual_rel_head_proportion = 1.0
    attn = HadamardDualAttention(cfg, layer_idx=0)
    assert attn.n_heads_ra == 6
    assert attn.n_heads_sa == 0


def test_dual_forward_shape_and_gqa():
    cfg = GPTConfig(
        sequence_len=16,
        vocab_size=64,
        n_layer=2,
        n_head=6,
        n_kv_head=3,
        n_embd=96,
        window_pattern="L",
        attention_impl="hadamard_dual",
        dual_rel_head_proportion=0.5,
        use_residual_augmentation=True,
        residual_stride=1,
        residual_gate_channels=16,
    )
    model = GPT(cfg)
    model.init_weights()

    idx = torch.randint(0, cfg.vocab_size, (2, 8), dtype=torch.long)
    logits = model.forward(idx)
    assert logits.shape == (2, 8, cfg.vocab_size)


def test_residual_schedule_stride_and_last_layer():
    n_layer = 6
    # stride=3, final layer index is 5 => parity class 2 plus forced final layer.
    enabled = [
        has_residual_layer(i, n_layer, use_residual_augmentation=True, residual_stride=3)
        for i in range(n_layer)
    ]
    assert enabled == [False, False, True, False, False, True]

    # final layer is always enabled when augmentation is enabled.
    assert has_residual_layer(5, n_layer, True, 4)
    assert not has_residual_layer(0, n_layer, False, 2)


def test_kv_cache_behavior_standard_vs_hadamard_dual():
    std_cfg = GPTConfig(
        sequence_len=16,
        vocab_size=64,
        n_layer=1,
        n_head=2,
        n_kv_head=1,
        n_embd=32,
        attention_impl="standard",
    )
    std_model = GPT(std_cfg)
    std_model.init_weights()

    kv_cache = KVCache(
        batch_size=1,
        num_heads=std_cfg.n_kv_head,
        seq_len=8,
        head_dim=std_cfg.n_embd // std_cfg.n_head,
        num_layers=std_cfg.n_layer,
        device=std_model.get_device(),
        dtype=torch.float32,
    )
    idx = torch.randint(0, std_cfg.vocab_size, (1, 1), dtype=torch.long)
    logits = std_model.forward(idx, kv_cache=kv_cache)
    assert logits.shape == (1, 1, std_cfg.vocab_size)

    dual_cfg = GPTConfig(
        sequence_len=16,
        vocab_size=64,
        n_layer=1,
        n_head=2,
        n_kv_head=1,
        n_embd=32,
        attention_impl="hadamard_dual",
    )
    dual_model = GPT(dual_cfg)
    dual_model.init_weights()

    with pytest.raises(NotImplementedError):
        dual_model.forward(idx, kv_cache=object())
