from types import SimpleNamespace

import torch

from nanovllm.layers.mla_attention import store_mla_cache
from nanovllm.models.deepseek_v2 import DeepseekV2Attention, DeepseekV2ForCausalLM


def _tiny_config():
    return SimpleNamespace(
        hidden_size=32,
        num_attention_heads=4,
        q_lora_rank=None,
        kv_lora_rank=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        attention_bias=False,
        rms_norm_eps=1e-6,
        max_position_embeddings=32,
        rope_theta=10000,
        rope_scaling={
            "type": "yarn",
            "factor": 2,
            "original_max_position_embeddings": 16,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 0.707,
            "mscale_all_dim": 0.707,
        },
        vocab_size=64,
        num_hidden_layers=2,
        intermediate_size=48,
        hidden_act="silu",
        n_routed_experts=3,
        num_experts_per_tok=2,
        routed_scaling_factor=1.0,
        norm_topk_prob=False,
        scoring_func="softmax",
        topk_method="greedy",
        moe_intermediate_size=12,
        n_shared_experts=2,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        tie_word_embeddings=False,
    )


def test_deepseek_projection_shapes(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    attention = DeepseekV2Attention(_tiny_config())

    tensors = attention.project(torch.randn(3, 32))
    assert [tensor.shape for tensor in tensors] == [
        (3, 4, 4),
        (3, 4, 4),
        (3, 8),
        (3, 1, 4),
        (3, 4, 4),
        (3, 4, 4),
    ]

    _, q_pe, _, raw_k_pe, _, _ = tensors
    q_pe, k_pe = attention.rotary_emb(torch.arange(3), q_pe, raw_k_pe)
    assert q_pe.shape == (3, 4, 4)
    assert k_pe.shape == (3, 1, 4)


def test_store_mla_cache_uses_slot_mapping():
    cache = torch.zeros(2, 4, 6)
    values = torch.arange(18, dtype=torch.float32).view(3, 6)
    slots = torch.tensor([1, 6, -1], dtype=torch.int32)

    store_mla_cache(values, cache, slots)

    flat_cache = cache.view(-1, 6)
    torch.testing.assert_close(flat_cache[1], values[0])
    torch.testing.assert_close(flat_cache[6], values[1])
    assert torch.count_nonzero(flat_cache[0]) == 0


def test_checkpoint_parameter_names_follow_official_layout(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    model = DeepseekV2ForCausalLM(_tiny_config())
    names = {name for name, _ in model.named_parameters()}

    assert {
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
        "model.layers.0.self_attn.kv_a_layernorm.weight",
        "model.layers.0.self_attn.kv_b_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.1.mlp.gate.weight",
        "model.layers.1.mlp.experts.0.gate_proj.weight",
        "model.layers.1.mlp.experts.0.up_proj.weight",
        "model.layers.1.mlp.experts.0.down_proj.weight",
        "model.layers.1.mlp.shared_experts.gate_proj.weight",
        "model.norm.weight",
        "lm_head.weight",
    } <= names
