import importlib
import os
from glob import glob

import pytest
import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from nanovllm.models.deepseek_v2 import DeepseekV2Attention


MODEL_PATH = os.environ.get("DEEPSEEK_V2_MODEL")
pytestmark = pytest.mark.skipif(
    not MODEL_PATH or not torch.cuda.is_available(),
    reason="set DEEPSEEK_V2_MODEL to a local DeepSeek-V2-Lite checkpoint on a CUDA host",
)


def _load_attention_layer(path: str, layer_idx: int = 0) -> dict[str, torch.Tensor]:
    prefix = f"model.layers.{layer_idx}.self_attn."
    state = {}
    for filename in sorted(glob(os.path.join(path, "*.safetensors"))):
        with safe_open(filename, framework="pt", device="cpu") as checkpoint:
            for name in checkpoint.keys():
                if name.startswith(prefix):
                    state[name.removeprefix(prefix)] = checkpoint.get_tensor(name)
    if not state:
        raise AssertionError(f"no attention weights found for {prefix}")
    return state


def test_layer_zero_mla_projection_and_yarn_match_official(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    reference_cls = get_class_from_dynamic_module(
        "modeling_deepseek.DeepseekV2Attention", MODEL_PATH
    )
    reference_module = importlib.import_module(reference_cls.__module__)
    reference = reference_cls(config, layer_idx=0).cuda().eval()
    actual = DeepseekV2Attention(config).cuda().eval()

    state = _load_attention_layer(MODEL_PATH)
    reference.load_state_dict(state, strict=True)
    actual.load_state_dict(state, strict=True)

    torch.manual_seed(0)
    num_tokens = 7
    model_dtype = getattr(config, "dtype", None) or config.torch_dtype
    hidden_states = torch.randn(
        1,
        num_tokens,
        config.hidden_size,
        device="cuda",
        dtype=model_dtype,
    )

    with torch.inference_mode():
        q = reference.q_proj(hidden_states).view(
            1, num_tokens, config.num_attention_heads, reference.q_head_dim
        ).transpose(1, 2)
        ref_q_nope, ref_q_pe = q.split(
            [config.qk_nope_head_dim, config.qk_rope_head_dim], dim=-1
        )
        kv_a = reference.kv_a_proj_with_mqa(hidden_states)
        ref_compressed, ref_raw_k_pe = kv_a.split(
            [config.kv_lora_rank, config.qk_rope_head_dim], dim=-1
        )
        ref_compressed = reference.kv_a_layernorm(ref_compressed)
        ref_kv = reference.kv_b_proj(ref_compressed).view(
            1,
            num_tokens,
            config.num_attention_heads,
            config.qk_nope_head_dim + config.v_head_dim,
        ).transpose(1, 2)
        ref_k_nope, ref_value = ref_kv.split(
            [config.qk_nope_head_dim, config.v_head_dim], dim=-1
        )
        ref_raw_k_pe = ref_raw_k_pe.view(
            1, num_tokens, 1, config.qk_rope_head_dim
        ).transpose(1, 2)

        projected = actual.project(hidden_states.view(num_tokens, -1))
        q_nope, q_pe, compressed, raw_k_pe, k_nope, value = projected

        torch.testing.assert_close(q_nope, ref_q_nope.squeeze(0).transpose(0, 1))
        torch.testing.assert_close(q_pe, ref_q_pe.squeeze(0).transpose(0, 1))
        torch.testing.assert_close(compressed, ref_compressed.squeeze(0))
        torch.testing.assert_close(
            raw_k_pe, ref_raw_k_pe.squeeze(0).transpose(0, 1)
        )
        torch.testing.assert_close(k_nope, ref_k_nope.squeeze(0).transpose(0, 1))
        torch.testing.assert_close(value, ref_value.squeeze(0).transpose(0, 1))

        positions = torch.arange(num_tokens, device="cuda").unsqueeze(0)
        cos, sin = reference.rotary_emb(ref_value, seq_len=num_tokens)
        ref_q_pe, ref_k_pe = reference_module.apply_rotary_pos_emb(
            ref_q_pe, ref_raw_k_pe, cos, sin, positions
        )
        q_pe, k_pe = actual.rotary_emb(positions.squeeze(0), q_pe, raw_k_pe)
        torch.testing.assert_close(
            q_pe, ref_q_pe.squeeze(0).transpose(0, 1), rtol=2e-3, atol=2e-3
        )
        torch.testing.assert_close(
            k_pe, ref_k_pe.squeeze(0).transpose(0, 1), rtol=2e-3, atol=2e-3
        )
