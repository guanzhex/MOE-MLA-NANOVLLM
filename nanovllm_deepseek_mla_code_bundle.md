# nano-vLLM → DeepSeek-style MLA 改造代码汇总

> 说明：这是我们前面讨论得到的第一版 correctness skeleton，供新 Work 中按真实 nano-vLLM 仓库接口逐步落地。  
> 目标：单 GPU、单请求，跑通 DeepSeek-V2-Lite checkpoint 的 `prefill → MLA cache → naive decode → token generation`。  
> 注意：这些代码需要根据当前 nano-vLLM 的真实 `Linear / RMSNorm / Attention / ModelRunner / cache` 接口做适配，不保证直接复制后一次编译通过。

---

## 1. `rotary_embedding.py`：DeepSeek YaRN RoPE

```python
import math
import torch
from torch import nn


def yarn_find_correction_dim(
    num_rotations: float,
    dim: int,
    base: float = 10000,
    max_position_embeddings: int = 2048,
):
    return (
        dim
        * math.log(
            max_position_embeddings
            / (num_rotations * 2 * math.pi)
        )
        / (2 * math.log(base))
    )


def yarn_find_correction_range(
    low_rot: float,
    high_rot: float,
    dim: int,
    base: float = 10000,
    max_position_embeddings: int = 2048,
):
    low = math.floor(
        yarn_find_correction_dim(
            low_rot,
            dim,
            base,
            max_position_embeddings,
        )
    )
    high = math.ceil(
        yarn_find_correction_dim(
            high_rot,
            dim,
            base,
            max_position_embeddings,
        )
    )
    return max(low, 0), min(high, dim - 1)


def yarn_get_mscale(
    scale: float = 1.0,
    mscale: float = 1.0,
):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_linear_ramp_mask(
    min_val: float,
    max_val: float,
    dim: int,
):
    if min_val == max_val:
        max_val += 0.001

    linear_func = (
        torch.arange(dim, dtype=torch.float32)
        - min_val
    ) / (max_val - min_val)

    return torch.clamp(linear_func, 0, 1)


class DeepseekV2YarnRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int,
        base: float,
        scaling_factor: float,
        original_max_position_embeddings: int = 4096,
        beta_fast: float = 32,
        beta_slow: float = 1,
        mscale: float = 1,
        mscale_all_dim: float = 0,
    ):
        super().__init__()

        self.dim = dim

        freq_extra = 1.0 / (
            base ** (
                torch.arange(0, dim, 2, dtype=torch.float32) / dim
            )
        )

        freq_inter = 1.0 / (
            scaling_factor
            * base ** (
                torch.arange(0, dim, 2, dtype=torch.float32) / dim
            )
        )

        low, high = yarn_find_correction_range(
            beta_fast,
            beta_slow,
            dim,
            base,
            original_max_position_embeddings,
        )

        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(
            low,
            high,
            dim // 2,
        )

        inv_freq = (
            freq_inter * (1 - inv_freq_mask)
            + freq_extra * inv_freq_mask
        )

        positions = torch.arange(
            max_position_embeddings,
            dtype=torch.float32,
        )

        freqs = torch.einsum(
            "i,j->ij",
            positions,
            inv_freq,
        )

        yarn_mscale = (
            yarn_get_mscale(scaling_factor, mscale)
            / yarn_get_mscale(
                scaling_factor,
                mscale_all_dim,
            )
        )

        cos = freqs.cos() * yarn_mscale
        sin = freqs.sin() * yarn_mscale

        cache = torch.cat(
            (cos, sin),
            dim=-1,
        ).unsqueeze(1)

        self.register_buffer(
            "cos_sin_cache",
            cache,
            persistent=False,
        )

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ):
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)

        query = apply_rotary_emb(
            query,
            cos,
            sin,
        )
        key = apply_rotary_emb(
            key,
            cos,
            sin,
        )

        return query, key
```

> 这里默认复用 nano-vLLM 原来的 `apply_rotary_emb()`。

---

## 2. `deepseek_v2.py`：`DeepseekV2Attention.__init__`

```python
import torch
from torch import nn
import torch.distributed as dist

from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from nanovllm.layers.rotary_embedding import (
    DeepseekV2YarnRotaryEmbedding,
)


class DeepseekV2Attention(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads

        tp_size = dist.get_world_size()
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size

        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim

        self.q_head_dim = (
            self.qk_nope_head_dim
            + self.qk_rope_head_dim
        )

        if self.q_lora_rank is None:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.total_num_heads * self.q_head_dim,
                bias=False,
            )
        else:
            raise NotImplementedError(
                "暂时只支持 q_lora_rank=None"
            )

        self.kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
        )

        self.kv_a_layernorm = RMSNorm(
            self.kv_lora_rank,
            eps=config.rms_norm_eps,
        )

        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.total_num_heads * (
                self.qk_nope_head_dim
                + self.v_head_dim
            ),
            bias=False,
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
        )

        rope_scaling = config.rope_scaling

        self.rotary_emb = DeepseekV2YarnRotaryEmbedding(
            dim=self.qk_rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            scaling_factor=rope_scaling["factor"],
            original_max_position_embeddings=(
                rope_scaling["original_max_position_embeddings"]
            ),
            beta_fast=rope_scaling["beta_fast"],
            beta_slow=rope_scaling["beta_slow"],
            mscale=rope_scaling["mscale"],
            mscale_all_dim=rope_scaling["mscale_all_dim"],
        )
```

---

## 3. Projection-only 调试函数

```python
def forward_projection_only(
    self,
    hidden_states,
):
    q = self.q_proj(hidden_states)

    q = q.view(
        -1,
        self.num_heads,
        self.q_head_dim,
    )

    q_nope, q_pe = q.split(
        [
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
        ],
        dim=-1,
    )

    compressed_kv_and_kpe = (
        self.kv_a_proj_with_mqa(
            hidden_states
        )
    )

    compressed_kv, k_pe = (
        compressed_kv_and_kpe.split(
            [
                self.kv_lora_rank,
                self.qk_rope_head_dim,
            ],
            dim=-1,
        )
    )

    k_pe = k_pe.unsqueeze(1)

    compressed_kv = self.kv_a_layernorm(
        compressed_kv
    )

    kv = self.kv_b_proj(
        compressed_kv
    )

    kv = kv.view(
        -1,
        self.num_heads,
        self.qk_nope_head_dim
        + self.v_head_dim,
    )

    k_nope, v = kv.split(
        [
            self.qk_nope_head_dim,
            self.v_head_dim,
        ],
        dim=-1,
    )

    return (
        q_nope,
        q_pe,
        compressed_kv,
        k_pe,
        k_nope,
        v,
    )
```

期望 shape：

```text
q_nope        [N, H, d_n]
q_pe          [N, H, d_r]
compressed_kv [N, kv_lora_rank]
k_pe          [N, 1, d_r]
k_nope        [N, H, d_n]
v             [N, H, d_v]
```

---

## 4. 正式 `DeepseekV2Attention.forward`

```python
def forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
):
    q = self.q_proj(hidden_states)

    q = q.view(
        -1,
        self.num_heads,
        self.q_head_dim,
    )

    q_nope, q_pe = q.split(
        [
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
        ],
        dim=-1,
    )

    kv_a = self.kv_a_proj_with_mqa(
        hidden_states
    )

    compressed_kv, k_pe = kv_a.split(
        [
            self.kv_lora_rank,
            self.qk_rope_head_dim,
        ],
        dim=-1,
    )

    compressed_kv_norm = (
        self.kv_a_layernorm(
            compressed_kv
        )
    )

    kv = self.kv_b_proj(
        compressed_kv_norm
    )

    kv = kv.view(
        -1,
        self.num_heads,
        self.qk_nope_head_dim
        + self.v_head_dim,
    )

    k_nope, v = kv.split(
        [
            self.qk_nope_head_dim,
            self.v_head_dim,
        ],
        dim=-1,
    )

    k_pe = k_pe.unsqueeze(1)

    q_pe, k_pe_rope = self.rotary_emb(
        positions,
        q_pe,
        k_pe,
    )

    q = torch.cat(
        [
            q_nope,
            q_pe,
        ],
        dim=-1,
    )

    k = torch.cat(
        [
            k_nope,
            k_pe_rope.expand(
                -1,
                self.num_heads,
                -1,
            ),
        ],
        dim=-1,
    )

    o = self.attn(
        q,
        k,
        v,
        compressed_kv_norm,
        k_pe,
        positions,
    )

    output = self.o_proj(
        o.flatten(1)
    )

    return output
```

---

## 5. `MLAAttention`：Prefill + Naive Decode

```python
class MLAAttention(nn.Module):
    def __init__(
        self,
        num_heads,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        kv_lora_rank,
        scale,
        kv_b_proj,
        rotary_emb,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.scale = scale

        self.kv_b_proj = kv_b_proj
        self.rotary_emb = rotary_emb

        self.mla_cache = torch.tensor([])

    def forward(
        self,
        q,
        k,
        v,
        compressed_kv,
        k_pe,
        positions,
    ):
        context = get_context()

        if self.mla_cache.numel():
            cache_value = torch.cat(
                [
                    compressed_kv,
                    k_pe.squeeze(1),
                ],
                dim=-1,
            )

            store_mla_cache(
                cache_value,
                self.mla_cache,
                context.slot_mapping,
            )

        if context.is_prefill:
            o = flash_attn_varlen_func(
                q,
                k,
                v,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,
            )

            return o

        block_tables = context.block_tables
        context_lens = context.context_lens

        assert context_lens.numel() == 1

        seq_len = int(
            context_lens[0].item()
        )

        block_size = self.mla_cache.shape[1]
        block_table = block_tables[0]

        token_slots = []

        for token_idx in range(seq_len):
            logical_block = (
                token_idx // block_size
            )

            offset = (
                token_idx % block_size
            )

            physical_block = int(
                block_table[
                    logical_block
                ].item()
            )

            token_slots.append(
                physical_block
                * block_size
                + offset
            )

        flat_cache = self.mla_cache.view(
            -1,
            self.kv_lora_rank
            + self.qk_rope_head_dim,
        )

        token_slots = torch.tensor(
            token_slots,
            device=flat_cache.device,
            dtype=torch.long,
        )

        history = flat_cache[
            token_slots
        ]

        history_compressed_kv = history[
            :,
            :self.kv_lora_rank,
        ]

        history_k_pe = history[
            :,
            self.kv_lora_rank:,
        ].unsqueeze(1)

        history_kv = self.kv_b_proj(
            history_compressed_kv
        )

        history_kv = history_kv.view(
            seq_len,
            self.num_heads,
            self.qk_nope_head_dim
            + self.v_head_dim,
        )

        (
            history_k_nope,
            history_v,
        ) = history_kv.split(
            [
                self.qk_nope_head_dim,
                self.v_head_dim,
            ],
            dim=-1,
        )

        history_positions = torch.arange(
            seq_len,
            device=q.device,
            dtype=positions.dtype,
        )

        dummy_q = history_k_pe.expand(
            -1,
            self.num_heads,
            -1,
        )

        _, history_k_pe = self.rotary_emb(
            history_positions,
            dummy_q,
            history_k_pe,
        )

        history_k_pe = (
            history_k_pe.expand(
                -1,
                self.num_heads,
                -1,
            )
        )

        history_k = torch.cat(
            [
                history_k_nope,
                history_k_pe,
            ],
            dim=-1,
        )

        cu_seqlens_q = torch.tensor(
            [0, 1],
            dtype=torch.int32,
            device=q.device,
        )

        cu_seqlens_k = torch.tensor(
            [0, seq_len],
            dtype=torch.int32,
            device=q.device,
        )

        o = flash_attn_varlen_func(
            q,
            history_k,
            history_v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=1,
            max_seqlen_k=seq_len,
            softmax_scale=self.scale,
            causal=True,
        )

        return o
```

---

## 6. MLA Cache Allocation

```python
mla_cache_dim = (
    hf_config.kv_lora_rank
    + hf_config.qk_rope_head_dim
)

block_bytes = (
    num_layers
    * block_size
    * mla_cache_dim
    * dtype.itemsize
)

mla_cache = torch.empty(
    num_layers,
    num_blocks,
    block_size,
    mla_cache_dim,
    dtype=dtype,
    device="cuda",
)
```

每层挂载：

```python
layer.self_attn.attn.mla_cache = (
    mla_cache[layer_idx]
)
```

推荐第一版 cache 语义：

```text
[compressed_kv_norm | raw_k_pe]
```

---

## 7. `store_mla_cache`：Python 版

```python
def store_mla_cache(
    cache_value,
    mla_cache,
    slot_mapping,
):
    cache_dim = cache_value.shape[-1]

    flat_cache = mla_cache.view(
        -1,
        cache_dim,
    )

    mask = slot_mapping >= 0

    flat_cache[
        slot_mapping[mask]
    ] = cache_value[mask]
```

---

## 8. `store_mla_cache`：Triton 版

```python
@triton.jit
def store_latentcache_kernel(
    latent_ptr,
    cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    token_idx = tl.program_id(0)

    slot = tl.load(
        slot_mapping_ptr
        + token_idx
    )

    if slot < 0:
        return

    offsets = tl.arange(
        0,
        D,
    )

    values = tl.load(
        latent_ptr
        + token_idx * D
        + offsets
    )

    tl.store(
        cache_ptr
        + slot * D
        + offsets,
        values,
    )
```

Wrapper：

```python
def store_mla_cache(
    cache_value,
    mla_cache,
    slot_mapping,
):
    N, D = cache_value.shape

    assert slot_mapping.numel() == N
    assert cache_value.stride(-1) == 1
    assert mla_cache.stride(-1) == 1

    store_latentcache_kernel[
        (N,)
    ](
        cache_value,
        mla_cache,
        slot_mapping,
        D,
    )
```

---

## 9. 第一阶段目标

```text
DeepSeek-V2-Lite checkpoint
        ↓
model init
        ↓
weight load
        ↓
prefill
        ↓
MLA cache write
        ↓
naive decode
        ↓
generate 2~4 tokens
```

当前先不做：

```text
TP=2
prefix cache
continuous batching 优化
absorbed MLA
latent-native kernel
Expert Parallel
```

但如果要完整加载并生成 DeepSeek-V2-Lite，还需要在真实仓库中补齐：

```text
DeepSeek MLP / MoE
gate
experts
shared experts
embedding
decoder layers
final norm
lm_head
checkpoint loader
```

---

## 10. 新 Work 落地顺序

```text
1. 读取当前原版 nano-vLLM 仓库
2. 对照真实 Linear / RMSNorm / Attention 接口
3. 新增 YaRN RoPE
4. 新增 DeepseekV2Attention
5. projection shape test
6. 接 RoPE + prefill
7. 改 MLA cache
8. naive decode
9. 补 MoE / model class
10. 对齐 checkpoint loader
11. 云 GPU 单卡验证
```
