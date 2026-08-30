import torch
import torch.nn.functional as F
from torch import nn

from flash_attn import flash_attn_varlen_func

from nanovllm.utils.context import get_context


def store_mla_cache(
    cache_value: torch.Tensor,
    mla_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Write [normalized compressed_kv | raw k_pe] into paged slots."""
    assert cache_value.ndim == 2
    assert cache_value.size(-1) == mla_cache.size(-1)
    assert slot_mapping.numel() == cache_value.size(0)
    valid = slot_mapping >= 0
    flat_cache = mla_cache.view(-1, mla_cache.size(-1))
    flat_cache[slot_mapping[valid].long()] = cache_value[valid]


class MLAAttention(nn.Module):
    """Correctness-first MLA executor using latent cache and naive decode."""

    def __init__(
        self,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        kv_lora_rank: int,
        scale: float,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.scale = scale
        self.mla_cache = torch.tensor([])

    @property
    def cache_dim(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim

    def _gather_sequence_cache(
        self,
        block_table: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        block_size = self.mla_cache.size(1)
        logical_positions = torch.arange(seq_len, device=block_table.device)
        logical_blocks = torch.div(logical_positions, block_size, rounding_mode="floor")
        offsets = logical_positions.remainder(block_size)
        physical_blocks = block_table[logical_blocks].long()
        slots = physical_blocks * block_size + offsets
        return self.mla_cache.view(-1, self.cache_dim)[slots]

    def _decode_one(
        self,
        query: torch.Tensor,
        block_table: torch.Tensor,
        seq_len: int,
        positions_dtype: torch.dtype,
        kv_b_proj: nn.Module,
        rotary_emb: nn.Module,
    ) -> torch.Tensor:
        history = self._gather_sequence_cache(block_table, seq_len)
        compressed_kv = history[:, : self.kv_lora_rank]
        raw_k_pe = history[:, self.kv_lora_rank :].unsqueeze(1)

        history_kv = kv_b_proj(compressed_kv).view(
            seq_len,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        history_k_nope, history_v = history_kv.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        history_positions = torch.arange(
            seq_len, device=query.device, dtype=positions_dtype
        )
        _, history_k_pe = rotary_emb(history_positions, raw_k_pe, raw_k_pe)
        history_k = torch.cat(
            [history_k_nope, history_k_pe.expand(-1, self.num_heads, -1)], dim=-1
        )

        # query: [1, H, D], history_k: [S, H, D], history_v: [S, H, Dv]
        scores = torch.einsum("qhd,khd->hqk", query, history_k) * self.scale
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        return torch.einsum("hqk,khd->qhd", probs, history_v)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        compressed_kv: torch.Tensor,
        raw_k_pe: torch.Tensor,
        positions: torch.Tensor,
        kv_b_proj: nn.Module,
        rotary_emb: nn.Module,
    ) -> torch.Tensor:
        context = get_context()
        if self.mla_cache.numel():
            cache_value = torch.cat([compressed_kv, raw_k_pe.squeeze(1)], dim=-1)
            store_mla_cache(cache_value, self.mla_cache, context.slot_mapping)

        if context.is_prefill:
            if context.block_tables is not None:
                raise NotImplementedError("MLA prefix/chunked prefill is not part of phase one")
            # FlashAttention requires V to use the same head dimension as Q/K.
            padded_v = F.pad(v, (0, q.size(-1) - v.size(-1)))
            output = flash_attn_varlen_func(
                q,
                k,
                padded_v,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,
            )
            return output[..., : self.v_head_dim]

        outputs = []
        for seq_idx in range(q.size(0)):
            outputs.append(
                self._decode_one(
                    q[seq_idx : seq_idx + 1],
                    context.block_tables[seq_idx],
                    int(context.context_lens[seq_idx].item()),
                    positions.dtype,
                    kv_b_proj,
                    rotary_emb,
                )
            )
        return torch.cat(outputs, dim=0)
