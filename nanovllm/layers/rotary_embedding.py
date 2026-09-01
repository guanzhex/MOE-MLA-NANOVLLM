import math
from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


def apply_deepseek_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Match the checkpoint reference's dtype-preserving YaRN arithmetic."""
    x1, x2 = torch.chunk(x, 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
):
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb


def yarn_find_correction_dim(
    num_rotations: float,
    dim: int,
    base: float = 10000,
    max_position_embeddings: int = 2048,
) -> float:
    return (
        dim
        * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))
        / (2 * math.log(base))
    )


def yarn_find_correction_range(
    low_rot: float,
    high_rot: float,
    dim: int,
    base: float = 10000,
    max_position_embeddings: int = 2048,
) -> tuple[int, int]:
    low = math.floor(
        yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)


def yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_linear_ramp_mask(
    min_value: float,
    max_value: float,
    dim: int,
) -> torch.Tensor:
    if min_value == max_value:
        max_value += 0.001
    linear = (torch.arange(dim, dtype=torch.float32) - min_value) / (
        max_value - min_value
    )
    return linear.clamp(0, 1)


class DeepseekV2YarnRotaryEmbedding(nn.Module):
    """DeepSeek-V2 YaRN RoPE for the decoupled q_pe and shared k_pe."""

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
    ) -> None:
        super().__init__()
        self.dim = dim

        freq_indices = torch.arange(0, dim, 2, dtype=torch.float32) / dim
        freq_extra = 1.0 / (base**freq_indices)
        freq_inter = 1.0 / (scaling_factor * base**freq_indices)
        low, high = yarn_find_correction_range(
            beta_fast,
            beta_slow,
            dim,
            base,
            original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2)
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask

        positions = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.einsum("i,j -> ij", positions, inv_freq)
        yarn_mscale = yarn_get_mscale(
            scaling_factor, mscale
        ) / yarn_get_mscale(scaling_factor, mscale_all_dim)
        cache = torch.cat(
            (freqs.cos() * yarn_mscale, freqs.sin() * yarn_mscale), dim=-1
        ).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @staticmethod
    def _to_rotate_half_layout(x: torch.Tensor) -> torch.Tensor:
        # DeepSeek's q_pe/k_pe projection stores adjacent rotary pairs, while
        # the reference rotate_half operation consumes first-half/second-half
        # layout. Convert once before applying the shared half-rotation helper.
        shape = x.shape
        return x.view(*shape[:-1], shape[-1] // 2, 2).transpose(-1, -2).reshape(shape)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        # The official DeepSeek cache casts cos/sin to the activation dtype on
        # first use, then performs the rotation without promoting BF16 inputs
        # to FP32. Keep that numerical order for checkpoint-level parity while
        # leaving the original Qwen rotary helper unchanged.
        cos = cos.to(query.dtype)
        sin = sin.to(query.dtype)
        query = apply_deepseek_rotary_emb(
            self._to_rotate_half_layout(query), cos, sin
        )
        key = apply_deepseek_rotary_emb(
            self._to_rotate_half_layout(key), cos, sin
        )
        return query, key


@lru_cache(1)
def get_deepseek_yarn_rope(
    dim: int,
    max_position_embeddings: int,
    base: float,
    scaling_factor: float,
    original_max_position_embeddings: int = 4096,
    beta_fast: float = 32,
    beta_slow: float = 1,
    mscale: float = 1,
    mscale_all_dim: float = 0,
) -> DeepseekV2YarnRotaryEmbedding:
    return DeepseekV2YarnRotaryEmbedding(
        dim=dim,
        max_position_embeddings=max_position_embeddings,
        base=base,
        scaling_factor=scaling_factor,
        original_max_position_embeddings=original_max_position_embeddings,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
        mscale=mscale,
        mscale_all_dim=mscale_all_dim,
    )
