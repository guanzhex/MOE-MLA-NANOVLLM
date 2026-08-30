import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from nanovllm.layers.mla_attention import MLAAttention
from nanovllm.layers.rotary_embedding import get_deepseek_yarn_rope, yarn_get_mscale


class DeepseekV2Attention(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        if tp_size != 1:
            raise NotImplementedError("DeepSeek-V2 phase one only supports tensor_parallel_size=1")

        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.num_heads = self.total_num_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        if self.q_lora_rank is not None:
            raise NotImplementedError("DeepSeek-V2 phase one only supports q_lora_rank=None")
        self.q_proj = ColumnParallelLinear(
            self.hidden_size,
            self.total_num_heads * self.q_head_dim,
            bias=False,
        )
        attention_bias = getattr(config, "attention_bias", False)
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=attention_bias,
        )
        self.kv_a_layernorm = RMSNorm(
            self.kv_lora_rank,
            eps=config.rms_norm_eps,
        )
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.total_num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.v_head_dim,
            self.hidden_size,
            bias=attention_bias,
        )

        rope_scaling = config.rope_scaling
        if not isinstance(rope_scaling, dict) or rope_scaling.get("type") != "yarn":
            raise NotImplementedError("DeepSeek-V2 phase one requires YaRN rope_scaling")
        self.rotary_emb = get_deepseek_yarn_rope(
            dim=self.qk_rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            scaling_factor=rope_scaling["factor"],
            original_max_position_embeddings=rope_scaling.get(
                "original_max_position_embeddings", 4096
            ),
            beta_fast=rope_scaling.get("beta_fast", 32),
            beta_slow=rope_scaling.get("beta_slow", 1),
            mscale=rope_scaling.get("mscale", 1),
            mscale_all_dim=rope_scaling.get("mscale_all_dim", 0),
        )

        scale = self.q_head_dim**-0.5
        mscale_all_dim = rope_scaling.get("mscale_all_dim", 0)
        if mscale_all_dim:
            mscale = yarn_get_mscale(rope_scaling["factor"], mscale_all_dim)
            scale *= mscale * mscale
        self.attn = MLAAttention(
            num_heads=self.num_heads,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            scale=scale,
        )

    def project(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Expose MLA projection tensors so their shapes can be tested directly."""
        q = self.q_proj(hidden_states).view(-1, self.num_heads, self.q_head_dim)
        q_nope, q_pe = q.split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        kv_a = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, raw_k_pe = kv_a.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        compressed_kv = self.kv_a_layernorm(compressed_kv)
        kv = self.kv_b_proj(compressed_kv).view(
            -1,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        k_nope, value = kv.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        return q_nope, q_pe, compressed_kv, raw_k_pe.unsqueeze(1), k_nope, value

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        q_nope, q_pe, compressed_kv, raw_k_pe, k_nope, value = self.project(
            hidden_states
        )
        q_pe, k_pe = self.rotary_emb(positions, q_pe, raw_k_pe)
        query = torch.cat([q_nope, q_pe], dim=-1)
        key = torch.cat(
            [k_nope, k_pe.expand(-1, self.num_heads, -1)], dim=-1
        )
        output = self.attn(
            query,
            key,
            value,
            compressed_kv,
            raw_k_pe,
            positions,
            self.kv_b_proj,
            self.rotary_emb,
        )
        return self.o_proj(output.flatten(1, -1))


class DeepseekV2MLP(nn.Module):

    def __init__(self, config, intermediate_size: int | None = None) -> None:
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        # Keep the three official names. Packing is unnecessary for phase-one
        # correctness and would make expert checkpoint mapping less transparent.
        self.gate_proj = ReplicatedLinear(config.hidden_size, intermediate_size)
        self.up_proj = ReplicatedLinear(config.hidden_size, intermediate_size)
        self.down_proj = ReplicatedLinear(intermediate_size, config.hidden_size)
        if config.hidden_act != "silu":
            raise NotImplementedError(f"unsupported DeepSeek activation: {config.hidden_act}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DeepseekV2MoEGate(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.norm_topk_prob = config.norm_topk_prob
        if config.scoring_func != "softmax" or config.topk_method != "greedy":
            raise NotImplementedError(
                "DeepSeek-V2 phase one supports softmax + greedy expert routing only"
            )
        self.weight = nn.Parameter(torch.empty(self.num_experts, config.hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = F.linear(
            hidden_states.float(), self.weight.float(), bias=None
        ).softmax(dim=-1)
        topk_weight, topk_idx = torch.topk(
            scores, k=self.top_k, dim=-1, sorted=False
        )
        if self.top_k > 1 and self.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        else:
            topk_weight = topk_weight * self.routed_scaling_factor
        return topk_idx, topk_weight


class DeepseekV2MoE(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.num_experts_per_tok = config.num_experts_per_tok
        self.experts = nn.ModuleList(
            [
                DeepseekV2MLP(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(config.n_routed_experts)
            ]
        )
        self.gate = DeepseekV2MoEGate(config)
        self.shared_experts = None
        if config.n_shared_experts is not None:
            self.shared_experts = DeepseekV2MLP(
                config,
                intermediate_size=config.moe_intermediate_size
                * config.n_shared_experts,
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        identity = hidden_states
        topk_idx, topk_weight = self.gate(hidden_states)
        expert_outputs = torch.empty(
            *topk_idx.shape,
            hidden_states.size(-1),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        for expert_idx, expert in enumerate(self.experts):
            token_idx, topk_slot = torch.where(topk_idx == expert_idx)
            if token_idx.numel():
                expert_outputs[token_idx, topk_slot] = expert(hidden_states[token_idx])
        output = (
            expert_outputs.float() * topk_weight.unsqueeze(-1)
        ).sum(dim=1).to(hidden_states.dtype)
        if self.shared_experts is not None:
            output = output + self.shared_experts(identity)
        return output


class DeepseekV2DecoderLayer(nn.Module):

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = DeepseekV2Attention(config)
        use_moe = (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        )
        self.mlp = DeepseekV2MoE(config) if use_moe else DeepseekV2MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class DeepseekV2Model(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                DeepseekV2DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class DeepseekV2ForCausalLM(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.model = DeepseekV2Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
