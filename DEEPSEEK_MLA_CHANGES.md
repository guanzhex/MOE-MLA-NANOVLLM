# nano-vLLM DeepSeek-V2 MLA 第一阶段改造说明

## 1. 改造目标

这次改造的目标，是在尽量保留 nano-vLLM 原有架构的前提下，为 DeepSeek-V2-Lite 增加一条 correctness-first 推理路径：

~~~text
prompt
  → prefill
  → 写入 MLA latent cache
  → naive decode
  → 恢复历史 K/V
  → 生成 token
~~~

第一阶段只考虑单 GPU correctness，不追求吞吐量，也暂时不实现 Tensor Parallel、Expert Parallel、absorbed MLA 或 fused kernel。

原来的 Qwen3 路径继续保留。DeepSeek 逻辑主要放在新的 model、MLA attention 和 YaRN RoPE class 中，避免把两种模型的细节混在一起。

## 2. 原 nano-vLLM 调用链

~~~text
LLMEngine
  → Scheduler
  → BlockManager
  → ModelRunner
  → Model
  → Attention
  → Cache
  → Sampler
~~~

- LLMEngine：接收 prompt、调用 tokenizer、循环执行 prefill/decode。
- Scheduler：决定本轮执行 prefill 还是 decode，以及调度哪些 sequence。
- BlockManager：分配物理 cache block，维护 block table 和 slot mapping。
- ModelRunner：准备 input ids、positions、slot mapping、block tables，并执行模型。
- Model：执行 embedding、decoder layers、final norm 和 lm head。
- Attention：消费 Q/K/V 和执行上下文，完成 prefill 或 decode。
- Sampler：根据 logits 和 temperature 采样下一个 token。

这次没有重写这条主链。Scheduler 仍然管理 token 调度，BlockManager 仍然管理分页 block，ModelRunner 仍然准备执行元数据。

## 3. DeepSeek-V2 模型结构

新增文件：

~~~text
nanovllm/models/deepseek_v2.py
~~~

其中实现了：

- DeepseekV2Attention
- DeepseekV2MLP
- DeepseekV2MoEGate
- DeepseekV2MoE
- DeepseekV2DecoderLayer
- DeepseekV2Model
- DeepseekV2ForCausalLM

### 3.1 为什么不能直接复用 Qwen Attention

Qwen 使用传统 packed QKV：

~~~text
hidden_states
  → qkv_proj
  → Q + K + V
~~~

DeepSeek MLA 使用两条投影路径：

~~~text
hidden_states
  → q_proj
  → q_nope + q_pe

hidden_states
  → kv_a_proj_with_mqa
  → compressed_kv + raw_k_pe
  → kv_a_layernorm(compressed_kv)
  → kv_b_proj
  → k_nope + value
~~~

最后组成：

~~~text
Q = [q_nope | RoPE(q_pe)]
K = [k_nope | RoPE(k_pe)]
V = value
~~~

因此 DeepSeek attention 使用独立 class，没有往原 Qwen attention 中加入大量条件分支。

### 3.2 checkpoint 参数名称

模型参数层级尽量保持和官方 checkpoint 一致：

~~~text
model.embed_tokens.weight

model.layers.0.self_attn.q_proj.weight
model.layers.0.self_attn.kv_a_proj_with_mqa.weight
model.layers.0.self_attn.kv_a_layernorm.weight
model.layers.0.self_attn.kv_b_proj.weight
model.layers.0.self_attn.o_proj.weight

model.layers.1.mlp.gate.weight
model.layers.1.mlp.experts.0.gate_proj.weight
model.layers.1.mlp.experts.0.up_proj.weight
model.layers.1.mlp.experts.0.down_proj.weight
model.layers.1.mlp.shared_experts.gate_proj.weight

model.norm.weight
lm_head.weight
~~~

MLP 没有打包成 gate_up_proj，而是保留 gate_proj、up_proj、down_proj，方便直接核对 checkpoint name 和 shape。

## 4. DeepSeek YaRN RoPE

修改文件：

~~~text
nanovllm/layers/rotary_embedding.py
~~~

原 RotaryEmbedding 和 get_rope 保持不变，继续服务 Qwen。

新增 DeepseekV2YarnRotaryEmbedding，负责：

- YaRN frequency interpolation；
- correction range 计算；
- magnitude scaling；
- DeepSeek rotary 维度排列转换；
- q_pe 和 head-shared k_pe 的旋转。

张量形状为：

~~~text
q_pe: [N, num_heads, qk_rope_head_dim]
k_pe: [N, 1,         qk_rope_head_dim]
~~~

其中 k_pe 为所有 attention heads 共享。

YaRN cos/sin cache 通过 lru_cache 在 decoder layers 之间共享，避免 27 层重复保存相同的长序列 table。

## 5. MLA Attention executor

新增文件：

~~~text
nanovllm/layers/mla_attention.py
~~~

新增 MLAAttention，与原 Attention 相互独立。

原 Attention 的 cache 语义是：

~~~text
k_cache + v_cache
~~~

MLA cache 的语义改成：

~~~text
[normalized compressed_kv | raw_k_pe]
~~~

DeepSeek-V2-Lite 单 token cache 宽度为：

~~~text
kv_lora_rank + qk_rope_head_dim
= 512 + 64
= 576
~~~

### 5.1 Cache 写入

模型计算当前 token 时，同时得到完整 Q/K/V、normalized compressed KV 和 raw k_pe。

后两者根据原 nano-vLLM 的 slot_mapping 写入分页 cache：

~~~text
logical token
  → slot_mapping
  → physical cache slot
  → [compressed_kv_norm | raw_k_pe]
~~~

block、block table 和 slot mapping 仍由原 Scheduler/BlockManager 管理。

### 5.2 Prefill

普通 prefill 时，当前 prompt 的完整 Q/K/V 可以现场计算，所以继续使用 flash_attn_varlen_func。

DeepSeek 的 V head dim 小于 Q/K head dim。prefill 会暂时把 V pad 到 Q/K head dim，FlashAttention 完成后再切回真正的 v_head_dim。

### 5.3 Naive decode

原 flash_attn_with_kvcache 假定 cache 中已经保存完整 K/V，不能直接读取 MLA latent cache。

当前 decode 每一步执行：

~~~text
block table + context length
  → gather 历史 latent cache
  → 拆出 compressed_kv_norm 和 raw_k_pe
  → kv_b_proj(compressed_kv_norm)
  → 恢复历史 k_nope 和 value
  → 对历史 raw_k_pe 应用 YaRN RoPE
  → K = [k_nope | rotated k_pe]
  → QKᵀ
  → softmax
  → probability × V
~~~

这个实现每层、每个 decode step 都会恢复全部历史 K/V，性能很差，但数据流清楚，适合作为 correctness baseline。

kv_b_proj 和 rotary module 在 forward 时传给 executor，没有在 executor 内重复注册。这样 checkpoint 中只保留官方的 self_attn.kv_b_proj.weight 参数路径。

## 6. 单卡 MoE

DeepSeek-V2-Lite 第一层使用 dense MLP，后续层使用 routed MoE。

当前实现：

- 所有 routed experts 放在同一张 GPU；
- 不做 Expert Parallel；
- 不做 All-to-All；
- gate logits 使用 FP32 softmax；
- 根据 top-k expert id 分发 token；
- expert 输出按 gate weight 在 FP32 中加权求和；
- 最后加上 shared experts 的输出。

~~~text
hidden_states
  → MoE gate
  → top-k expert ids + weights
  → routed expert forward
  → weighted sum
  + shared expert forward
  → MoE output
~~~

expert dispatch 目前使用 Python 循环，目标是容易检查，而不是高性能。

## 7. ModelRunner 和 cache 分配

修改文件：

~~~text
nanovllm/engine/model_runner.py
~~~

ModelRunner 原来硬编码 Qwen3ForCausalLM，现在根据 hf_config.model_type 选择：

~~~text
qwen3       → Qwen3ForCausalLM
deepseek_v2 → DeepseekV2ForCausalLM
~~~

Qwen cache 仍然是：

~~~text
[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
~~~

DeepSeek cache 是：

~~~text
[num_layers, num_blocks, block_size,
 kv_lora_rank + qk_rope_head_dim]
~~~

每层 cache 挂载到 layer.self_attn.attn.mla_cache。

ModelRunner 只负责 cache 的分配和挂载，不负责 kv_b_proj、RoPE 或历史 K/V 恢复。

MLA naive decode 包含动态 gather 和 Python 循环，目前不适合 CUDA Graph，所以 DeepSeek 第一阶段自动使用 eager execution。Qwen 的 CUDA Graph 路径不变。

## 8. Prefix cache

修改文件：

~~~text
nanovllm/engine/block_manager.py
nanovllm/engine/scheduler.py
~~~

第一阶段没有实现从 MLA prefix cache 恢复历史 K/V，因此给 BlockManager 增加了一个默认开启的开关：

~~~text
Qwen     → enable_prefix_cache=True
DeepSeek → enable_prefix_cache=False
~~~

Scheduler 的主要调度流程没有重写。

如果 DeepSeek 进入 chunked prefill/prefix prefill，MLA executor 会明确抛出 NotImplementedError，避免静默计算出错误结果。

## 9. Config、Tokenizer 和 Loader

DeepSeek-V2-Lite checkpoint 自带 remote config/modeling 文件，所以 Config 和 Tokenizer 加入 trust_remote_code=True。

ModelRunner 同时兼容 transformers 版本差异带来的 dtype 和 torch_dtype 字段。

Loader 增加了以下检查：

1. checkpoint 目录下是否存在 safetensors；
2. checkpoint name 能否映射到 runtime parameter；
3. checkpoint shape 和 runtime shape 是否一致；
4. packed Qwen shard 是否成功加载；
5. DeepSeek strict 模式下是否有 runtime parameter 没有初始化。

DeepSeek 使用 strict load；Qwen 暂时保持非 strict，减少对原路径的影响。

## 10. 测试

新增：

~~~text
tests/test_deepseek_v2_components.py
tests/test_deepseek_v2_reference.py
~~~

组件测试检查：

- MLA projection 六个张量的 shape；
- q_pe 每 head 独立；
- k_pe head-shared；
- latent cache 是否按 slot mapping 写入；
- runtime 参数名是否符合官方 checkpoint 层级。

Reference test 不加载两份完整模型，而是只读取真实 checkpoint 第 0 层 attention 权重，对比官方实现和 nano-vLLM 实现的：

- q_nope
- q_pe
- normalized compressed_kv
- raw k_pe
- k_nope
- value
- YaRN 后的 q_pe/k_pe

云 GPU 上可以运行：

~~~bash
pytest -q tests/test_deepseek_v2_components.py

DEEPSEEK_V2_MODEL=/path/to/DeepSeek-V2-Lite \
pytest -q tests/test_deepseek_v2_reference.py
~~~

## 11. 当前状态和限制

已经落地：

~~~text
DeepSeek model class
MLA projections
DeepSeek YaRN RoPE
latent cache allocation/write
prefill attention
naive decode attention
single-GPU MoE
checkpoint name/shape validation
reference component test
~~~

当前还没有在真实 DeepSeek-V2-Lite checkpoint 上完成端到端运行。

当前限制：

- 只支持单 GPU；
- 只支持 DeepSeek-V2-Lite 的 q_lora_rank=None；
- 不支持 prefix cache；
- 不支持 chunked prefill；
- 不支持 Tensor Parallel；
- 不支持 Expert Parallel；
- 不支持 CUDA Graph；
- 不支持 absorbed MLA；
- 不支持 fused/latent-native MLA kernel；
- naive decode 每层、每步都会恢复全部历史 K/V。

本地只完成了 Python 语法和 diff 检查。真实 checkpoint load、prefill、decode token 1～4 和最终生成仍需要在有足够显存的云 GPU 上验证。

## 12. 面试时如何概括

可以这样说明第一阶段设计：

> 我没有修改 nano-vLLM 的调度架构。Scheduler 和 BlockManager 仍然负责逻辑 token 到物理 block 的映射，变化的是 cache 中保存的数据语义。
>
> Qwen 保存完整 K/V；DeepSeek MLA 保存 normalized compressed KV 和 raw positional key。Prefill 时完整 Q/K/V 可以现场计算，decode 时根据 block table gather latent，使用 kv_b_proj 恢复历史 content key/value，并对 raw positional key 重新做 RoPE。
>
> 这个 naive decode 很慢，但它把模型数学、分页 cache 管理和 attention 执行三个层次分开了。correctness 验证后，可以单独替换 decode executor，继续做 absorbed MLA 或专用 kernel，而不需要重写 Scheduler 和模型参数结构。
