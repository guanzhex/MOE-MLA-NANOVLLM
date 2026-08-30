# DeepSeek-V2-Lite 云端实验步骤

## 1. GPU 选择

DeepSeek-V2-Lite 有 16B 总参数、2.4B 激活参数。官方给出的 BF16 推理要求是单张约 40GB GPU。

本项目当前把所有 experts 放在单 GPU，而且包含 correctness-first 的 naive decode，因此建议：

- 首选：A100 80GB 或 H100 80GB；
- 可尝试：L40S 48GB、RTX 6000 Ada 48GB、A6000 48GB；
- 不建议首轮使用 40GB 以下 GPU；
- 不要选择多张小显存卡，因为第一阶段还不支持 TP 或 Expert Parallel。

80GB 卡的主要价值不是性能，而是第一次调试时给模型权重、FlashAttention workspace、MoE 临时张量和 MLA cache 留出余量。

## 2. 推荐环境

~~~text
Ubuntu 22.04
Python 3.10 或 3.11
CUDA 12.x
单张 NVIDIA GPU
~~~

先确认 GPU：

~~~bash
nvidia-smi
~~~

创建环境：

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
~~~

安装与云主机 CUDA 匹配的 PyTorch。下面只是 CUDA 12.4 示例：

~~~bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
~~~

安装项目：

~~~bash
pip install -e .
pip install pytest huggingface_hub
~~~

如果 flash-attn 构建失败，可以单独重装：

~~~bash
pip install flash-attn --no-build-isolation
~~~

## 3. 下载 checkpoint

~~~bash
huggingface-cli download deepseek-ai/DeepSeek-V2-Lite \
  --local-dir /workspace/models/DeepSeek-V2-Lite
~~~

检查关键文件：

~~~bash
ls /workspace/models/DeepSeek-V2-Lite
~~~

目录中应该包含 config.json、tokenizer 文件、configuration_deepseek.py、modeling_deepseek.py 和 safetensors shards。

不要把 checkpoint 提交到 GitHub。

## 4. 第一轮：组件测试

先运行不需要完整模型 forward 的组件测试：

~~~bash
pytest -q tests/test_deepseek_v2_components.py
~~~

它会检查：

- MLA projection shape；
- q_pe 和 head-shared k_pe；
- latent cache slot 写入；
- checkpoint 参数命名层级。

## 5. 第二轮：官方单层 reference test

这个测试只加载 checkpoint 的第 0 层 attention 权重，不会同时加载两个完整 16B 模型：

~~~bash
DEEPSEEK_V2_MODEL=/workspace/models/DeepSeek-V2-Lite \
pytest -q tests/test_deepseek_v2_reference.py -s
~~~

它会对比官方实现与 nano-vLLM 实现的：

- q_nope；
- q_pe；
- normalized compressed_kv；
- raw k_pe；
- k_nope；
- value；
- YaRN 后的 q_pe/k_pe。

如果这里失败，先不要跑完整模型。projection 不一致通常意味着参数 shape、RMSNorm 或张量 reshape 有问题；只有 RoPE 不一致通常意味着 rotary layout 或 YaRN scaling 有问题。

## 6. 第三轮：真实 checkpoint load 和 4-token smoke test

为了降低第一次 warmup 的时间和显存波动，先把最大长度限制为 256，只生成 4 个 token：

~~~bash
python deepseek_smoke.py \
  --model /workspace/models/DeepSeek-V2-Lite \
  --max-model-len 256 \
  --max-tokens 4 \
  --gpu-memory-utilization 0.9
~~~

期望依次通过：

~~~text
checkpoint load
model init
prefill
MLA cache write
sample first token from prefill logits
decode step 1
decode steps 2-4
print generated token ids and text
~~~

第一轮只需要确认生成 token id 合法且程序没有中途报错，不要用生成文本质量判断 MLA 是否正确。

## 7. 推荐的调试顺序

遇到错误时按下面顺序处理，不要一次改多个模块：

1. Loader 报 unknown parameter：核对 checkpoint name 和 runtime parameter name。
2. Loader 报 shape mismatch：核对 config 字段和 Linear 输入/输出维度。
3. Projection reference test 失败：检查 q/kv projection、RMSNorm 和 reshape。
4. RoPE reference test 失败：检查偶奇维排列、YaRN frequency 和 mscale。
5. Prefill 失败：检查 Q/K/V head dim、V padding 和 cu_seqlens。
6. Cache write 失败：检查 slot_mapping 和 cache_dim=576。
7. Decode 失败：检查 block table gather、context length 和历史位置。
8. 能生成但 logits 差异大：增加逐层 hidden state/reference logits 对比。

## 8. 记录实验信息

每次运行至少保存：

~~~bash
nvidia-smi
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "import flash_attn; print(flash_attn.__version__)"
git rev-parse HEAD
~~~

建议把完整日志保存到：

~~~bash
mkdir -p logs
python deepseek_smoke.py \
  --model /workspace/models/DeepSeek-V2-Lite \
  --max-model-len 256 \
  --max-tokens 4 \
  2>&1 | tee logs/deepseek_smoke.log
~~~

如果需要继续协助定位，把以下内容带回来：

- git commit；
- GPU 型号与显存；
- torch、CUDA、flash-attn 版本；
- 完整 traceback；
- 最后一个成功阶段；
- 如果是 shape 错误，附 checkpoint shape 和 runtime shape。
