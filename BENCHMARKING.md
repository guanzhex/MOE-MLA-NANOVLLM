# DeepSeek-V2-Lite phase-one benchmarking

This benchmark documents the single-GPU correctness baseline. It does not claim
that the naive MLA decode path is optimized.

## What is measured

`deepseek_benchmark.py` calls `LLMEngine.step()` directly and synchronizes CUDA
around every step. Tokenization and model initialization are therefore excluded
from the request latency measurements.

- **TTFT**: the sum of prefill step time for one request. With the default
  non-chunked prompts this is one prefill step, including model execution and
  sampling the first output token.
- **TPOT**: total decode time divided by the number of scheduled decode tokens.
- **E2E**: measured prefill time plus measured decode time.
- **Incremental peak memory**: peak allocated CUDA memory during the request
  minus allocated CUDA memory immediately before the request. Model weights and
  the preallocated paged cache remain in the baseline.
- **Output throughput**: all generated output tokens divided by E2E time.

The script performs one pipeline warmup by default and reports the median of
three measured runs. The ModelRunner also performs its normal initialization
warmup before the benchmark begins.

## Cache calculation

For a full reconstructed DeepSeek KV representation, each token and layer uses:

```text
num_attention_heads
  * (qk_nope_head_dim + qk_rope_head_dim + v_head_dim)
```

The phase-one MLA cache stores:

```text
kv_lora_rank + qk_rope_head_dim
```

The report labels this as representation or payload reduction. It does not
include allocator metadata, block fragmentation, model weights, or temporary
K/V tensors reconstructed by the naive decode path.

## Run

```bash
export DEEPSEEK_V2_MODEL=/root/autodl-tmp/models/DeepSeek-V2-Lite

python deepseek_benchmark.py \
  --model "$DEEPSEEK_V2_MODEL" \
  --input-lengths 32,128,256,512,1024 \
  --output-tokens 16 \
  --repeats 3 \
  --warmup-runs 1 \
  --concurrency 1,2,4
```

Results are written to:

```text
benchmark_results/deepseek_phase1_benchmark.json
benchmark_results/deepseek_phase1_benchmark.md
```

Run the reference test with `-s` to print max absolute error, mean absolute
error, and cosine similarity for the tested MLA projection and YaRN tensors:

```bash
python -m pytest -q tests/test_deepseek_v2_reference.py -s
```

## Reporting rules

Always record the model checkpoint, GPU, CUDA, PyTorch, Transformers,
FlashAttention, dtype, tensor-parallel size, prompt length, output length,
warmup count, and repeat count with the result.

Do not compare these numbers with another framework unless both sides use the
same checkpoint, dtype, prompt token IDs, output length, synchronization
boundary, warmup policy, and sampling policy.
