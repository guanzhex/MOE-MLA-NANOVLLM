import argparse
import json
import platform
import statistics
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter

import torch

from nanovllm import LLM, SamplingParams


MIB = 1024**2


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def make_prompt_tokens(tokenizer, length: int) -> list[int]:
    if length <= 0:
        raise ValueError("prompt length must be positive")
    seed = tokenizer.encode(
        "Multi-head latent attention compresses the key value cache. ",
        add_special_tokens=False,
    )
    if not seed:
        raise RuntimeError("tokenizer returned an empty benchmark seed")

    tokens = []
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if bos_token_id is not None:
        tokens.append(bos_token_id)
    remaining = length - len(tokens)
    if remaining > 0:
        repeats = (remaining + len(seed) - 1) // len(seed)
        tokens.extend((seed * repeats)[:remaining])
    return tokens


def run_batch(
    llm: LLM,
    prompt_tokens: list[int],
    num_requests: int,
    output_tokens: int,
) -> dict:
    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=output_tokens,
        ignore_eos=True,
    )
    for _ in range(num_requests):
        llm.add_request(list(prompt_tokens), sampling_params)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()

    prefill_times_ms = []
    decode_times_ms = []
    finished_outputs = []
    scheduled_prefill_tokens = 0
    scheduled_decode_tokens = 0

    while not llm.is_finished():
        torch.cuda.synchronize()
        start = perf_counter()
        outputs, num_tokens = llm.step()
        torch.cuda.synchronize()
        elapsed_ms = (perf_counter() - start) * 1000

        if num_tokens > 0:
            prefill_times_ms.append(elapsed_ms)
            scheduled_prefill_tokens += num_tokens
        else:
            decode_times_ms.append(elapsed_ms)
            scheduled_decode_tokens += -num_tokens
        finished_outputs.extend(outputs)

    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    generated_tokens = sum(len(token_ids) for _, token_ids in finished_outputs)
    total_ms = sum(prefill_times_ms) + sum(decode_times_ms)
    decode_ms = sum(decode_times_ms)

    return {
        "num_requests": num_requests,
        "prompt_tokens_per_request": len(prompt_tokens),
        "output_tokens_per_request": output_tokens,
        "generated_tokens": generated_tokens,
        "scheduled_prefill_tokens": scheduled_prefill_tokens,
        "scheduled_decode_tokens": scheduled_decode_tokens,
        "prefill_steps": len(prefill_times_ms),
        "decode_steps": len(decode_times_ms),
        "ttft_ms": sum(prefill_times_ms),
        "decode_total_ms": decode_ms,
        "decode_step_median_ms": median(decode_times_ms),
        "tpot_ms": decode_ms / scheduled_decode_tokens
        if scheduled_decode_tokens
        else 0.0,
        "e2e_ms": total_ms,
        "output_throughput_tok_s": generated_tokens / (total_ms / 1000),
        "decode_throughput_tok_s": scheduled_decode_tokens / (decode_ms / 1000)
        if decode_ms
        else 0.0,
        "incremental_peak_allocated_mib": max(
            0, peak_allocated - baseline_allocated
        )
        / MIB,
        "incremental_peak_reserved_mib": max(0, peak_reserved - baseline_reserved)
        / MIB,
    }


def aggregate_runs(runs: list[dict]) -> dict:
    stable_fields = {
        key: runs[0][key]
        for key in (
            "num_requests",
            "prompt_tokens_per_request",
            "output_tokens_per_request",
            "generated_tokens",
            "scheduled_prefill_tokens",
            "scheduled_decode_tokens",
            "prefill_steps",
            "decode_steps",
        )
    }
    metric_fields = (
        "ttft_ms",
        "decode_total_ms",
        "decode_step_median_ms",
        "tpot_ms",
        "e2e_ms",
        "output_throughput_tok_s",
        "decode_throughput_tok_s",
        "incremental_peak_allocated_mib",
        "incremental_peak_reserved_mib",
    )
    stable_fields.update(
        {key: median([run[key] for run in runs]) for key in metric_fields}
    )
    return stable_fields


def cache_metrics(llm: LLM) -> dict:
    runner = llm.model_runner
    config = runner.config
    hf_config = config.hf_config
    dtype_bytes = runner.dtype.itemsize
    num_heads = hf_config.num_attention_heads
    full_elements = num_heads * (
        hf_config.qk_nope_head_dim
        + hf_config.qk_rope_head_dim
        + hf_config.v_head_dim
    )
    latent_elements = hf_config.kv_lora_rank + hf_config.qk_rope_head_dim
    full_bytes = full_elements * dtype_bytes
    latent_bytes = latent_elements * dtype_bytes
    num_layers = hf_config.num_hidden_layers

    context_rows = []
    for context_length in (1024, 4096, 32768):
        context_rows.append(
            {
                "context_tokens": context_length,
                "full_kv_mib": full_bytes * num_layers * context_length / MIB,
                "mla_cache_mib": latent_bytes
                * num_layers
                * context_length
                / MIB,
            }
        )

    cache = runner.kv_cache
    return {
        "dtype": str(runner.dtype),
        "dtype_bytes": dtype_bytes,
        "num_layers": num_layers,
        "num_attention_heads": num_heads,
        "full_kv_elements_per_token_layer": full_elements,
        "mla_elements_per_token_layer": latent_elements,
        "full_kv_bytes_per_token_layer": full_bytes,
        "mla_bytes_per_token_layer": latent_bytes,
        "compression_ratio": full_bytes / latent_bytes,
        "reduction_percent": (1 - latent_bytes / full_bytes) * 100,
        "allocated_cache_shape": list(cache.shape),
        "allocated_cache_mib": cache.numel() * cache.element_size() / MIB,
        "allocated_token_capacity": config.num_kvcache_blocks
        * config.kvcache_block_size,
        "contexts": context_rows,
    }


def environment_metrics(model_path: str) -> dict:
    props = torch.cuda.get_device_properties(0)
    return {
        "model": model_path,
        "gpu": props.name,
        "gpu_total_memory_gib": props.total_memory / 1024**3,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": package_version("transformers"),
        "flash_attn": package_version("flash-attn"),
        "triton": package_version("triton"),
    }


def markdown_report(results: dict) -> str:
    env = results["environment"]
    cache = results["cache"]
    lines = [
        "# DeepSeek-V2-Lite nano-vLLM Phase-One Benchmark",
        "",
        "## Environment",
        "",
        "| Item | Value |",
        "|---|---|",
    ]
    for key, value in env.items():
        lines.append(f"| {key} | {value} |")

    lines.extend(
        [
            "",
            "## MLA cache representation",
            "",
            "| Metric | Full KV | MLA latent |",
            "|---|---:|---:|",
            f"| Elements/token/layer | {cache['full_kv_elements_per_token_layer']} | {cache['mla_elements_per_token_layer']} |",
            f"| Bytes/token/layer | {cache['full_kv_bytes_per_token_layer']} | {cache['mla_bytes_per_token_layer']} |",
            "",
            f"Compression: **{cache['compression_ratio']:.2f}x**; reduction: **{cache['reduction_percent']:.2f}%**.",
            "",
            f"Allocated MLA cache: {cache['allocated_cache_mib']:.2f} MiB for {cache['allocated_token_capacity']} token slots.",
            "",
            "| Context | Full KV MiB | MLA MiB |",
            "|---:|---:|---:|",
        ]
    )
    for row in cache["contexts"]:
        lines.append(
            f"| {row['context_tokens']} | {row['full_kv_mib']:.2f} | {row['mla_cache_mib']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Single-request latency",
            "",
            "Tokenizer and model initialization are excluded. Values are medians across measured runs.",
            "",
            "| Input tokens | Output tokens | TTFT ms | TPOT ms | E2E ms | Decode tok/s | Incremental peak MiB |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in results["single_request"]:
        lines.append(
            f"| {row['prompt_tokens_per_request']} | {row['output_tokens_per_request']} | "
            f"{row['ttft_ms']:.2f} | {row['tpot_ms']:.2f} | {row['e2e_ms']:.2f} | "
            f"{row['decode_throughput_tok_s']:.2f} | {row['incremental_peak_allocated_mib']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Multi-request smoke",
            "",
            "| Requests | Prompt/request | Output/request | Prefill steps | Decode steps | E2E ms | Output tok/s |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in results["multi_request"]:
        lines.append(
            f"| {row['num_requests']} | {row['prompt_tokens_per_request']} | "
            f"{row['output_tokens_per_request']} | {row['prefill_steps']} | {row['decode_steps']} | "
            f"{row['e2e_ms']:.2f} | {row['output_throughput_tok_s']:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_lengths(value: str) -> list[int]:
    lengths = [int(item) for item in value.split(",") if item.strip()]
    if not lengths or any(length <= 0 for length in lengths):
        raise argparse.ArgumentTypeError("lengths must be positive comma-separated integers")
    return lengths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure DeepSeek-V2-Lite MLA correctness-baseline metrics"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-lengths", type=parse_lengths, default=[32, 128, 256, 512, 1024])
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--multi-prompt-length", type=int, default=128)
    parser.add_argument("--multi-output-tokens", type=int, default=8)
    parser.add_argument("--concurrency", type=parse_lengths, default=[1, 2, 4])
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--output-dir", default="benchmark_results", help="JSON and Markdown output directory"
    )
    args = parser.parse_args()

    max_concurrency = max(args.concurrency)
    max_prompt_tokens = max(max(args.input_lengths), args.multi_prompt_length)
    max_output_tokens = max(args.output_tokens, args.multi_output_tokens)
    max_model_len = max_prompt_tokens + max_output_tokens
    max_num_batched_tokens = max(
        max(args.input_lengths), args.multi_prompt_length * max_concurrency
    )

    print("initializing model and allocating MLA cache...", flush=True)
    initialization_start = perf_counter()
    llm = LLM(
        args.model,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_concurrency,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    torch.cuda.synchronize()
    initialization_seconds = perf_counter() - initialization_start

    results = {
        "environment": environment_metrics(args.model),
        "settings": {
            "dtype": str(llm.model_runner.dtype),
            "tensor_parallel_size": 1,
            "enforce_eager": True,
            "max_model_len": max_model_len,
            "max_num_batched_tokens": max_num_batched_tokens,
            "max_num_seqs": max_concurrency,
            "repeats": args.repeats,
            "warmup_runs": args.warmup_runs,
            "initialization_seconds": initialization_seconds,
        },
        "cache": cache_metrics(llm),
        "single_request": [],
        "multi_request": [],
    }

    warmup_prompt = make_prompt_tokens(llm.tokenizer, min(args.input_lengths))
    for _ in range(args.warmup_runs):
        run_batch(llm, warmup_prompt, 1, min(2, args.output_tokens))

    for input_length in args.input_lengths:
        print(f"single request: input={input_length}, output={args.output_tokens}", flush=True)
        prompt_tokens = make_prompt_tokens(llm.tokenizer, input_length)
        runs = [
            run_batch(llm, prompt_tokens, 1, args.output_tokens)
            for _ in range(args.repeats)
        ]
        results["single_request"].append(aggregate_runs(runs))

    multi_prompt = make_prompt_tokens(llm.tokenizer, args.multi_prompt_length)
    for concurrency in args.concurrency:
        print(
            f"multi request: concurrency={concurrency}, input={args.multi_prompt_length}, "
            f"output={args.multi_output_tokens}",
            flush=True,
        )
        runs = [
            run_batch(llm, multi_prompt, concurrency, args.multi_output_tokens)
            for _ in range(args.repeats)
        ]
        results["multi_request"].append(aggregate_runs(runs))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "deepseek_phase1_benchmark.json"
    markdown_path = output_dir / "deepseek_phase1_benchmark.md"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    markdown_path.write_text(markdown_report(results))
    print(f"wrote {json_path}")
    print(f"wrote {markdown_path}")


if __name__ == "__main__":
    main()
