import argparse

from nanovllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(
        description="DeepSeek-V2-Lite phase-one correctness smoke test"
    )
    parser.add_argument("--model", required=True, help="local checkpoint directory")
    parser.add_argument(
        "--prompt",
        default="The key idea of multi-head latent attention is",
    )
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()

    llm = LLM(
        args.model,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )
    output = llm.generate([args.prompt], sampling_params, use_tqdm=False)[0]
    print("token_ids:", output["token_ids"])
    print("text:", repr(output["text"]))


if __name__ == "__main__":
    main()
