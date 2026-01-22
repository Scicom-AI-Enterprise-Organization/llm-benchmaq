import subprocess


def run_benchmark(model_path, port, output_dir, result_name, ctx, output_len, num_prompts, concurrency):
    print()
    print("=" * 64)
    print(f"BENCHMARK: {result_name}")
    print("=" * 64)

    subprocess.run([
        "vllm", "bench", "serve",
        "--backend", "vllm",
        "--base-url", f"http://localhost:{port}",
        "--model", model_path,
        "--endpoint", "/v1/completions",
        "--dataset-name", "random",
        "--random-input-len", str(ctx),
        "--random-output-len", str(output_len),
        "--num-prompts", str(num_prompts),
        "--max-concurrency", str(concurrency),
        "--request-rate", "inf",
        "--ignore-eos",
        "--save-result",
        "--result-dir", output_dir,
        "--result-filename", f"{result_name}.json",
        "--percentile-metrics", "ttft,tpot,itl,e2el",
    ])
