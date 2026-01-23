# vLLM Engine

Benchmarking module for [vLLM](https://github.com/vllm-project/vllm).

## How It Works

1. Starts vLLM server
2. Runs benchmarks across all parameter combinations
3. Saves results to JSON
4. Repeats for each TP/DP/PP configuration

## Usage

```bash
# Install with vllm
uv pip install "benchmaxxing[vllm] @ git+https://github.com/Scicom-AI-Enterprise-Organization/llm-benchmaxxing.git"

# Download model
huggingface-cli download <huggingface_model_path> \
  --local-dir /download/dir

# single run
benchmaxxing bench examples/1_run_single.yaml

# multiple run
benchmaxxing bench examples/2_run_multiple.yaml
```
## Config Format

```yaml
runs:
  - name: "my-benchmark"
    engine: "vllm"
    serve:
      model_path: "meta-llama/Llama-2-7b-hf"  # HuggingFace model or local path
      port: 8000                               # Server port
      gpu_memory_utilization: 0.9              # GPU memory usage (0.0-1.0)
      max_model_len: 4096                      # Maximum sequence length
      max_num_seqs: 256                        # Maximum concurrent sequences
      dtype: "bfloat16"                        # Data type: bfloat16, float16, auto
      disable_log_requests: true               # Disable request logging
      enable_expert_parallel: false            # Expert parallelism for MoE models
      tp_dp_pairs:                             # Parallelism configurations to test
        - tp: 1                                # Tensor parallelism
          dp: 1                                # Data parallelism
          pp: 1                                # Pipeline parallelism
    bench:
      save_results: true                       # Save results to JSON
      output_dir: "./results"                  # Output directory
      context_size: [512, 1024]                # Input context sizes to test
      concurrency: [50, 100]                   # Concurrency levels to test
      num_prompts: [100]                       # Number of prompts per test
      output_len: [128]                        # Output token lengths to test
```

## Output

Results saved as JSON in `output_dir`:

```
results/
├── my-benchmark_TP1_DP1_CTX512_C50_P100_O128.json
└── ...
```

Metrics: TTFT, TPOT, ITL, E2EL, throughput.
