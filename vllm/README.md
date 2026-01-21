# vLLM Benchmark Script

Simple benchmarking tool for vLLM models. Automatically starts/stops vLLM server for each parallelism configuration.

## Pre-requisites

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Quick Start

### 1. Create Config

Copy the template and edit:

```bash
cp config.template.yaml config.yaml
```

### 2. Configure Benchmark

Edit `config.yaml`:

```yaml
runs:
  - name: "my-model-run1"
    model_path: "/path/to/model"
    port: 8000
    output_dir: "./benchmark_results"
    tp_dp_pairs:
      - tp: 4
        dp: 2
        pp: 1
    context_size: [1024, 2048, 4096]
    concurrency: [100]
    num_prompts: [100]
    output_len: [128]
```

### 3. Run Benchmark

```bash
# Using default config.yaml
./run.sh

# Using custom config
./run.sh my-config.yaml
```

## Multiple Runs

To benchmark multiple models or configurations, add more entries under `runs`:

```yaml
runs:
  # First model
  - name: "model-a-benchmark"
    model_path: "/path/to/model-a"
    port: 8000
    output_dir: "./results/model-a"
    tp_dp_pairs:
      - tp: 8
        dp: 1
        pp: 1
    context_size: [1024, 2048]
    concurrency: [100]
    num_prompts: [100]
    output_len: [128]

  # Second model
  - name: "model-b-benchmark"
    model_path: "/path/to/model-b"
    port: 8000
    output_dir: "./results/model-b"
    tp_dp_pairs:
      - tp: 4
        dp: 2
        pp: 1
      - tp: 2
        dp: 4
        pp: 1
    context_size: [1024, 4096, 8192]
    concurrency: [50, 100]
    num_prompts: [100]
    output_len: [128, 256]

  # Third model with different settings
  - name: "model-c-benchmark"
    model_path: "/path/to/model-c"
    port: 8001
    output_dir: "./results/model-c"
    gpu_memory_utilization: 0.85
    tp_dp_pairs:
      - tp: 8
        dp: 1
        pp: 1
    context_size: [2048]
    concurrency: [100]
    num_prompts: [200]
    output_len: [512]
```

Each run will:
1. Start vLLM server with the specified parallelism (TP/DP/PP)
2. Run all benchmark combinations (context_size × concurrency × num_prompts × output_len)
3. Stop the server and move to the next configuration

## Examples

See `examples/` folder for sample configurations.
