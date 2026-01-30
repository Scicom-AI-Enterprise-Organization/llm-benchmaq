# SkyPilot Module

End-to-end SkyPilot GPU benchmarking: launch cluster, run benchmarks, tear down cluster.

## Prerequisites

Users must authenticate with SkyPilot before using this module:

```bash
# Option 1: Use sky auth
sky auth

# Option 2: Set environment variables for remote API server
export SKYPILOT_API_SERVER_URL="https://your-skypilot-api.example.com"
export SKYPILOT_API_KEY="your-api-key"
```

## CLI Usage

```bash
benchmaq sky bench --config config.yaml
# or short form:
benchmaq sky bench -c config.yaml
```

This will:
1. Launch a SkyPilot cluster on your configured cloud provider
2. Run benchmarks on the cluster
3. Tear down the cluster automatically

If you press `Ctrl+C`, the cluster will still be cleaned up.

## Python API

```python
import benchmaq

# SkyPilot end-to-end: launch -> benchmark -> cleanup
benchmaq.skypilot.bench.from_yaml("examples/skypilot_config.yaml")
```

## Configuration

The config file has two sections:
- `skypilot`: SkyPilot task configuration (passed directly to sky.Task)
- `benchmark`: Benchmark configuration (read by benchmaq on the remote cluster)

```yaml
# SkyPilot task configuration
skypilot:
  name: benchmaq-cluster
  workdir: .
  resources:
    accelerators: A100-80GB:2
    disk_size: 500
    any_of:
      - cloud: aws
      - cloud: gcp
      - cloud: runpod
  envs:
    HF_TOKEN: "your-hf-token"
  setup: |
    pip install "benchmaq[vllm]"
  run: |
    # $config is replaced with the --config path automatically
    benchmaq bench $config

# Benchmark configuration (read by benchmaq bench on the cluster)
benchmark:
  - name: tp2_dp1
    engine: vllm
    model:
      repo_id: "Qwen/Qwen2.5-7B-Instruct"
      local_dir: "/workspace/model"
    serve:
      tensor_parallel_size: 2
      max_model_len: 8192
    bench:
      - backend: vllm
        dataset_name: random
        random_input_len: 1024
        num_prompts: 100
    results:
      save_result: true
      result_dir: "./benchmark_results"
```

## How It Works

1. **Local**: `benchmaq sky bench --config config.yaml` reads the `skypilot:` section
2. **Substitution**: `$config` in the YAML is replaced with `config.yaml`
3. **SkyPilot**: Launches a cluster with the specified resources and runs the task
4. **Remote**: The `run:` command executes `benchmaq bench config.yaml`
5. **Remote**: `benchmaq bench` reads the `benchmark:` section and runs vLLM benchmarks
6. **Cleanup**: SkyPilot tears down the cluster after the job completes

## Supported Clouds

SkyPilot supports many cloud providers out of the box:
- AWS
- GCP
- Azure
- RunPod
- Lambda Labs
- And more...

See [SkyPilot documentation](https://docs.skypilot.co/) for the full list and configuration.

See [examples/](../../examples/) for more configuration examples.
