# LLM-Benchmaq

Seamless scripts for LLM performance benchmarking, written in Northern Malaysia slang, with end 'k' sounds replaced by a 'q' sound.

## Features

1. Seamless remote benchmarking over SSH, automatic venv/setup, upload model, install dependencies, start server and run benchmarks on a remote GPU host, special thanks to [Scicom-AI-Enterprise-Organization/pyremote](https://github.com/Scicom-AI-Enterprise-Organization/pyremote)
2. End-to-end RunPod integration, deploy, bench and cleanup RunPod instances from CLI or Python API; supports API key, ports and SSH access.
3. CLI and Python API, `benchmaq` CLI for quick runs and a programmatic `benchmaq.bench(...)` Python API for automation.
4. Multi-engine architecture, vLLM supported today; additional engines (e.g., TensorRT-LLM, SGLang) planned for future releases.
5. Flexible YAML config format with examples, single-run and multi-run configs, run-level overrides, remote and runpod sections.
6. Parameter sweeps and combinatorial runs, sweep tensor/pipeline/data parallelism (TP/PP/DP), context sizes, concurrency, number of prompts, output lengths, etc.
7. Serve-mode benchmarking, benchmark against a running inference server (host/port/endpoint) instead of starting a server each run.
8. Detailed metrics and structured outputs, we use `vllm bench serve` to generate metrics include TTFT, TPOT, ITL, E2EL and throughput. Results saved as JSON.
9. Environment & dependency management, uses uv for virtualenv management; can install specified dependencies locally or on the remote host.
10. Authentication & model access, SSH password/key support for remote hosts and HuggingFace token support for gated models.
11. RunPod management utilities, `benchmaq runpod` CLI and Python client to deploy, find, start, stop and delete pods; list and query pods programmatically.
12. Advanced runtime tuning, control dtype, GPU memory utilization, max model length/num sequences, disable logging, enable expert parallel, and set parallelism pairs.

## Supported Engines

- [x] [vLLM](./benchmaq/vllm/) - vLLM inference server
- [ ] TensorRT-LLM - *(coming soon)*
- [ ] SGLang - *(coming soon)*

## Installation

Easily using UV,

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv --python 3.11
source .venv/bin/activate

uv pip install "benchmaq @ git+https://github.com/Scicom-AI-Enterprise-Organization/llm-benchmaq.git"
```

## Usage

### 1. Benchmark locally (GPU Server)

```bash
# Install with vllm
uv pip install "benchmaq[vllm] @ git+https://github.com/Scicom-AI-Enterprise-Organization/llm-benchmaq.git"

# single run
benchmaq bench examples/1_run_single.yaml

# multiple run
benchmaq bench examples/2_run_multiple.yaml
```

### 2. Benchmark Remotely via SSH

```bash
benchmaq bench examples/3_remote_gpu_ssh_password.yaml
```

### 3. Benchmark Remotely on Runpod

#### Deploy RunPod Instance

```bash
benchmaq runpod deploy examples/4_remote_gpu_runpod.yaml
```

Output:
```
Pod created: abc123xyz
✓ Done!
  SSH: ssh root@1.2.3.4 -p 12345 -i ~/.ssh/id_ed25519
```

#### Run Benchmarks

Copy the SSH info to your config's `remote` section, then:

```bash
benchmaq bench examples/4_remote_gpu_runpod.yaml
```

#### Delete RunPod Instance

```bash
benchmaq runpod delete examples/4_remote_gpu_runpod.yaml
```

## Python API

```python
import benchmaq

# Run benchmark (local or remote SSH)
benchmaq.bench("examples/1_run_single.yaml")
benchmaq.bench("examples/3_remote_gpu_ssh_password.yaml")

# Runpod end-to-end: deploy -> benchmark -> cleanup
benchmaq.runpod.bench("examples/5_config_runpod.yaml")

# Runpod deploy / delete
benchmaq.runpod.deploy("examples/4_remote_gpu_runpod.yaml")
benchmaq.runpod.delete("examples/4_remote_gpu_runpod.yaml")
```

### Multiprocessing (Parallel Benchmarks)

```python
from multiprocessing import Pool

# run_benchmark is a module-level function that supports multiprocessing
from benchmaq.runpod import run_benchmark

configs = [
    "examples/5_config_runpod_multiprocess_1.yaml",
    "examples/5_config_runpod_multiprocess_2.yaml",
]

with Pool(processes=len(configs)) as pool:
    results = pool.map(run_benchmark, configs)
```

## Config Format

See [examples/](./examples/) for more config samples.

## Unit & Integration Test

```
uv run python -m pytest tests/ -v -s
```
