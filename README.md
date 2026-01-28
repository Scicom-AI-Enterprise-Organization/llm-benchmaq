# LLM-Benchmaq

Seamless scripts for LLM performance benchmarking, written in Northern Malaysia slang, with end 'k' sounds replaced by a 'q' sound.

## Features

- Seamless remote benchmarking over SSH
  - Automatic venv/setup, upload model, install dependencies, start server and run benchmarks on a remote GPU host.

- End-to-end RunPod integration
  - Deploy, bench and cleanup RunPod instances from CLI or Python API; supports API key, ports and SSH access.

- CLI and Python API
  - `benchmaq` CLI for quick runs and a programmatic `benchmaq.bench(...)` Python API for automation.

- Multi-engine architecture
  - vLLM supported today; additional engines (e.g., TensorRT-LLM, SGLang) planned for future releases.

- Flexible YAML config format with examples
  - Single-run and multi-run configs, run-level overrides, remote and runpod sections.

- Parameter sweeps and combinatorial runs
  - Sweep tensor/pipeline/data parallelism (TP/PP/DP), context sizes, concurrency, number of prompts, output lengths, etc.

- Serve-mode benchmarking
  - Benchmark against a running inference server (host/port/endpoint) instead of starting a server each run.

- Detailed metrics and structured outputs
  - Metrics include TTFT, TPOT, ITL, E2EL and throughput. Results saved as JSON (and optionally text) with descriptive filenames in configurable output_dir.

- Multiprocessing support
  - Module-level entrypoints support Python multiprocessing for parallel benchmark execution.

- Environment & dependency management
  - Uses uv for virtualenv management; can install specified dependencies locally or on the remote host.

- Authentication & model access
  - SSH password/key support for remote hosts and Hugging Face token support for gated models.

- RunPod management utilities
  - `benchmaq runpod` CLI and Python client to deploy, find, start, stop and delete pods; list and query pods programmatically.

- Advanced runtime tuning
  - Control dtype, GPU memory utilization, max model length/num sequences, disable logging, enable expert parallel, and set parallelism pairs.

- Extensible & reproducible
  - Modular engine-specific code (benchmaq.vllm), reproducible output naming conventions and configurable save_results/output_dir options.

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
