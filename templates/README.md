# Provider templates

Shared, reusable provider blocks so a benchmark config only has to declare what
makes it unique (`model:`, `serve:`, `bench:`) and pull the provider in by
reference instead of pasting it.

## How it works

Add an `extends:` key to a config pointing at one or more template files. The
templates are merged in first, then the config's own top-level keys override
them (**shallow** merge — a restated top-level block replaces the template's
wholesale). The `extends:` key supports a single path or a list; paths are
relative to the config file. Templates may themselves `extends:` other
templates. Implemented in `benchmaq/config.py::load_config`.

```yaml
# 01_Qwen3.6-27B/voice.yaml
extends: ../../templates/runpod_h200.yaml
benchmark:
  - name: qwen3.6-27b-voice
    engine: vllm
    model: { repo_id: "Qwen/Qwen3.6-27B" }
    serve: { tensor_parallel_size: 2, data_parallel_size: 4 }
    bench:
      - { endpoint: /v1/completions, dataset_name: random, random_input_len: 8192, random_output_len: 1024, num_prompts: 200, max_concurrency: 100 }
    results: { save_result: true, result_dir: ./results }
```

## The three providers

| Template            | What it does                                              | Needs                       |
|---------------------|----------------------------------------------------------|-----------------------------|
| `ingress.yaml`      | Bench an already-served, ingressed vLLM (URL only)       | `vllm` on the local box     |
| `runpod_h200.yaml`  | Provision pod → serve → bench → tear down                | `RUNPOD_API_KEY`, SSH key   |
| `vm.yaml`           | SSH into your own GPU box → serve → bench (box stays up)  | host + SSH key              |

## Overriding

Merge is shallow at the top level, so to tweak one nested value restate its
whole block in the child config:

```yaml
extends: ../../templates/runpod_h200.yaml
runpod:                 # replaces the template's runpod: entirely
  ssh_private_key: "~/.runpod/ssh/RunPod-Key-Go"
  pod: { name: "benchmaq-h200-4x", gpu_type: "NVIDIA H200", gpu_count: 4, instance_type: on_demand, secure_cloud: true }
  container: { image: "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404", disk_size: 700 }
  storage: { volume_size: 700, mount_path: /workspace }
  ports: { http: [8000], tcp: [22] }
  env: { HF_HOME: /workspace/hf_home }
```

Compose multiple templates with a list (later wins, then the config wins):

```yaml
extends:
  - ../../templates/vm.yaml          # provider
  - ../../templates/bench_grid.yaml  # a shared bench: sweep, if you make one
```
