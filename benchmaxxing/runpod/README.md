# RunPod CLI

Deploy and manage RunPod GPU pods.

## Deploy

```bash
benchmaxxing runpod deploy config.yaml
```

## Delete

```bash
benchmaxxing runpod delete config.yaml
```

## Find

```bash
benchmaxxing runpod find config.yaml
```

## Start

```bash
benchmaxxing runpod start config.yaml
```

## Config Format

```yaml
api_key: "your-runpod-api-key"
ssh_key: "~/.ssh/id_ed25519"

pod:
  name: "my-pod"
  gpu_type: "NVIDIA H100 80GB HBM3"
  gpu_count: 2
  instance_type: spot

container:
  image: "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
  disk_size: 20

storage:
  volume_size: 200
  mount_path: "/workspace"

ports:
  http: [8888, 8000]
  tcp: [22]

env:
  HF_HOME: "/workspace/hf_home"
```

## Example Configs

| File | GPUs | Volume |
|------|------|--------|
| `examples/2x_h100_sxm.yaml` | 2x H100 | 200GB |
| `examples/4x_h100_sxm.yaml` | 4x H100 | 300GB |
| `examples/8x_h100_sxm.yaml` | 8x H100 | 500GB |
