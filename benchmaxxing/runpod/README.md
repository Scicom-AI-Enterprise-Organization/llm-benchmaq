# Runpod CLI

Deploy and manage RunPod GPU pods.

## Usage

### Deploy a Pod

Create and start a new GPU pod on RunPod.

```bash
benchmaxxing runpod deploy config.yaml
```

Output:
```
Pod created: abc123xyz
✓ Done!
  SSH: ssh root@1.2.3.4 -p 12345 -i ~/.ssh/id_ed25519
```

### Get Pod Info

Retrieve pod status, IP address, and SSH connection details.

```bash
benchmaxxing runpod find config.yaml
```

Output:
```
Pod: my-pod (abc123xyz)
  Status: RUNNING
  SSH: ssh root@1.2.3.4 -p 12345 -i ~/.ssh/id_ed25519
```

### Start a Stopped Pod

Resume a previously stopped pod without losing data.

```bash
benchmaxxing runpod start config.yaml
```

### Delete a Pod

Terminate and remove a pod permanently.

```bash
benchmaxxing runpod delete config.yaml
```

## Configuration

```yaml
runpod:
  api_key: "your-api-key"                # RunPod API key
  ssh_key: "~/.ssh/id_ed25519"           # Path to SSH private key
  
  pod:
    name: "my-pod"                       # Pod name
    gpu_type: "NVIDIA H100 80GB HBM3"    # GPU type
    gpu_count: 2                         # Number of GPUs
    instance_type: on_demand             # on_demand or spot
    secure_cloud: true                   # Use secure cloud
  
  container:
    image: "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"  # Docker image
    disk_size: 200                       # Container disk size (GB)
  
  storage:
    volume_size: 200                     # Persistent volume size (GB)
    mount_path: "/workspace"             # Volume mount path
  
  ports:
    http: [8888, 8000]                   # HTTP ports to expose
    tcp: [22]                            # TCP ports to expose
  
  env:
    HF_HOME: "/workspace/hf_home"        # Environment variables
```

See [examples/](../../examples/) for more.
