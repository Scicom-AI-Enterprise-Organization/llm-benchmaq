"""RunPod client using runpodctl CLI for pod management."""

import json
import os
import subprocess
import time
from typing import Optional


def _run_runpodctl(*args, timeout=120) -> subprocess.CompletedProcess:
    """Run a runpodctl command and return the result."""
    cmd = ["runpodctl"] + list(args)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _parse_json(output: str) -> dict:
    """Parse JSON from runpodctl output, handling potential preamble text."""
    text = output.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Find first { or [ in the output
        for i, ch in enumerate(text):
            if ch in ("{", "["):
                try:
                    return json.loads(text[i:])
                except json.JSONDecodeError:
                    continue
        raise ValueError(f"No JSON found in output: {text[:200]}")


def set_api_key(key: str):
    """Set RunPod API key via environment variable for runpodctl.

    Also writes to runpodctl config so the key persists for subprocesses.
    """
    os.environ["RUNPOD_API_KEY"] = key
    try:
        _run_runpodctl("config", "--apiKey", key, timeout=15)
    except Exception:
        pass  # env var is sufficient


def get_api_key() -> Optional[str]:
    """Get RunPod API key from environment."""
    return os.environ.get("RUNPOD_API_KEY")


def deploy(
    gpu_type: str,
    gpu_count: int,
    image: str,
    disk_size: int,
    container_disk_size: int = 20,
    volume_mount_path: str = "/workspace",
    secure_cloud: bool = True,
    spot: bool = True,
    bid_per_gpu: Optional[float] = None,
    env: Optional[dict] = None,
    name: Optional[str] = None,
    ports: Optional[str] = None,
    ssh_key_path: Optional[str] = None,
    wait_for_ready: bool = True,
    **kwargs,
) -> dict:
    """Deploy a RunPod GPU pod using runpodctl CLI."""
    if env is None:
        env = {}

    if ports is None:
        ports = "8888/http,8000/http,22/tcp"
    elif isinstance(ports, list):
        ports = ",".join(ports)

    if name is None:
        name = f"{gpu_type}_{gpu_count}".replace(" ", "_")

    if spot:
        print("Note: runpodctl creates on-demand pods. Spot/bid config is ignored.")

    cloud_type = "SECURE" if secure_cloud else "COMMUNITY"

    cmd_args = [
        "pod", "create",
        "--image", image,
        "--gpu-id", gpu_type,
        "--gpu-count", str(gpu_count),
        "--container-disk-in-gb", str(container_disk_size),
        "--volume-in-gb", str(disk_size),
        "--volume-mount-path", volume_mount_path,
        "--ports", ports,
        "--cloud-type", cloud_type,
        "--name", name,
        "--ssh",
    ]

    if env:
        cmd_args.extend(["--env", json.dumps(env)])

    print(f"Creating pod: {name}")
    print(f"  GPU: {gpu_type} x{gpu_count}")
    print(f"  Image: {image}")
    print(f"  Storage: {disk_size}GB volume, {container_disk_size}GB container disk")

    result = _run_runpodctl(*cmd_args, timeout=120)

    if result.returncode != 0:
        error = result.stderr.strip() or result.stdout.strip()
        raise Exception(f"Failed to create pod: {error}")

    try:
        pod_data = _parse_json(result.stdout)
    except ValueError:
        raise Exception(f"Could not parse pod creation output: {result.stdout.strip()[:300]}")

    pod_id = pod_data.get("id")
    if not pod_id:
        raise Exception(f"No pod ID in creation response: {result.stdout.strip()[:300]}")

    print(f"Pod created: {pod_id}")

    instance = {
        "id": pod_id,
        "name": name,
        "url": f"https://{pod_id}-8000.proxy.runpod.net",
    }

    if wait_for_ready:
        ssh_info = _wait_for_ssh(pod_id, ssh_key_path)
        if ssh_info:
            instance["ssh"] = ssh_info
            print(f"SSH ready: {ssh_info['command']}")
        else:
            raise Exception(f"Pod {name} failed to become SSH accessible")

    return instance


def _wait_for_ssh(pod_id: str, ssh_key_path: Optional[str] = None, timeout: int = 600) -> Optional[dict]:
    """Wait for pod SSH to be ready using runpodctl."""
    print("Waiting for pod to be ready...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            # Check pod status via runpodctl pod get
            result = _run_runpodctl("pod", "get", pod_id, timeout=30)
            if result.returncode != 0:
                print(f"  Waiting for pod to initialize...")
                time.sleep(10)
                continue

            try:
                pod_data = _parse_json(result.stdout)
            except ValueError:
                time.sleep(10)
                continue

            status = pod_data.get("desiredStatus") or pod_data.get("status", "")
            if status != "RUNNING":
                print(f"  Pod status: {status}")
                time.sleep(10)
                continue

            # Pod is RUNNING - get SSH info via runpodctl ssh info
            ssh_result = _run_runpodctl("ssh", "info", pod_id, timeout=30)
            if ssh_result.returncode != 0:
                print(f"  SSH info not available yet...")
                time.sleep(10)
                continue

            ip = None
            port = None

            try:
                ssh_data = _parse_json(ssh_result.stdout)
                ip = ssh_data.get("host") or ssh_data.get("ip")
                port = ssh_data.get("port")
            except ValueError:
                pass

            # Fallback: try extracting from pod runtime ports
            if not ip or not port:
                runtime = pod_data.get("runtime", {})
                if runtime:
                    for p in runtime.get("ports", []):
                        if p.get("privatePort") == 22:
                            ip = p.get("ip")
                            port = p.get("publicPort")
                            break

            if ip and port:
                key_path = ssh_key_path or "~/.runpod/ssh/RunPod-Key-Go"
                ssh_info = {
                    "ip": ip,
                    "port": int(port),
                    "command": f"ssh root@{ip} -p {port} -i {key_path}",
                }
                if _check_ssh(ip, int(port), ssh_key_path):
                    return ssh_info
                print(f"  SSH not ready yet, retrying...")

            time.sleep(10)
        except subprocess.TimeoutExpired:
            print(f"  Command timed out, retrying...")
            time.sleep(10)
        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(10)

    return None


def _check_ssh(ip: str, port: int, ssh_key_path: Optional[str] = None) -> bool:
    """Test SSH connection."""
    key_path = os.path.expanduser(ssh_key_path or "~/.runpod/ssh/RunPod-Key-Go")
    try:
        cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=10",
            "-o", "BatchMode=yes",
            "-i", key_path,
            "-p", str(port),
            f"root@{ip}",
            "echo ok",
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=15)
        return result.returncode == 0
    except Exception:
        return False


def delete(pod_id: Optional[str] = None, name: Optional[str] = None) -> dict:
    """Delete a RunPod pod using runpodctl."""
    if name and not pod_id:
        # Find pod by name using runpodctl pod list
        result = _run_runpodctl("pod", "list", "--name", name, timeout=30)
        if result.returncode == 0:
            try:
                pods = _parse_json(result.stdout)
                if isinstance(pods, list) and pods:
                    pod_id = pods[0].get("id")
                elif isinstance(pods, dict):
                    # Might be wrapped in a key
                    pod_list = pods.get("pods", pods.get("data", []))
                    if isinstance(pod_list, list) and pod_list:
                        pod_id = pod_list[0].get("id")
            except ValueError:
                pass
        if not pod_id:
            raise Exception(f"Pod with name '{name}' not found")

    if not pod_id:
        raise Exception("Either pod_id or name is required")

    result = _run_runpodctl("pod", "delete", pod_id, timeout=30)
    if result.returncode != 0:
        error = result.stderr.strip() or result.stdout.strip()
        raise Exception(f"Failed to delete pod {pod_id}: {error}")

    print(f"Pod {pod_id} deleted")
    return {"status": "deleted", "id": pod_id, "name": name}
