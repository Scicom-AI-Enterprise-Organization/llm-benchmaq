"""
RunPod cloud GPU module.

Deploy pods, run benchmarks, and manage RunPod infrastructure.

Usage:
    import benchmaxxing
    
    # End-to-end: deploy -> bench -> delete
    result = benchmaxxing.runpod.bench(
        api_key="rpa_xxx",
        gpu_type="NVIDIA A100 80GB PCIe",
        gpu_count=2,
        model_path="meta-llama/Llama-3.1-8B-Instruct",
        tensor_parallel=2,
    )
    
    # Individual pod operations
    pod = benchmaxxing.runpod.deploy(api_key="rpa_xxx", gpu_type="...", ...)
    benchmaxxing.runpod.delete(api_key="rpa_xxx", pod_id="abc123")
"""

from typing import Optional, List, Dict, Any
import os

from .core.client import (
    deploy as _deploy,
    delete as _delete,
    find as _find,
    find_by_name as _find_by_name,
    start as _start,
    stop as _stop,
    list_pods as _list_pods,
    set_api_key,
    get_api_key,
    wait_for_pod,
    get_ssh_info,
    check_ssh,
)


def _ensure_api_key(api_key: Optional[str] = None) -> str:
    """Ensure API key is set, either from parameter or environment."""
    if api_key:
        set_api_key(api_key)
        return api_key
    
    existing = get_api_key()
    if existing:
        return existing
    
    env_key = os.environ.get("RUNPOD_API_KEY")
    if env_key:
        set_api_key(env_key)
        return env_key
    
    raise ValueError(
        "RunPod API key required. Provide api_key parameter or set RUNPOD_API_KEY environment variable."
    )


def deploy(
    api_key: Optional[str] = None,
    *,
    gpu_type: str,
    gpu_count: int = 1,
    image: str = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
    disk_size: int = 100,
    container_disk_size: int = 20,
    volume_mount_path: str = "/workspace",
    secure_cloud: bool = True,
    spot: bool = True,
    bid_per_gpu: Optional[float] = None,
    env: Optional[Dict[str, str]] = None,
    name: Optional[str] = None,
    ports: Optional[str] = None,
    ssh_key_path: Optional[str] = None,
    wait_for_ready: bool = True,
    health_check_retries: int = 60,
    health_check_interval: float = 10.0,
    **kwargs,
) -> Dict[str, Any]:
    """
    Deploy a RunPod GPU pod.
    
    Args:
        api_key: RunPod API key (or set RUNPOD_API_KEY env var)
        gpu_type: GPU type (e.g., "NVIDIA A100 80GB PCIe")
        gpu_count: Number of GPUs
        image: Docker image
        disk_size: Volume disk size in GB
        container_disk_size: Container disk size in GB
        volume_mount_path: Volume mount path
        secure_cloud: Use secure cloud (recommended)
        spot: Use spot instance (cheaper but can be interrupted)
        bid_per_gpu: Bid price per GPU for spot instances
        env: Environment variables
        name: Pod name
        ports: Port mappings (e.g., "8000/http,22/tcp")
        ssh_key_path: Path to SSH private key for connection
        wait_for_ready: Wait for pod to be ready before returning
        health_check_retries: Number of health check retries
        health_check_interval: Seconds between health checks
        
    Returns:
        Dict with pod info including id, name, url, and ssh connection details
    """
    _ensure_api_key(api_key)
    
    return _deploy(
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        image=image,
        disk_size=disk_size,
        container_disk_size=container_disk_size,
        volume_mount_path=volume_mount_path,
        secure_cloud=secure_cloud,
        spot=spot,
        bid_per_gpu=bid_per_gpu,
        env=env,
        name=name,
        ports=ports,
        ssh_key_path=ssh_key_path,
        wait_for_ready=wait_for_ready,
        health_check_retries=health_check_retries,
        health_check_interval=health_check_interval,
        **kwargs,
    )


def delete(
    api_key: Optional[str] = None,
    *,
    pod_id: Optional[str] = None,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Delete a RunPod pod.
    
    Args:
        api_key: RunPod API key
        pod_id: Pod ID to delete
        name: Pod name to delete (alternative to pod_id)
        
    Returns:
        Dict with deletion status
    """
    _ensure_api_key(api_key)
    return _delete(pod_id=pod_id, name=name)


def find(
    api_key: Optional[str] = None,
    *,
    pod_id: str,
) -> Dict[str, Any]:
    """
    Find a pod by ID.
    
    Args:
        api_key: RunPod API key
        pod_id: Pod ID to find
        
    Returns:
        Dict with pod info
    """
    _ensure_api_key(api_key)
    return _find(pod_id)


def find_by_name(
    api_key: Optional[str] = None,
    *,
    name: str,
) -> Optional[Dict[str, Any]]:
    """
    Find a pod by name.
    
    Args:
        api_key: RunPod API key
        name: Pod name to find
        
    Returns:
        Dict with pod info or None if not found
    """
    _ensure_api_key(api_key)
    return _find_by_name(name)


def start(
    api_key: Optional[str] = None,
    *,
    pod_id: str,
) -> Dict[str, Any]:
    """
    Start a stopped pod.
    
    Args:
        api_key: RunPod API key
        pod_id: Pod ID to start
        
    Returns:
        Dict with start result
    """
    _ensure_api_key(api_key)
    return _start(pod_id)


def stop(
    api_key: Optional[str] = None,
    *,
    pod_id: str,
) -> Dict[str, Any]:
    """
    Stop a running pod.
    
    Args:
        api_key: RunPod API key
        pod_id: Pod ID to stop
        
    Returns:
        Dict with stop result
    """
    _ensure_api_key(api_key)
    return _stop(pod_id)


def list_pods(
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    List all pods.
    
    Args:
        api_key: RunPod API key
        
    Returns:
        List of pod info dicts
    """
    _ensure_api_key(api_key)
    return _list_pods()


def bench(
    config_path: Optional[str] = None,
    *,
    api_key: Optional[str] = None,
    # Pod configuration
    gpu_type: Optional[str] = None,
    gpu_count: Optional[int] = None,  # None means use config file value
    image: str = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
    disk_size: int = 100,
    container_disk_size: int = 20,
    mount_path: str = "/workspace",
    secure_cloud: bool = True,
    spot: bool = True,
    instance_type: Optional[str] = None,  # "spot" or "on-demand"
    bid_per_gpu: Optional[float] = None,
    pod_name: Optional[str] = None,
    ports_http: Optional[List[int]] = None,
    ports_tcp: Optional[List[int]] = None,
    env: Optional[Dict[str, str]] = None,
    ssh_private_key: Optional[str] = None,
    # Remote UV config
    uv_path: str = "~/.benchmark-venv",
    python_version: str = "3.11",
    dependencies: Optional[List[str]] = None,
    # Benchmark options
    name: str = "benchmark",
    model_path: Optional[str] = None,
    hf_token: Optional[str] = None,
    tensor_parallel: int = 1,
    data_parallel: int = 1,
    pipeline_parallel: int = 1,
    parallelism_pairs: Optional[List[Dict[str, int]]] = None,
    gpu_memory_utilization: float = 0.9,
    max_model_len: Optional[int] = None,
    max_num_seqs: Optional[int] = None,
    dtype: Optional[str] = None,
    disable_log_requests: bool = False,
    enable_expert_parallel: bool = False,
    context_sizes: Optional[List[int]] = None,
    concurrency: Optional[List[int]] = None,
    num_prompts: Optional[List[int]] = None,
    output_len: Optional[List[int]] = None,
    output_dir: str = "./benchmark_results",
    save_results: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    End-to-end RunPod benchmarking: deploy pod -> run benchmarks -> delete pod.
    
    Can be called with a config file path, kwargs, or both (kwargs override config).
    
    Args:
        config_path: Path to YAML config file (optional)
        api_key: RunPod API key (or set RUNPOD_API_KEY env var)
        gpu_type: GPU type (e.g., "NVIDIA A100 80GB PCIe")
        gpu_count: Number of GPUs
        image: Docker image
        disk_size: Volume disk size in GB
        container_disk_size: Container disk size in GB
        mount_path: Volume mount path
        secure_cloud: Use secure cloud
        spot: Use spot instance
        instance_type: "spot" or "on-demand" (overrides spot param)
        bid_per_gpu: Bid price per GPU for spot instances
        pod_name: Pod name
        ports_http: HTTP ports to expose
        ports_tcp: TCP ports to expose
        env: Environment variables
        ssh_private_key: Path to SSH private key
        uv_path: Path for UV virtual environment on remote
        python_version: Python version for UV environment
        dependencies: List of pip packages to install
        name: Benchmark run name
        model_path: Path or HuggingFace model ID
        hf_token: HuggingFace token for gated models
        tensor_parallel: Tensor parallel size
        data_parallel: Data parallel size
        pipeline_parallel: Pipeline parallel size
        parallelism_pairs: List of parallelism configs to try
        gpu_memory_utilization: GPU memory fraction to use
        max_model_len: Maximum model context length
        max_num_seqs: Maximum concurrent sequences
        dtype: Model dtype
        disable_log_requests: Disable vLLM request logging
        enable_expert_parallel: Enable expert parallelism for MoE
        context_sizes: List of input context sizes to benchmark
        concurrency: List of concurrency levels to benchmark
        num_prompts: List of prompt counts to benchmark
        output_len: List of output lengths to benchmark
        output_dir: Directory for benchmark results
        save_results: Whether to save results to files
        
    Returns:
        Dict with benchmark results and status
    """
    from ..config import load_config, merge_config, kwargs_to_runpod_config
    from ..runner import run_e2e as _run_e2e
    
    # Build config from file and/or kwargs
    config = {}
    
    if config_path:
        config = load_config(config_path)
    
    # Apply direct overrides to config (these always take precedence)
    # Override runpod.pod settings
    if "runpod" not in config:
        config["runpod"] = {}
    if "pod" not in config["runpod"]:
        config["runpod"]["pod"] = {}
    
    # Apply gpu_count override
    if gpu_count is not None:
        config["runpod"]["pod"]["gpu_count"] = gpu_count
    
    # Apply gpu_type override
    if gpu_type:
        config["runpod"]["pod"]["gpu_type"] = gpu_type
    
    # Apply other runpod overrides if provided
    if api_key:
        config["runpod"]["runpod_api_key"] = api_key
    if ssh_private_key:
        config["runpod"]["ssh_private_key"] = ssh_private_key
    if pod_name:
        config["runpod"]["pod"]["name"] = pod_name
    if instance_type:
        config["runpod"]["pod"]["instance_type"] = instance_type
    
    # Apply benchmark overrides to runs
    if config.get("runs"):
        for run_cfg in config["runs"]:
            if "benchmark" not in run_cfg:
                run_cfg["benchmark"] = {}
            
            # Override context_sizes
            if context_sizes is not None:
                run_cfg["benchmark"]["context_size"] = context_sizes
            
            # Override concurrency
            if concurrency is not None:
                run_cfg["benchmark"]["concurrency"] = concurrency
            
            # Override num_prompts
            if num_prompts is not None:
                run_cfg["benchmark"]["num_prompts"] = num_prompts
            
            # Override output_len
            if output_len is not None:
                run_cfg["benchmark"]["output_len"] = output_len
            
            # Override parallelism_pairs
            if parallelism_pairs is not None:
                if "vllm_serve" not in run_cfg:
                    run_cfg["vllm_serve"] = {}
                run_cfg["vllm_serve"]["parallelism_pairs"] = parallelism_pairs
            
            # Override model_path
            if model_path:
                if "vllm_serve" not in run_cfg:
                    run_cfg["vllm_serve"] = {}
                run_cfg["vllm_serve"]["model_path"] = model_path
    
    # If no config file and building from scratch, use kwargs_to_runpod_config
    elif gpu_type or model_path:
        use_spot = spot
        if instance_type:
            use_spot = instance_type == "spot"
        
        kwargs_config = kwargs_to_runpod_config(
            api_key=api_key,
            gpu_type=gpu_type or "",
            gpu_count=gpu_count if gpu_count is not None else 1,
            image=image,
            disk_size=disk_size,
            container_disk_size=container_disk_size,
            mount_path=mount_path,
            secure_cloud=secure_cloud,
            instance_type="spot" if use_spot else "on-demand",
            bid_per_gpu=bid_per_gpu,
            name=pod_name,
            ports_http=ports_http or [8888, 8000],
            ports_tcp=ports_tcp or [22],
            env=env or {},
            ssh_private_key=ssh_private_key,
            uv_path=uv_path,
            python_version=python_version,
            dependencies=dependencies or ["pyyaml", "requests", "vllm==0.11.0", "huggingface_hub"],
            model_path=model_path or "",
            hf_token=hf_token,
            tensor_parallel=tensor_parallel,
            data_parallel=data_parallel,
            pipeline_parallel=pipeline_parallel,
            parallelism_pairs=parallelism_pairs,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            dtype=dtype,
            disable_log_requests=disable_log_requests,
            enable_expert_parallel=enable_expert_parallel,
            context_sizes=context_sizes or [1024],
            concurrency=concurrency or [50],
            num_prompts=num_prompts or [100],
            output_len=output_len or [128],
            output_dir=output_dir,
            save_results=save_results,
        )
        config = merge_config(config, kwargs_config)
    
    # Ensure API key is set
    runpod_cfg = config.get("runpod", {})
    final_api_key = api_key or runpod_cfg.get("runpod_api_key") or os.environ.get("RUNPOD_API_KEY")
    if final_api_key:
        set_api_key(final_api_key)
        if "runpod" not in config:
            config["runpod"] = {}
        config["runpod"]["runpod_api_key"] = final_api_key
    
    # Validate config
    if not config.get("runpod", {}).get("pod", {}).get("gpu_type") and not gpu_type:
        raise ValueError("gpu_type is required. Provide gpu_type parameter or config file with runpod.pod.gpu_type.")
    
    if not config.get("runs"):
        raise ValueError("No benchmark runs defined. Provide model_path parameter or config file with 'runs' section.")
    
    # Run end-to-end benchmark
    try:
        _run_e2e(config)
        return {
            "status": "success",
            "gpu_type": config.get("runpod", {}).get("pod", {}).get("gpu_type"),
            "gpu_count": config.get("runpod", {}).get("pod", {}).get("gpu_count"),
            "runs": len(config.get("runs", [])),
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


__all__ = [
    "deploy",
    "delete",
    "find",
    "find_by_name",
    "start",
    "stop",
    "list_pods",
    "bench",
]
