"""
Benchmaxxing - Seamless scripts for LLM performance benchmarking.

Usage:
    import benchmaxxing
    
    # Local benchmark
    benchmaxxing.bench(
        model_path="meta-llama/Llama-3.1-8B-Instruct",
        tensor_parallel=2,
        context_sizes=[1024, 2048],
    )
    
    # Remote SSH benchmark (if host is provided)
    benchmaxxing.bench(
        host="gpu-server.example.com",
        key_filename="~/.ssh/id_ed25519",
        model_path="meta-llama/Llama-3.1-8B-Instruct",
        tensor_parallel=2,
    )
    
    # RunPod end-to-end (deploy -> bench -> delete)
    benchmaxxing.runpod.bench(
        api_key="rpa_xxx",
        gpu_type="NVIDIA A100 80GB PCIe",
        model_path="meta-llama/Llama-3.1-8B-Instruct",
    )
    
    # RunPod individual operations
    pod = benchmaxxing.runpod.deploy(api_key="rpa_xxx", gpu_type="...", ...)
    benchmaxxing.runpod.delete(api_key="rpa_xxx", pod_id="abc123")
    
    # vLLM low-level access
    with benchmaxxing.vllm.VLLMServer(...) as server:
        benchmaxxing.vllm.run_benchmark(...)
"""

__version__ = "0.3.0"

from typing import Optional, List, Dict, Any

from . import vllm
from . import runpod
from .config import (
    BenchConfig,
    BenchmarkResult,
    load_config,
    merge_config,
    kwargs_to_run_config,
    kwargs_to_remote_config,
)


def bench(
    config_path: Optional[str] = None,
    *,
    # Remote connection (if provided, runs remote benchmark)
    host: Optional[str] = None,
    port: int = 22,
    username: str = "root",
    password: Optional[str] = None,
    key_filename: Optional[str] = None,
    # UV environment (for remote)
    uv_path: str = "~/.benchmark-venv",
    python_version: str = "3.11",
    dependencies: Optional[List[str]] = None,
    # Benchmark options
    name: str = "benchmark",
    model_path: Optional[str] = None,
    hf_token: Optional[str] = None,
    server_port: int = 8000,
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
    Run benchmarks locally or remotely.
    
    If `host` is provided, runs on remote GPU server via SSH.
    Otherwise, runs locally.
    
    Can be called with a config file path, kwargs, or both (kwargs override config).
    
    Args:
        config_path: Path to YAML config file (optional)
        
        # Remote connection (if provided, runs remote benchmark)
        host: Remote server hostname or IP (if provided, runs remote)
        port: SSH port (default: 22)
        username: SSH username (default: root)
        password: SSH password (optional)
        key_filename: Path to SSH private key
        uv_path: Path for UV virtual environment on remote
        python_version: Python version for UV environment
        dependencies: List of pip packages to install on remote
        
        # Benchmark options
        name: Benchmark run name
        model_path: Path or HuggingFace model ID
        hf_token: HuggingFace token for gated models
        server_port: vLLM server port (default: 8000)
        tensor_parallel: Tensor parallel size
        data_parallel: Data parallel size
        pipeline_parallel: Pipeline parallel size
        parallelism_pairs: List of parallelism configs to try
        gpu_memory_utilization: GPU memory fraction to use
        max_model_len: Maximum model context length
        max_num_seqs: Maximum concurrent sequences
        dtype: Model dtype (auto, float16, bfloat16)
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
        
    Examples:
        # Local benchmark
        benchmaxxing.bench(
            model_path="meta-llama/Llama-3.1-8B-Instruct",
            tensor_parallel=2,
            context_sizes=[1024, 2048],
            concurrency=[50, 100],
        )
        
        # Remote benchmark
        benchmaxxing.bench(
            host="gpu-server.example.com",
            key_filename="~/.ssh/id_ed25519",
            model_path="meta-llama/Llama-3.1-8B-Instruct",
            tensor_parallel=4,
        )
        
        # From config file
        benchmaxxing.bench("config.yaml")
        
        # Config file + overrides
        benchmaxxing.bench("config.yaml", tensor_parallel=4, context_sizes=[2048])
    """
    from .runner import run as _run, run_remote as _run_remote
    
    # Build config from file and/or kwargs
    config = {}
    
    if config_path:
        config = load_config(config_path)
    
    # Check if this is a remote run (host in kwargs or config)
    is_remote = host is not None or config.get("remote", {}).get("host")
    
    # Build kwargs config
    kwargs_config = {}
    
    # Remote config from kwargs
    if host:
        kwargs_config["remote"] = kwargs_to_remote_config(
            host=host,
            port=port,
            username=username,
            password=password,
            key_filename=key_filename,
            uv_path=uv_path,
            python_version=python_version,
            dependencies=dependencies or ["pyyaml", "requests", "vllm==0.11.0", "huggingface_hub"],
        )
    
    # Run config from kwargs
    if model_path:
        kwargs_config.update(kwargs_to_run_config(
            name=name,
            model_path=model_path,
            hf_token=hf_token,
            port=server_port,
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
        ))
    
    # Merge configs (kwargs override file config)
    config = merge_config(config, kwargs_config)
    
    # Validate
    if not config.get("runs"):
        raise ValueError("No benchmark runs defined. Provide model_path or config file with 'runs' section.")
    
    # Run benchmark
    try:
        if is_remote:
            remote_cfg = config.get("remote", {})
            if not remote_cfg.get("host"):
                raise ValueError("Remote host is required for remote benchmark.")
            _run_remote(config, remote_cfg)
            return {
                "status": "success",
                "mode": "remote",
                "host": remote_cfg.get("host"),
                "runs": len(config.get("runs", [])),
            }
        else:
            # Local run
            return vllm.run(config)
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "mode": "remote" if is_remote else "local",
        }


__all__ = [
    "__version__",
    "bench",
    "vllm",
    "runpod",
    "BenchConfig",
    "BenchmarkResult",
    "load_config",
]
