"""vLLM benchmarking module.

This module provides a YAML-only API for running vLLM benchmarks.
All configuration is done through YAML files with dynamic kwargs support.

Usage:
    import benchmaq.vllm as vllm
    
    # Run benchmarks from YAML config
    result = vllm.from_yaml("config.yaml")
"""

import os
import time
import hashlib
from typing import Optional, List, Dict, Any

from .core import VLLMServer, run_benchmark
from .core.benchmark import run_benchmark_legacy


def from_yaml(config_path: str) -> Dict[str, Any]:
    """Run vLLM benchmarks from YAML config only.
    
    This is the main entry point for the standardized YAML-only API.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dict with status and results
        
    Example YAML structure:
        benchmark:
          - name: my_benchmark
            engine: vllm
            model:
              repo_id: "model/repo"
              hf_token: ""
            serve:
              model: "model/repo"
              tensor_parallel_size: 8
              max_num_seqs: 256
            bench:
              - backend: vllm
                endpoint: /v1/completions
                dataset_name: random
                random_input_len: 1024
                num_prompts: 100
            results:
              save_result: true
              result_dir: "./results"
    """
    from ..config import load_config
    
    config = load_config(config_path)
    return run(config)


def run(config: dict) -> Dict[str, Any]:
    """Run vLLM benchmarks based on config dict.
    
    Supports both new 'benchmark:' structure and legacy 'runs:' structure.
    
    New structure (recommended):
        benchmark:
          - name: serve_config_1
            serve: {...}      # vLLM serve kwargs
            bench: [...]      # Array of vLLM bench serve kwargs
            results: {...}    # Results configuration
            
    Legacy structure (backward compatible):
        runs:
          - name: run_1
            vllm_serve: {...}
            benchmark: {...}
    """
    results = []
    
    # Check for new 'benchmark:' structure first
    if "benchmark" in config:
        results = _run_new_structure(config)
    # Fall back to legacy 'runs:' structure
    elif "runs" in config:
        results = _run_legacy_structure(config)
    else:
        raise ValueError("No 'benchmark:' or 'runs:' section found in config.")
    
    return {"status": "success", "results": results}


def _download_model(repo_id: str, local_dir: str, hf_token: Optional[str] = None):
    """Download model from HuggingFace Hub."""
    import subprocess
    
    print()
    print("=" * 64)
    print(f"DOWNLOADING MODEL: {repo_id}")
    print(f"Destination: {local_dir}")
    print("=" * 64)
    
    os.makedirs(local_dir, exist_ok=True)
    
    env = os.environ.copy()
    token = hf_token or os.environ.get("HF_TOKEN")
    if token:
        env["HF_TOKEN"] = token
    
    # Enable hf_transfer for faster downloads if available
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    
    cmd = ["huggingface-cli", "download", repo_id, "--local-dir", local_dir]
    print(f"Running: {' '.join(cmd)}")
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    for line in process.stdout:
        print(line, end='', flush=True)
    process.wait()
    
    if process.returncode != 0:
        raise Exception(f"Model download failed with exit code {process.returncode}")
    
    print()
    print("Model download completed!")


def _run_new_structure(config: dict) -> List[Dict[str, Any]]:
    """Run benchmarks using new standardized YAML structure.
    
    Structure:
        benchmark:
          - name: config_name
            engine: vllm
            model: {...}      # Optional model download config
            serve: {...}      # vLLM serve kwargs
            bench: [...]      # Array of vLLM bench serve kwargs
            results: {...}    # Results config
    """
    results = []
    
    for run_cfg in config.get("benchmark", []):
        name = run_cfg.get("name", "benchmark")
        engine = run_cfg.get("engine", "vllm")
        
        if engine != "vllm":
            print(f"Skipping {name}: engine '{engine}' not supported (only 'vllm' supported)")
            continue
        
        # Extract configurations
        model_cfg = run_cfg.get("model", {})
        serve_cfg = run_cfg.get("serve", {}).copy()  # Copy to avoid modifying original
        bench_configs = run_cfg.get("bench", [])
        results_cfg = run_cfg.get("results", {})
        
        # Handle HF token from model config
        hf_token = model_cfg.get("hf_token") or os.environ.get("HF_TOKEN")
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
        
        # Download model if repo_id and local_dir are specified
        if model_cfg.get("repo_id") and model_cfg.get("local_dir"):
            _download_model(model_cfg["repo_id"], model_cfg["local_dir"], hf_token)
        
        # Determine model: serve.model > serve.model_path > model.local_dir > model.repo_id
        model = (
            serve_cfg.pop("model", None) or 
            serve_cfg.pop("model_path", None) or 
            model_cfg.get("local_dir") or 
            model_cfg.get("repo_id", "")
        )
        port = serve_cfg.pop("port", 8000)
        
        if not model:
            print(f"Skipping {name}: no model specified in 'serve:' or 'model:' section")
            continue
        
        if not bench_configs:
            print(f"Skipping {name}: no 'bench:' configurations found")
            continue
        
        print()
        print("=" * 64)
        print(f"CONFIGURATION: {name}")
        print(f"Model: {model}")
        print(f"Serve kwargs: {serve_cfg}")
        print("=" * 64)
        
        # Start vLLM server with serve kwargs
        with VLLMServer(model=model, port=port, **serve_cfg) as server:
            # Run each bench configuration (parameter sweep)
            for i, bench_cfg in enumerate(bench_configs):
                # Generate unique result name
                result_name = _generate_result_name(name, i, bench_cfg)
                
                print()
                print(f"--- Benchmark {i + 1}/{len(bench_configs)}: {result_name} ---")
                
                # Run benchmark with bench kwargs
                run_benchmark(
                    model=model,
                    port=port,
                    result_name=result_name,
                    results_config=results_cfg,
                    **bench_cfg
                )
                
                results.append({
                    "name": result_name,
                    "config": name,
                    "bench_index": i,
                    **bench_cfg
                })
        
        # Brief pause between configurations
        time.sleep(5)
    
    return results


def _generate_result_name(config_name: str, index: int, bench_cfg: dict) -> str:
    """Generate a unique result name from config name and bench parameters."""
    # Create a short hash of the bench config for uniqueness
    cfg_str = str(sorted(bench_cfg.items()))
    cfg_hash = hashlib.md5(cfg_str.encode()).hexdigest()[:6]
    
    # Include key parameters in the name if present
    parts = [config_name]
    
    if "random_input_len" in bench_cfg:
        parts.append(f"in{bench_cfg['random_input_len']}")
    if "random_output_len" in bench_cfg:
        parts.append(f"out{bench_cfg['random_output_len']}")
    if "num_prompts" in bench_cfg:
        parts.append(f"p{bench_cfg['num_prompts']}")
    if "max_concurrency" in bench_cfg:
        parts.append(f"c{bench_cfg['max_concurrency']}")
    
    parts.append(cfg_hash)
    
    return "_".join(parts)


def _run_legacy_structure(config: dict) -> List[Dict[str, Any]]:
    """Run benchmarks using legacy 'runs:' structure for backward compatibility."""
    results = []
    
    for run_cfg in config.get("runs", []):
        name = run_cfg.get("name", "")
        model_cfg = run_cfg.get("model", {})
        vllm_serve_cfg = run_cfg.get("vllm_serve", run_cfg)
        benchmark_cfg = run_cfg.get("benchmark", run_cfg)

        hf_token = model_cfg.get("hf_token") or config.get("hf_token")
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token

        model_path = vllm_serve_cfg.get("model_path", "")
        port = vllm_serve_cfg.get("port", 8000)
        parallelism_pairs = vllm_serve_cfg.get("parallelism_pairs", [])
        
        # Extract serve kwargs for new VLLMServer
        serve_kwargs = {}
        for key in ["gpu_memory_utilization", "max_model_len", "max_num_seqs", 
                    "dtype", "disable_log_requests", "enable_expert_parallel"]:
            if key in vllm_serve_cfg:
                serve_kwargs[key] = vllm_serve_cfg[key]

        output_dir = benchmark_cfg.get("output_dir", "./benchmark_results")
        context_sizes = benchmark_cfg.get("context_size", [])
        concurrencies = benchmark_cfg.get("concurrency", [])
        num_prompts_list = benchmark_cfg.get("num_prompts", [])
        output_lens = benchmark_cfg.get("output_len", [])
        save_results = benchmark_cfg.get("save_results", False)

        if not name or not model_path:
            continue

        if save_results:
            os.makedirs(output_dir, exist_ok=True)

        for pair in parallelism_pairs:
            tp = pair.get("tensor_parallel", 1)
            dp = pair.get("data_parallel", 1)
            pp = pair.get("pipeline_parallel", 1)

            print()
            print("=" * 64)
            print(f"RUN: {name} | TP={tp} DP={dp} PP={pp}")
            print("=" * 64)

            # Build serve kwargs with parallelism
            current_serve_kwargs = serve_kwargs.copy()
            current_serve_kwargs["tensor_parallel_size"] = tp
            current_serve_kwargs["pipeline_parallel_size"] = pp
            if dp > 1:
                current_serve_kwargs["data_parallel_size"] = dp

            with VLLMServer(model=model_path, port=port, **current_serve_kwargs) as server:
                for ctx in context_sizes:
                    for concurrency in concurrencies:
                        for num_prompts in num_prompts_list:
                            for output_len in output_lens:
                                result_name = f"{name}_TP{tp}_DP{dp}_CTX{ctx}_C{concurrency}_P{num_prompts}_O{output_len}"
                                run_benchmark_legacy(
                                    model_path, port, output_dir, result_name,
                                    ctx, output_len, num_prompts, concurrency,
                                    save_results=save_results
                                )
                                results.append({
                                    "name": result_name, "tp": tp, "dp": dp, "pp": pp,
                                    "ctx": ctx, "concurrency": concurrency,
                                    "num_prompts": num_prompts, "output_len": output_len,
                                })

            time.sleep(5)

    return results


def bench(
    config_path: Optional[str] = None,
    *,
    name: str = "benchmark",
    model_path: Optional[str] = None,
    hf_token: Optional[str] = None,
    port: int = 8000,
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
    """Run vLLM benchmarks with kwargs or config file.
    
    DEPRECATED: Use from_yaml() with a YAML config file instead.
    This function is kept for backward compatibility.
    """
    import warnings
    warnings.warn(
        "bench() is deprecated. Use from_yaml() with a YAML config file instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    from ..config import load_config, merge_config, kwargs_to_run_config

    config = {}
    if config_path:
        config = load_config(config_path)

    if model_path:
        kwargs_config = kwargs_to_run_config(
            name=name, model_path=model_path, hf_token=hf_token, port=port,
            tensor_parallel=tensor_parallel, data_parallel=data_parallel, pipeline_parallel=pipeline_parallel,
            parallelism_pairs=parallelism_pairs, gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len, max_num_seqs=max_num_seqs, dtype=dtype,
            disable_log_requests=disable_log_requests, enable_expert_parallel=enable_expert_parallel,
            context_sizes=context_sizes or [1024], concurrency=concurrency or [50],
            num_prompts=num_prompts or [100], output_len=output_len or [128],
            output_dir=output_dir, save_results=save_results,
        )
        config = merge_config(config, kwargs_config)

    if not config.get("runs"):
        raise ValueError("No benchmark runs defined. Provide model_path or config file with 'runs' section.")

    try:
        return run(config)
    except Exception as e:
        return {"status": "error", "error": str(e)}


__all__ = ["VLLMServer", "run_benchmark", "run", "bench", "from_yaml"]
