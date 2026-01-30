"""SkyPilot end-to-end benchmark runner.

Usage:
    import benchmaq.skypilot.bench as bench
    result = bench.from_yaml("config.yaml")

CLI:
    benchmaq skypilot bench config.yaml

Prerequisites:
    Users must authenticate with SkyPilot before using this module:
    - Run `sky auth` or
    - Set SKYPILOT_API_SERVER_URL and SKYPILOT_API_KEY environment variables
"""

import yaml
from uuid import uuid4
from typing import Dict, Any


def from_yaml(config_path: str) -> Dict[str, Any]:
    """Run end-to-end SkyPilot benchmark from YAML config.
    
    This launches a SkyPilot cluster, runs benchmarks, and tears down the cluster.
    The config file should contain both 'skypilot' and 'benchmark' sections.
    
    Args:
        config_path: Path to YAML configuration file.
        
    Returns:
        Dict with status and results.
        
    Example YAML structure:
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
            HF_TOKEN: "..."
          setup: |
            pip install "benchmaq[vllm]"
          run: |
            benchmaq bench config.yaml
        
        benchmark:
          - name: my_benchmark
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
    
    Prerequisites:
        Users must authenticate with SkyPilot before calling this:
        - Run `sky auth` or
        - Set SKYPILOT_API_SERVER_URL and SKYPILOT_API_KEY environment variables
    """
    from benchmaq.config import load_config
    from .core.client import launch_cluster, teardown_cluster
    
    config = load_config(config_path)
    skypilot_cfg = config.get("skypilot", {})
    
    if not skypilot_cfg:
        raise ValueError("No 'skypilot' section found in config")
    
    # Get cluster name from config or generate one
    cluster_name = skypilot_cfg.get("name", f"benchmaq-{uuid4().hex[:8]}")
    
    # Convert skypilot config to YAML string for sky.Task.from_yaml_str()
    skypilot_yaml = yaml.dump(skypilot_cfg)
    
    # Replace $config placeholder with actual config path
    skypilot_yaml = skypilot_yaml.replace("$config", config_path)
    
    print()
    print("=" * 64)
    print("SKYPILOT BENCHMARK")
    print("=" * 64)
    print(f"Cluster name: {cluster_name}")
    print(f"Config: {config_path}")
    print()
    
    try:
        print("=" * 64)
        print("STEP 1: LAUNCHING SKYPILOT CLUSTER")
        print("=" * 64)
        print()
        
        # Launch cluster with down=True for automatic cleanup after job completes
        result = launch_cluster(
            task_yaml=skypilot_yaml,
            cluster_name=cluster_name,
            down=True,
        )
        
        print()
        print("=" * 64)
        print("BENCHMARK COMPLETED!")
        print("=" * 64)
        print(f"Job ID: {result.get('job_id')}")
        print()
        
        return {
            "status": "success",
            "cluster_name": cluster_name,
            "job_id": result.get("job_id"),
        }
        
    except KeyboardInterrupt:
        print()
        print("=" * 64)
        print("INTERRUPTED BY USER")
        print("=" * 64)
        print()
        print("Attempting to tear down cluster...")
        
        try:
            teardown_cluster(cluster_name)
            print(f"Cluster {cluster_name} torn down successfully")
        except Exception as cleanup_error:
            print(f"Warning: Failed to tear down cluster {cluster_name}: {cleanup_error}")
            print("Please manually run: sky down " + cluster_name)
        
        return {"status": "interrupted", "cluster_name": cluster_name}
        
    except Exception as e:
        print()
        print("=" * 64)
        print(f"ERROR: {e}")
        print("=" * 64)
        
        # Try to clean up on error
        print()
        print("Attempting to tear down cluster...")
        try:
            teardown_cluster(cluster_name)
            print(f"Cluster {cluster_name} torn down successfully")
        except Exception as cleanup_error:
            print(f"Warning: Failed to tear down cluster {cluster_name}: {cleanup_error}")
            print("Please manually run: sky down " + cluster_name)
        
        return {"status": "error", "error": str(e), "cluster_name": cluster_name}
