#!/usr/bin/env python3
"""
benchmaq CLI - LLM benchmarking toolkit

Usage:
    benchmaq vllm from-yaml <config.yaml>     # Run vLLM benchmark from YAML (recommended)
    benchmaq bench <config.yaml>              # Run benchmark (legacy)
    benchmaq runpod <subcommand> ...          # RunPod pod management
"""
import os
import sys
import argparse
import json

from .runner import run, run_e2e


def main():
    parser = argparse.ArgumentParser(
        prog="benchmaq",
        description="LLM benchmarking toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  benchmaq vllm from-yaml config.yaml     Run vLLM benchmark from YAML config
  benchmaq bench config.yaml              Run benchmark (legacy format)
  benchmaq runpod deploy config.yaml      Deploy a RunPod pod
  benchmaq runpod bench config.yaml       End-to-end RunPod benchmark
        """
    )
    subparsers = parser.add_subparsers(dest="command")

    # =================================================================
    # vllm command - new YAML-only API
    # =================================================================
    vllm_parser = subparsers.add_parser(
        "vllm",
        help="vLLM benchmarking commands",
        description="Run vLLM benchmarks using standardized YAML configuration"
    )
    vllm_subparsers = vllm_parser.add_subparsers(dest="vllm_command")

    # vllm from-yaml
    from_yaml_parser = vllm_subparsers.add_parser(
        "from-yaml",
        help="Run vLLM benchmark from YAML config file",
        description="""
Run vLLM benchmarks using the standardized YAML configuration format.

YAML Structure:
  benchmark:
    - name: config_name
      engine: vllm
      serve:           # vLLM serve kwargs
        model: "model/repo"
        tensor_parallel_size: 8
      bench:           # Array of vLLM bench serve kwargs
        - backend: vllm
          random_input_len: 1024
          num_prompts: 100
      results:         # Output configuration
        save_result: true
        result_dir: "./results"
        """
    )
    from_yaml_parser.add_argument("config", help="Path to YAML config file")

    # =================================================================
    # bench command - legacy
    # =================================================================
    bench_parser = subparsers.add_parser(
        "bench",
        help="Run benchmark (legacy format)",
        description="Run benchmark using legacy 'runs:' YAML format"
    )
    bench_parser.add_argument("config", help="Config YAML file")

    # =================================================================
    # runpod command
    # =================================================================
    runpod_parser = subparsers.add_parser(
        "runpod",
        help="RunPod pod management",
        description="Deploy, manage, and run benchmarks on RunPod GPU pods"
    )
    runpod_subparsers = runpod_parser.add_subparsers(dest="runpod_command")

    # runpod deploy
    deploy_parser = runpod_subparsers.add_parser("deploy", help="Deploy a pod from YAML config")
    deploy_parser.add_argument("config", help="Config YAML file")
    deploy_parser.add_argument("--no-wait", action="store_true", help="Don't wait for ready")

    # runpod delete
    delete_parser = runpod_subparsers.add_parser("delete", help="Delete a pod")
    delete_parser.add_argument("target", help="Pod ID or config YAML path")

    # runpod find
    find_parser = runpod_subparsers.add_parser("find", help="Get pod info")
    find_parser.add_argument("target", help="Pod ID or config YAML path")

    # runpod start
    start_parser = runpod_subparsers.add_parser("start", help="Start a stopped pod")
    start_parser.add_argument("target", help="Pod ID or config YAML path")

    # runpod stop
    stop_parser = runpod_subparsers.add_parser("stop", help="Stop a running pod")
    stop_parser.add_argument("target", help="Pod ID or config YAML path")

    # runpod bench - end-to-end: deploy -> run -> delete
    e2e_parser = runpod_subparsers.add_parser(
        "bench",
        help="End-to-end: deploy pod, run benchmarks, delete pod"
    )
    e2e_parser.add_argument("config", help="Config YAML file")

    args = parser.parse_args()

    # =================================================================
    # Handle vllm command
    # =================================================================
    if args.command == "vllm":
        if args.vllm_command == "from-yaml":
            from .vllm import from_yaml
            
            config_path = args.config
            if not os.path.exists(config_path):
                print(f"Error: Config file not found: {config_path}")
                sys.exit(1)
            
            print(f"Running vLLM benchmark from: {config_path}")
            result = from_yaml(config_path)
            
            if result.get("status") == "success":
                print("\n" + "=" * 64)
                print("BENCHMARK COMPLETED SUCCESSFULLY")
                print("=" * 64)
                print(f"Total runs: {len(result.get('results', []))}")
                for r in result.get("results", []):
                    print(f"  - {r.get('name', 'unknown')}")
            else:
                print(f"\nError: {result.get('error', 'Unknown error')}")
                sys.exit(1)
        else:
            vllm_parser.print_help()

    # =================================================================
    # Handle bench command (legacy)
    # =================================================================
    elif args.command == "bench":
        print("Note: 'benchmaq bench' uses legacy format. Consider using 'benchmaq vllm from-yaml' instead.")
        run(args.config)

    # =================================================================
    # Handle runpod command
    # =================================================================
    elif args.command == "runpod":
        from .runpod.core.client import deploy, delete, find, find_by_name, start, stop, set_api_key, get_api_key
        from .runpod.config import load_config

        def ensure_api_key():
            """Ensure API key is set on runpod module."""
            # Always try to set from environment if not already set via config
            env_key = os.environ.get("RUNPOD_API_KEY")
            if env_key:
                set_api_key(env_key)
            elif not get_api_key():
                print("Error: RUNPOD_API_KEY not set. Set it in environment or use a config file.")
                sys.exit(1)

        def load_api_key_from_config(config_path):
            config = load_config(config_path)
            if config.get("api_key"):
                set_api_key(config["api_key"])
            else:
                ensure_api_key()
            return config

        if args.runpod_command == "deploy":
            config = load_api_key_from_config(args.config)
            if args.no_wait:
                config["wait_for_ready"] = False
            instance = deploy(**config)
            print(json.dumps(instance, indent=2))

        elif args.runpod_command == "delete":
            target = args.target
            if target.endswith(".yaml") or target.endswith(".yml"):
                config = load_api_key_from_config(target)
                result = delete(name=config.get("name"))
            else:
                ensure_api_key()
                result = delete(pod_id=target)
            print(json.dumps(result, indent=2))

        elif args.runpod_command == "find":
            target = args.target
            if target.endswith(".yaml") or target.endswith(".yml"):
                config = load_api_key_from_config(target)
                pod = find_by_name(config.get("name"))
                result = pod if pod else {"error": f"Pod '{config.get('name')}' not found"}
            else:
                ensure_api_key()
                result = find(target)
            print(json.dumps(result, indent=2))

        elif args.runpod_command == "start":
            target = args.target
            if target.endswith(".yaml") or target.endswith(".yml"):
                config = load_api_key_from_config(target)
                pod = find_by_name(config.get("name"))
                if pod:
                    gpu_count = pod.get("gpuCount", 1)
                    result = start(pod["id"], gpu_count=gpu_count)
                else:
                    result = {"error": f"Pod '{config.get('name')}' not found"}
            else:
                ensure_api_key()
                # Need to get pod info first to know gpu_count
                pod = find(target)
                if pod:
                    gpu_count = pod.get("gpuCount", 1)
                    result = start(target, gpu_count=gpu_count)
                else:
                    result = {"error": f"Pod '{target}' not found"}
            print(json.dumps(result, indent=2))

        elif args.runpod_command == "stop":
            target = args.target
            if target.endswith(".yaml") or target.endswith(".yml"):
                config = load_api_key_from_config(target)
                pod = find_by_name(config.get("name"))
                if pod:
                    result = stop(pod["id"])
                else:
                    result = {"error": f"Pod '{config.get('name')}' not found"}
            else:
                ensure_api_key()
                result = stop(target)
            print(json.dumps(result, indent=2))

        elif args.runpod_command == "bench":
            import yaml
            config_path = args.config
            if not os.path.isabs(config_path):
                config_path = os.path.abspath(config_path)
            with open(config_path) as f:
                config = yaml.safe_load(f)
            run_e2e(config)

        else:
            runpod_parser.print_help()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
