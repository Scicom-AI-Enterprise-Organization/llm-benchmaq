#!/usr/bin/env python3
"""
LLM Benchmark Runner

Dispatches benchmark runs to the appropriate engine (vllm, tensorrt-llm, sglang, etc.)
"""

import importlib
import os
import sys

import yaml


SUPPORTED_ENGINES = ["vllm"]
CONFIG_FILE = "config.yaml"


def main():
    # Read base config
    if not os.path.exists(CONFIG_FILE):
        print(f"Error: {CONFIG_FILE} not found in current directory")
        print(f"Create a {CONFIG_FILE} with your benchmark configuration")
        sys.exit(1)

    with open(CONFIG_FILE) as f:
        base_config = yaml.safe_load(f)

    # Get config_path from base config (or use base config directly)
    config_path = base_config.get("config_path")
    
    if config_path:
        # Load actual config from config_path
        if not os.path.isabs(config_path):
            config_path = os.path.abspath(config_path)
        
        if not os.path.exists(config_path):
            print(f"Error: Config file not found: {config_path}")
            sys.exit(1)
        
        print(f"Loading config: {config_path}")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        base_path = os.path.dirname(config_path)
    else:
        # Use base config directly
        config = base_config
        base_path = os.getcwd()
        print(f"Loading config: {CONFIG_FILE}")

    runs = config.get("runs", [])
    if not runs:
        print("Error: No runs defined in config")
        sys.exit(1)

    # Get engine from first run (default: vllm)
    engine = runs[0].get("engine", "vllm")

    if engine not in SUPPORTED_ENGINES:
        print(f"Error: Unsupported engine '{engine}'. Supported: {SUPPORTED_ENGINES}")
        sys.exit(1)

    print()
    print("=" * 64)
    print(f"ENGINE: {engine}")
    print("=" * 64)

    # Import and run the engine module
    engine_module = importlib.import_module(engine)
    engine_module.run(config, base_path=base_path)


if __name__ == "__main__":
    main()
