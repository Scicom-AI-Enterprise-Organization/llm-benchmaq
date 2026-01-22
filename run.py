#!/usr/bin/env python3
"""
LLM Benchmark Runner

Dispatches benchmark runs to the appropriate engine (vllm, tensorrt-llm, sglang, etc.)
Supports remote execution on GPU servers via SSH with live streaming output.
"""

import importlib
import json
import os
import sys

import yaml


SUPPORTED_ENGINES = ["vllm"]


def generate_benchmark_script(config: dict) -> str:
    """Generate Python script to run benchmarks on remote server."""
    # Convert config to Python-compatible string (True/False instead of true/false)
    config_str = json.dumps(config)
    config_str = config_str.replace(': true', ': True').replace(': false', ': False').replace(': null', ': None')
    
    return '''
import os
import signal
import socket
import subprocess
import sys
import time

import requests

CONFIG = ''' + config_str + '''

class VLLMServer:
    def __init__(self, model_path, port, tp, dp, pp,
                 gpu_memory_utilization=0.9,
                 max_model_len=None,
                 max_num_seqs=None,
                 dtype=None,
                 disable_log_requests=False,
                 enable_expert_parallel=False):
        self.model_path = model_path
        self.port = port
        self.tp = tp
        self.dp = dp
        self.pp = pp
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.dtype = dtype
        self.disable_log_requests = disable_log_requests
        self.enable_expert_parallel = enable_expert_parallel
        self.process = None
        self.base_url = f"http://localhost:{port}"

    def start(self):
        cmd = [
            "vllm", "serve", self.model_path,
            "--port", str(self.port),
            "--tensor-parallel-size", str(self.tp),
            "--pipeline-parallel-size", str(self.pp),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
        ]
        if self.dp > 1:
            cmd.extend(["--data-parallel-size", str(self.dp)])
        if self.max_model_len:
            cmd.extend(["--max-model-len", str(self.max_model_len)])
        if self.max_num_seqs:
            cmd.extend(["--max-num-seqs", str(self.max_num_seqs)])
        if self.dtype:
            cmd.extend(["--dtype", self.dtype])
        if self.disable_log_requests:
            cmd.append("--disable-log-requests")
        if self.enable_expert_parallel:
            cmd.append("--enable-expert-parallel")

        print(f"Starting vLLM server: {' '.join(cmd)}")
        sys.stdout.flush()
        self.process = subprocess.Popen(cmd, text=True)
        return self._wait_for_health()

    def _wait_for_health(self, max_attempts=200, interval=5.0):
        health_url = f"{self.base_url}/health"
        print(f"Waiting for server at {health_url}...")
        sys.stdout.flush()
        for attempt in range(max_attempts):
            try:
                resp = requests.get(health_url, timeout=5.0)
                if resp.status_code == 200:
                    print(f"Server healthy after {attempt + 1} attempts")
                    sys.stdout.flush()
                    return True
            except requests.RequestException:
                pass
            if attempt % 10 == 0:
                print(f"Health check attempt {attempt + 1}...")
                sys.stdout.flush()
            time.sleep(interval)
        print(f"Server failed to become healthy after {max_attempts} attempts")
        sys.stdout.flush()
        return False

    def stop(self):
        if self.process is None or self.process.poll() is not None:
            return
        print(f"Stopping vLLM server on port {self.port}...")
        sys.stdout.flush()
        self.process.send_signal(signal.SIGINT)
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            print("Force killing server...")
            self.process.kill()
            self.process.wait()
        for _ in range(30):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", self.port))
                print(f"Port {self.port} released")
                break
            except OSError:
                time.sleep(1.0)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


def run_benchmark(model_path, port, output_dir, result_name, ctx, output_len, num_prompts, concurrency, save_results=False):
    print()
    print("=" * 64)
    print(f"BENCHMARK: {result_name}")
    print("=" * 64)
    sys.stdout.flush()

    cmd = [
        "vllm", "bench", "serve",
        "--backend", "vllm",
        "--base-url", f"http://localhost:{port}",
        "--model", model_path,
        "--endpoint", "/v1/completions",
        "--dataset-name", "random",
        "--random-input-len", str(ctx),
        "--random-output-len", str(output_len),
        "--num-prompts", str(num_prompts),
        "--max-concurrency", str(concurrency),
        "--request-rate", "inf",
        "--ignore-eos",
        "--percentile-metrics", "ttft,tpot,itl,e2el",
    ]

    if save_results:
        cmd.extend([
            "--save-result",
            "--result-dir", output_dir,
            "--result-filename", f"{result_name}.json",
        ])

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    for line in process.stdout:
        print(line, end='', flush=True)
    process.wait()


def main():
    config = CONFIG
    for run_cfg in config.get("runs", []):
        name = run_cfg.get("name", "")
        serve_cfg = run_cfg.get("serve", run_cfg)
        bench_cfg = run_cfg.get("bench", run_cfg)

        model_path = serve_cfg.get("model_path", "")
        port = serve_cfg.get("port", 8000)
        gpu_memory_utilization = serve_cfg.get("gpu_memory_utilization", 0.9)
        max_model_len = serve_cfg.get("max_model_len")
        max_num_seqs = serve_cfg.get("max_num_seqs")
        dtype = serve_cfg.get("dtype")
        disable_log_requests = serve_cfg.get("disable_log_requests", False)
        enable_expert_parallel = serve_cfg.get("enable_expert_parallel", False)
        tp_dp_pairs = serve_cfg.get("tp_dp_pairs", [])

        output_dir = bench_cfg.get("output_dir", "./benchmark_results")
        context_sizes = bench_cfg.get("context_size", [])
        concurrencies = bench_cfg.get("concurrency", [])
        num_prompts_list = bench_cfg.get("num_prompts", [])
        output_lens = bench_cfg.get("output_len", [])
        save_results = bench_cfg.get("save_results", False)

        if not name or not model_path:
            continue

        if save_results:
            os.makedirs(output_dir, exist_ok=True)

        for pair in tp_dp_pairs:
            tp = pair.get("tp", 1)
            dp = pair.get("dp", 1)
            pp = pair.get("pp", 1)

            print()
            print("=" * 64)
            print(f"RUN: {name} | TP={tp} DP={dp} PP={pp}")
            print("=" * 64)
            sys.stdout.flush()

            with VLLMServer(
                model_path, port, tp, dp, pp,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                max_num_seqs=max_num_seqs,
                dtype=dtype,
                disable_log_requests=disable_log_requests,
                enable_expert_parallel=enable_expert_parallel
            ) as server:
                for ctx in context_sizes:
                    for concurrency in concurrencies:
                        for num_prompts in num_prompts_list:
                            for output_len in output_lens:
                                result_name = f"{name}_TP{tp}_DP{dp}_CTX{ctx}_C{concurrency}_P{num_prompts}_O{output_len}"
                                run_benchmark(
                                    model_path, port, output_dir, result_name,
                                    ctx, output_len, num_prompts, concurrency,
                                    save_results=save_results
                                )

            time.sleep(5)

    print()
    print("=" * 64)
    print("BENCHMARK COMPLETED!")
    print("=" * 64)


if __name__ == "__main__":
    main()
'''


def run_remote(config: dict, remote_cfg: dict):
    """Execute benchmark on a remote GPU server via SSH with live streaming."""
    import paramiko
    
    host = remote_cfg["host"]
    username = remote_cfg["username"]
    password = remote_cfg["password"]
    
    uv_cfg = remote_cfg.get("uv", {})
    uv_path = uv_cfg.get("path", "~/.benchmark-venv")
    python_version = uv_cfg.get("python_version", "3.11")
    
    deps = remote_cfg.get("dependencies", [
        "pyyaml",
        "requests",
        "vllm==0.11.0",
        "huggingface_hub",
    ])
    
    print(f"Connecting to remote server: {username}@{host}")
    print(f"UV environment: {uv_path} (Python {python_version})")
    print(f"Dependencies: {deps}")
    print()
    
    # Connect via SSH
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, username=username, password=password)
    
    # Expand uv_path
    if uv_path.startswith("~"):
        stdin, stdout, stderr = ssh.exec_command("echo $HOME")
        home_dir = stdout.read().decode().strip()
        uv_path = uv_path.replace("~", home_dir)
    
    # Setup UV environment if needed
    print("Setting up UV environment...")
    setup_cmd = f"""
    if ! command -v uv &> /dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
    
    if [ ! -d "{uv_path}" ]; then
        uv venv "{uv_path}" --python {python_version}
    fi
    
    source "{uv_path}/bin/activate"
    uv pip install {' '.join(deps)}
    """
    
    stdin, stdout, stderr = ssh.exec_command(setup_cmd)
    exit_status = stdout.channel.recv_exit_status()
    if exit_status != 0:
        print(f"Setup error: {stderr.read().decode()}")
    else:
        print("Environment ready!")
    print()
    
    # Generate and upload benchmark script
    script_content = generate_benchmark_script(config)
    remote_script_path = "/tmp/benchmark_runner.py"
    
    sftp = ssh.open_sftp()
    with sftp.file(remote_script_path, 'w') as f:
        f.write(script_content)
    sftp.close()
    
    print("=" * 64)
    print("STARTING REMOTE BENCHMARK (LIVE STREAM)")
    print("=" * 64)
    print()
    
    # Execute with live streaming output
    run_cmd = f'source "{uv_path}/bin/activate" && python {remote_script_path}'
    
    # Use invoke_shell for better streaming
    channel = ssh.get_transport().open_session()
    channel.get_pty()
    channel.exec_command(run_cmd)
    
    # Stream output in real-time
    while True:
        if channel.recv_ready():
            data = channel.recv(4096).decode('utf-8', errors='replace')
            print(data, end='', flush=True)
        if channel.recv_stderr_ready():
            data = channel.recv_stderr(4096).decode('utf-8', errors='replace')
            print(data, end='', flush=True)
        if channel.exit_status_ready():
            # Get any remaining data
            while channel.recv_ready():
                data = channel.recv(4096).decode('utf-8', errors='replace')
                print(data, end='', flush=True)
            break
    
    exit_status = channel.recv_exit_status()
    channel.close()
    ssh.close()
    
    print()
    print("=" * 64)
    print(f"Remote execution completed! (exit code: {exit_status})")
    print("=" * 64)


def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py <config.yaml>")
        print("Example: python run.py config/my-benchmark.yaml")
        sys.exit(1)

    config_path = sys.argv[1]

    if not os.path.isabs(config_path):
        config_path = os.path.abspath(config_path)

    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    print(f"Loading config: {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    runs = config.get("runs", [])
    if not runs:
        print("Error: No runs defined in config")
        sys.exit(1)

    engine = runs[0].get("engine", "vllm")

    if engine not in SUPPORTED_ENGINES:
        print(f"Error: Unsupported engine '{engine}'. Supported: {SUPPORTED_ENGINES}")
        sys.exit(1)

    print()
    print("=" * 64)
    print(f"ENGINE: {engine}")
    print("=" * 64)

    remote_cfg = config.get("remote")
    if remote_cfg:
        print("REMOTE EXECUTION ENABLED")
        print("=" * 64)
        run_remote(config, remote_cfg)
    else:
        engine_module = importlib.import_module(engine)
        engine_module.run(config)


if __name__ == "__main__":
    main()
