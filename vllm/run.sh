#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${1:-config.yaml}"

uv pip install -r "${SCRIPT_DIR}/requirements.txt"
uv run python "${SCRIPT_DIR}/run.py" --config "$CONFIG_FILE"
