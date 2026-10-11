#!/usr/bin/env bash
set -euo pipefail

ARCHIVE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ARCHIVE_ROOT"

export PYTHONPATH="$ARCHIVE_ROOT/src:$ARCHIVE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

GPU_ID="${1:-0}"
python -u addition/check_fast_dataset_inputs.py
python -u addition/dispatch_fast_native.py --gpus "$GPU_ID" --plan
python -u addition/dispatch_fast_native.py --gpus "$GPU_ID"
