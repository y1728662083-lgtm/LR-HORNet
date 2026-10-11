#!/usr/bin/env bash
set -euo pipefail

# Run this script while the shell's current directory is the paper4 Git root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/../OpenNeruro_ds003626}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/4.4.2}"
GPU_ID="${GPU_ID:-0}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python -u "$SCRIPT_DIR/run_4_4_2.py" \
  --config "$SCRIPT_DIR/config.yaml" \
  --project-root "$PROJECT_ROOT" \
  --data-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --gpu-id "$GPU_ID" \
  "$@"

