#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/../OpenNeruro_ds003626}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/4.6/4.6.2}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-auto}"
if [[ -n "$GPU_ID" ]]; then export CUDA_VISIBLE_DEVICES="$GPU_ID"; fi
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

arguments=(
  --project-root "$PROJECT_ROOT" --data-root "$DATA_ROOT"
  --output-root "$OUTPUT_ROOT" --device "$DEVICE"
  --shard-index "${SHARD_INDEX:-0}" --shard-count "${SHARD_COUNT:-1}"
)
[[ -n "${VARIANTS:-}" ]] && arguments+=(--variants "$VARIANTS")
[[ -n "${REPEATS:-}" ]] && arguments+=(--repeats "$REPEATS")
python -u "$SCRIPT_DIR/scripts/run_4_6_2.py" "${arguments[@]}" "$@"
