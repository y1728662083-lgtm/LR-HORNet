#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/../OpenNeruro_ds003626}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/4.7/4.7.2/group_04_visualization}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-auto}"
if [[ -n "$GPU_ID" ]]; then export CUDA_VISIBLE_DEVICES="$GPU_ID"; fi
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

arguments=(
  --project-root "$PROJECT_ROOT" --data-root "$DATA_ROOT"
  --config-dir "$SCRIPT_DIR/configs/group04_visualization"
  --output-dir "$OUTPUT_ROOT" --device "$DEVICE"
)
if [[ -n "${CHECKPOINT_ROOT:-}" ]]; then
  arguments+=(--checkpoint-root "$CHECKPOINT_ROOT")
else
  for repeat in 01 02 03 04 05; do
    variable="CHECKPOINT_${repeat}"
    path="${!variable:-}"
    if [[ -z "$path" ]]; then
      echo "Set CHECKPOINT_ROOT or CHECKPOINT_01 through CHECKPOINT_05." >&2; exit 2
    fi
    arguments+=(--checkpoint "$((10#$repeat))=$path")
  done
fi
python -u "$SCRIPT_DIR/scripts/visualize_fixed_trial.py" "${arguments[@]}" "$@"
