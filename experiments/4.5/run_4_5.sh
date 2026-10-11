#!/usr/bin/env bash
set -euo pipefail

# Run from the paper4 Git repository root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/../OpenNeruro_ds003626}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/4.5}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-auto}"
MODELS="${MODELS:-eegnet,causal1_eegnet,group4_causal1_deformableLstm,group4_patchLstm_noDeformation,group4_patchLstm_deformableLstm}"

if [[ -n "$GPU_ID" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

arguments=(
  --project-root "$PROJECT_ROOT"
  --data-root "$DATA_ROOT"
  --output-root "$OUTPUT_ROOT"
  --device "$DEVICE"
  --models "$MODELS"
  --shard-index "${SHARD_INDEX:-0}"
  --shard-count "${SHARD_COUNT:-1}"
)

python -u "$SCRIPT_DIR/scripts/run_4_5.py" "${arguments[@]}" "$@"
