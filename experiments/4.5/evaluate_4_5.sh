#!/usr/bin/env bash
set -euo pipefail

# Run from the paper4 Git repository root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/../OpenNeruro_ds003626}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-auto}"

if [[ -n "$GPU_ID" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

arguments=(
  --project-root "$PROJECT_ROOT"
  --data-root "$DATA_ROOT"
  --device "$DEVICE"
)

if [[ -n "${CHECKPOINT_PATH:-}" ]]; then
  arguments+=(--checkpoint "$CHECKPOINT_PATH")
  if [[ -n "${MODEL_NAME:-}" ]]; then
    arguments+=(--model "$MODEL_NAME")
  fi
  if [[ -n "${FOLD:-}" ]]; then
    arguments+=(--fold "$FOLD")
  fi
elif [[ -n "${CHECKPOINT_ROOT:-}" ]]; then
  arguments+=(--checkpoint-root "$CHECKPOINT_ROOT")
else
  echo "Set CHECKPOINT_PATH for one .pt file or CHECKPOINT_ROOT for the 4.5 result directory." >&2
  exit 2
fi

if [[ -n "${EVAL_OUTPUT:-}" ]]; then
  arguments+=(--output "$EVAL_OUTPUT")
fi

python -u "$SCRIPT_DIR/scripts/evaluate_4_5.py" "${arguments[@]}" "$@"
