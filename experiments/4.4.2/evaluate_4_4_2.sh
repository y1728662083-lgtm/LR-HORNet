#!/usr/bin/env bash
set -euo pipefail

# Run this script while the shell's current directory is the paper4 Git root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/../OpenNeruro_ds003626}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-auto}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

arguments=(
  --config "$SCRIPT_DIR/config.yaml"
  --project-root "$PROJECT_ROOT"
  --data-root "$DATA_ROOT"
  --device "$DEVICE"
)

if [[ -n "${CHECKPOINT_PATH:-}" ]]; then
  arguments+=(--checkpoint "$CHECKPOINT_PATH")
  if [[ -n "${TEST_SUBJECT:-}" ]]; then
    arguments+=(--test-subject "$TEST_SUBJECT")
  fi
  if [[ -n "${VAL_SUBJECT:-}" ]]; then
    arguments+=(--val-subject "$VAL_SUBJECT")
  fi
elif [[ -n "${CHECKPOINT_ROOT:-}" ]]; then
  arguments+=(--checkpoint-root "$CHECKPOINT_ROOT")
else
  echo "Set CHECKPOINT_PATH for one .pt file or CHECKPOINT_ROOT for a group_05 directory." >&2
  exit 2
fi

if [[ -n "${EVAL_OUTPUT:-}" ]]; then
  arguments+=(--output "$EVAL_OUTPUT")
fi

python -u "$SCRIPT_DIR/evaluate_4_4_2.py" "${arguments[@]}" "$@"

