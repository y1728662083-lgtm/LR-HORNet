#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/../OpenNeruro_ds003626}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-auto}"
if [[ -n "$GPU_ID" ]]; then export CUDA_VISIBLE_DEVICES="$GPU_ID"; fi
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

arguments=(--project-root "$PROJECT_ROOT" --data-root "$DATA_ROOT" --device "$DEVICE")
if [[ -n "${CHECKPOINT_PATH:-}" ]]; then
  arguments+=(--checkpoint "$CHECKPOINT_PATH")
  [[ -n "${MODEL_NAME:-}" ]] && arguments+=(--model "$MODEL_NAME")
  [[ -n "${FOLD:-}" ]] && arguments+=(--fold "$FOLD")
elif [[ -n "${CHECKPOINT_ROOT:-}" ]]; then
  arguments+=(--checkpoint-root "$CHECKPOINT_ROOT")
else
  echo "Set CHECKPOINT_PATH or CHECKPOINT_ROOT." >&2; exit 2
fi
[[ -n "${EVAL_OUTPUT:-}" ]] && arguments+=(--output "$EVAL_OUTPUT")
python -u "$SCRIPT_DIR/scripts/evaluate_4_7_1.py" "${arguments[@]}" "$@"
