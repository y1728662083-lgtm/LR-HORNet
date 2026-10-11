#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATA_ROOT="${DATA_ROOT:-$(dirname "$PROJECT_ROOT")/OpenNeruro_ds003626}"
DEVICE="${DEVICE:-auto}"

args=(--project-root "$PROJECT_ROOT" --data-root "$DATA_ROOT" --device "$DEVICE")
if [[ -n "${CHECKPOINT_PATH:-}" ]]; then
  args+=(--checkpoint "$CHECKPOINT_PATH")
elif [[ -n "${CHECKPOINT_ROOT:-}" ]]; then
  args+=(--checkpoint-root "$CHECKPOINT_ROOT")
else
  echo "Set CHECKPOINT_PATH or CHECKPOINT_ROOT." >&2
  exit 2
fi
if [[ -n "${MODEL_NAME:-}" ]]; then args+=(--model "$MODEL_NAME"); fi
if [[ -n "${FOLD:-}" ]]; then args+=(--fold "$FOLD"); fi
if [[ -n "${EVAL_OUTPUT:-}" ]]; then args+=(--output "$EVAL_OUTPUT"); fi

python "$(dirname "$0")/scripts/evaluate_4_8_1.py" "${args[@]}" "$@"
