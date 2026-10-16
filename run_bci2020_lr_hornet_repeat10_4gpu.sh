#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER4_ROOT="${PAPER4_ROOT:-$SCRIPT_DIR}"
BCI2020_ROOT="${BCI2020_ROOT:-$PAPER4_ROOT/data/bci2020}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PAPER4_ROOT/outputs/bci2020_lr_hornet_fixed_seed_repeat10}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
CPU_THREADS="${CPU_THREADS:-auto}"

cd "$PAPER4_ROOT"
export PYTHONPATH="$PAPER4_ROOT/src:$PAPER4_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

python -u addition/bci2020_suite/preflight_all_models.py \
  --configs-only \
  --dataset-root "$BCI2020_ROOT" \
  --output-root "$PAPER4_ROOT/outputs/bci2020_all_models"
python -u addition/bci2020_preprocessing/verify_outputs.py \
  --root "$BCI2020_ROOT/derivatives/paper4/bci2020_standard" \
  --expected-records 45

ARGS=(
  --gpus "$GPU_LIST"
  --repeats 10
  --folds 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
  --max-failures 1
  --dataset-root "$BCI2020_ROOT"
  --output-root "$OUTPUT_ROOT"
)
if [[ "$CPU_THREADS" != "auto" ]]; then
  ARGS+=(--cpu-threads-per-process "$CPU_THREADS")
fi

python -u addition/bci2020_suite/dispatch_lr_hornet_fixed_seed_repeats_4gpu.py \
  "${ARGS[@]}" --plan
python -u addition/bci2020_suite/dispatch_lr_hornet_fixed_seed_repeats_4gpu.py \
  "${ARGS[@]}"
python -u addition/bci2020_suite/summarize_lr_hornet_fixed_seed_repeats.py \
  --root "$OUTPUT_ROOT" \
  --expected-folds 15 \
  --expected-repeats 10 \
  --expected-seed 42
