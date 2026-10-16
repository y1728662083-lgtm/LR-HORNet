#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER4_ROOT="${PAPER4_ROOT:-$SCRIPT_DIR}"
BCI2020_ROOT="${BCI2020_ROOT:-$PAPER4_ROOT/data/bci2020}"
BASELINE_OUTPUT_ROOT="${BASELINE_OUTPUT_ROOT:-$PAPER4_ROOT/outputs/bci2020_all_models}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
CPU_THREADS="${CPU_THREADS:-auto}"

cd "$PAPER4_ROOT"
export PYTHONPATH="$PAPER4_ROOT/src:$PAPER4_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
mkdir -p logs/bci2020_all_models

python -u addition/bci2020_suite/preflight_all_models.py \
  --dataset-root "$BCI2020_ROOT" \
  --output-root "$BASELINE_OUTPUT_ROOT"

BASELINE_ARGS=(
  --gpus "$GPU_LIST"
  --models eegnet atcnet eegmamba fast labram
  --folds 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
  --loader-workers-per-process 0
  --max-failures 1
  --dataset-root "$BCI2020_ROOT"
  --output-root "$BASELINE_OUTPUT_ROOT"
)
if [[ "$CPU_THREADS" != "auto" ]]; then
  BASELINE_ARGS+=(--cpu-threads-per-process "$CPU_THREADS")
fi

echo "PHASE 1/2: five comparison models, 15 folds each (75 fits)."
python -u addition/bci2020_suite/dispatch_all_models_4gpu.py \
  "${BASELINE_ARGS[@]}" --plan \
  > logs/bci2020_all_models/other_models_plan.log
python -u addition/bci2020_suite/dispatch_all_models_4gpu.py \
  "${BASELINE_ARGS[@]}"

python -u addition/bci2020_suite/summarize_all_models.py \
  --root "$BASELINE_OUTPUT_ROOT" \
  --expected-folds 15 \
  --models eegnet atcnet eegmamba fast labram

echo "PHASE 1/2 COMPLETE: all 75 comparison-model fits passed."
echo "PHASE 2/2: LR-HORNet, fixed seed 42, 15 folds x 10 repeats (150 fits)."
PAPER4_ROOT="$PAPER4_ROOT" BCI2020_ROOT="$BCI2020_ROOT" \
  bash "$PAPER4_ROOT/run_bci2020_lr_hornet_repeat10_4gpu.sh"
echo "ALL PHASES COMPLETE: 75 comparison fits + 150 LR-HORNet fits."
