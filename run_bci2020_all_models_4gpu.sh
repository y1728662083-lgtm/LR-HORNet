#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER4_ROOT="${PAPER4_ROOT:-$SCRIPT_DIR}"
BCI2020_ROOT="${BCI2020_ROOT:-$PAPER4_ROOT/data/bci2020}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PAPER4_ROOT/outputs/bci2020_all_models}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
CPU_THREADS="${CPU_THREADS:-auto}"
LOADER_WORKERS="${LOADER_WORKERS:-0}"

cd "$PAPER4_ROOT"
export PYTHONPATH="$PAPER4_ROOT/src:$PAPER4_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

if [[ ! -d "$BCI2020_ROOT" ]]; then
  echo "ERROR: dataset directory not found: $BCI2020_ROOT" >&2
  exit 2
fi

mkdir -p logs/bci2020_all_models

if [[ -d "$BCI2020_ROOT/derivatives/paper4/bci2020_standard" ]]; then
  FIF_COUNT="$(find "$BCI2020_ROOT/derivatives/paper4/bci2020_standard" -type f -name '*_desc-preproc_epo.fif' | wc -l)"
else
  FIF_COUNT=0
fi
if [[ "$FIF_COUNT" -ne 45 ]]; then
  echo "Preprocessed records=$FIF_COUNT; running the audited standard preprocessing..."
  OVERWRITE_ARGS=()
  if [[ "$FIF_COUNT" -gt 0 ]]; then
    echo "Partial preprocessing output detected; rebuilding all records consistently."
    OVERWRITE_ARGS+=(--overwrite)
  fi
  python -u addition/bci2020_preprocessing/preprocess_bci2020.py \
    --config addition/bci2020_preprocessing/configs/bci2020_standard.yaml \
    --dataset-root "$BCI2020_ROOT" \
    --output-root "$BCI2020_ROOT/derivatives/paper4/bci2020_standard" \
    "${OVERWRITE_ARGS[@]}"
else
  echo "Reusing 45 existing preprocessed records."
fi

python -u addition/bci2020_preprocessing/verify_outputs.py \
  --root "$BCI2020_ROOT/derivatives/paper4/bci2020_standard" \
  --expected-records 45

python -u addition/bci2020_suite/preflight_all_models.py \
  --dataset-root "$BCI2020_ROOT" \
  --output-root "$OUTPUT_ROOT"
DISPATCH_ARGS=(
  --gpus "$GPU_LIST"
  --dataset-root "$BCI2020_ROOT"
  --output-root "$OUTPUT_ROOT"
)
if [[ "$CPU_THREADS" != "auto" ]]; then
  DISPATCH_ARGS+=(--cpu-threads-per-process "$CPU_THREADS")
fi
DISPATCH_ARGS+=(--loader-workers-per-process "$LOADER_WORKERS")

python -u addition/bci2020_suite/dispatch_all_models_4gpu.py \
  "${DISPATCH_ARGS[@]}" \
  --plan > logs/bci2020_all_models/plan.log

echo "Launching 90 fits across GPUs $GPU_LIST"
python -u addition/bci2020_suite/dispatch_all_models_4gpu.py \
  "${DISPATCH_ARGS[@]}" \
  --max-failures 1

python -u addition/bci2020_suite/summarize_all_models.py \
  --root "$OUTPUT_ROOT" \
  --expected-folds 15
