#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/../OpenNeruro_ds003626}"
CONFIG_PATH="${CONFIG_PATH:?Set CONFIG_PATH to one archived training YAML.}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to its best_model.pt.}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/4.7/4.7.2/offset_mechanism/$(date +%Y%m%d_%H%M%S)}"
CPU_THREADS="${CPU_THREADS:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_TRIALS="${MAX_TRIALS:-0}"
DEVICE="${DEVICE:-cpu}"

export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"
export NUMEXPR_NUM_THREADS="$CPU_THREADS"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python -u "$SCRIPT_DIR/scripts/offset_mechanism.py" \
  --project-root "$PROJECT_ROOT" --data-root "$DATA_ROOT" \
  --config "$CONFIG_PATH" --checkpoint "$CHECKPOINT_PATH" \
  --output-dir "$OUTPUT_DIR" --device "$DEVICE" \
  --cpu-threads "$CPU_THREADS" --batch-size "$BATCH_SIZE" \
  --max-trials "$MAX_TRIALS" --experiment "${EXPERIMENT:-both}" \
  --shifts=-8,-4,-2,-1,-0.5,0,0.5,1,2,4,8 "$@"
