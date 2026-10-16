#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER4_ROOT="${PAPER4_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
BCI2020_ROOT="${BCI2020_ROOT:-$PAPER4_ROOT/data/bci2020}"
DERIVATIVES_ROOT="${DERIVATIVES_ROOT:-$BCI2020_ROOT/derivatives/paper4/bci2020_standard}"

cd "$PAPER4_ROOT"
export PYTHONPATH="$PAPER4_ROOT/src:$PAPER4_ROOT:${PYTHONPATH:-}"

python -u addition/bci2020_preprocessing/preprocess_bci2020.py \
  --config addition/bci2020_preprocessing/configs/bci2020_standard.yaml \
  --dataset-root "$BCI2020_ROOT" \
  --output-root "$DERIVATIVES_ROOT"

python -u addition/bci2020_preprocessing/verify_outputs.py \
  --root "$DERIVATIVES_ROOT"

