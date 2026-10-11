#!/usr/bin/env bash
set -euo pipefail

ARCHIVE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER4_ROOT="${PAPER4_ROOT:-$(cd "$ARCHIVE_ROOT/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ARCHITECTURE="ATCNet"

export PYTHONPATH="$PAPER4_ROOT/src:$PAPER4_ROOT:${PYTHONPATH:-}"

if [[ "${PREPROCESS_BIMODAL:-0}" == "1" ]]; then
  "$PYTHON_BIN" -u "$PAPER4_ROOT/preprocessing.py" \
    --config "$ARCHIVE_ROOT/configs/bimodal_inner_speech.yaml"
fi

for config in \
  "$ARCHIVE_ROOT/configs/ds003626.yaml" \
  "$ARCHIVE_ROOT/configs/bimodal_inner_speech.yaml" \
  "$ARCHIVE_ROOT/configs/directional_words_russian.yaml" \
  "$ARCHIVE_ROOT/configs/directional_words_spanish.yaml"; do
  "$PYTHON_BIN" -u \
    "$PAPER4_ROOT/src/eeg_pipeline/models/baseline_model.py" \
    --config "$config" \
    --architecture "$ARCHITECTURE"
done

