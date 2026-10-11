#!/usr/bin/env bash
set -euo pipefail

ARCHIVE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ARCHIVE_ROOT"

export PYTHONPATH="$ARCHIVE_ROOT/src:$ARCHIVE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python -m pytest -q \
  addition/tests/test_fast_native_fullcheck.py \
  addition/tests/test_directional_words_intersection_adapter.py
python -u addition/preflight_fast_native.py
python -u addition/check_fast_dataset_inputs.py
