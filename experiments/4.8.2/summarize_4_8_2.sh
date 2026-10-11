#!/usr/bin/env bash
set -euo pipefail

RESULT_ROOT="${RESULT_ROOT:-${OUTPUT_ROOT:-$(pwd)/outputs/4.8/4.8.2}}"
SUMMARY_DIR="${SUMMARY_DIR:-$RESULT_ROOT/summary}"

python "$(dirname "$0")/scripts/summarize_4_8_2.py" \
  --result-root "$RESULT_ROOT" --output-dir "$SUMMARY_DIR" "$@"
