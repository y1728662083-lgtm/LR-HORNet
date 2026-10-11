#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_ROOT="${RESULT_ROOT:?Set RESULT_ROOT to the directory containing stored 4.7.1 results.}"
SUMMARY_DIR="${SUMMARY_DIR:-$SCRIPT_DIR/summary}"
python -u "$SCRIPT_DIR/scripts/summarize_4_7_1.py" \
  --result-root "$RESULT_ROOT" --output-dir "$SUMMARY_DIR" "$@"
