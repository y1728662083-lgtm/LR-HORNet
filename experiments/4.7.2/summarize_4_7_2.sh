#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VISUAL_DIAGNOSTICS="${VISUAL_DIAGNOSTICS:-$SCRIPT_DIR/reference_results/group_04_visualization/offset_trend_diagnostics.csv}"
MECHANISM_ROOT="${MECHANISM_ROOT:-$SCRIPT_DIR/reference_results/group_06_offset_mechanism_rank01_04}"
RANK5_ARCHIVE="${RANK5_ARCHIVE:-$SCRIPT_DIR/reference_results/rank05_incomplete_archive}"
SUMMARY_DIR="${SUMMARY_DIR:-$SCRIPT_DIR/summary}"
python -u "$SCRIPT_DIR/scripts/summarize_4_7_2.py" \
  --visual-diagnostics "$VISUAL_DIAGNOSTICS" --mechanism-root "$MECHANISM_ROOT" \
  --rank5-archive "$RANK5_ARCHIVE" --output-dir "$SUMMARY_DIR" "$@"
