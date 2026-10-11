#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_ROOT="${RESULT_ROOT:?Set RESULT_ROOT to the directory containing imagine/.}"

arguments=(--result-root "$RESULT_ROOT")
if [[ -n "${SUMMARY_OUTPUT:-}" ]]; then
  arguments+=(--output "$SUMMARY_OUTPUT")
fi

python -u "$SCRIPT_DIR/scripts/summarize_4_5.py" "${arguments[@]}" "$@"
