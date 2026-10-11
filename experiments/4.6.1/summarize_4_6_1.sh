#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_ROOT="${RESULT_ROOT:?Set RESULT_ROOT to the Section 4.6.1 result directory.}"
arguments=(--result-root "$RESULT_ROOT")
[[ -n "${SUMMARY_OUTPUT:-}" ]] && arguments+=(--output "$SUMMARY_OUTPUT")
python -u "$SCRIPT_DIR/scripts/summarize_4_6_1.py" "${arguments[@]}" "$@"
