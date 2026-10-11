#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${1:-configs/braindecode_baselines.yaml}"
RUNNER="src/eeg_pipeline/models/baseline_model.py"
SUMMARY="src/eeg_pipeline/utils/summary.py"

if ! OUTPUT_ROOT="$($PYTHON_BIN "$RUNNER" \
    --config "$CONFIG_PATH" \
    --print-output-root)"; then
    printf 'Failed to resolve output_root from %s\n' "$CONFIG_PATH" >&2
    exit 2
fi
mkdir -p "$OUTPUT_ROOT/_status" || exit 2
if ! $PYTHON_BIN "$RUNNER" \
    --config "$CONFIG_PATH" \
    --write-catalog "$OUTPUT_ROOT/_status/catalog.json"; then
    printf 'Failed to write the current-run model catalog.\n' >&2
    exit 2
fi

if ! MODEL_LIST="$($PYTHON_BIN "$RUNNER" \
    --config "$CONFIG_PATH" \
    --list-runnable)"; then
    printf 'Failed to enumerate the official classifier catalog.\n' >&2
    exit 2
fi
mapfile -t MODELS <<< "$MODEL_LIST"
overall_status=0

for architecture in "${MODELS[@]}"; do
    printf '\n[baseline] architecture=%s\n' "$architecture"
    if $PYTHON_BIN "$RUNNER" \
        --config "$CONFIG_PATH" \
        --architecture "$architecture"; then
        printf '[baseline] completed=%s\n' "$architecture"
    else
        printf '[baseline] failed=%s (continuing)\n' "$architecture" >&2
        overall_status=1
    fi
done

if ! $PYTHON_BIN "$SUMMARY" \
    --output-root "$OUTPUT_ROOT" \
    --output "$OUTPUT_ROOT/braindecode_baseline_summary.csv"; then
    printf 'Failed to build baseline summary.\n' >&2
    overall_status=1
fi

printf '\nAll official Braindecode classification candidates were attempted.\n'
printf 'Summary: %s\n' "$OUTPUT_ROOT/braindecode_baseline_summary.csv"
exit "$overall_status"
