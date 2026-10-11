#!/usr/bin/env bash
set -uo pipefail

if (( $# != 2 )); then
  printf 'Usage: %s <dataset-key> <config-path>\n' "$0" >&2
  exit 2
fi

DATASET_KEY="$1"
CONFIG_RELATIVE="$2"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="$PROJECT_ROOT/$CONFIG_RELATIVE"
RUNNER="$PROJECT_ROOT/src/eeg_pipeline/models/baseline_model.py"
SUMMARY_RUNNER="$PROJECT_ROOT/src/eeg_pipeline/utils/summary.py"
SUMMARY_OUTPUT="$PROJECT_ROOT/anxiliary/summary/${DATASET_KEY}_baseline_summary.csv"

if ! "$PYTHON_BIN" "$PROJECT_ROOT/anxiliary/prepare_official_archives.py"; then
  printf 'Failed to prepare bundled official preprocessing archives.\n' >&2
  exit 2
fi

printf '\n[preprocessing] dataset=%s config=%s\n' "$DATASET_KEY" "$CONFIG_PATH"
if ! "$PYTHON_BIN" -u "$PROJECT_ROOT/preprocessing.py" --config "$CONFIG_PATH"; then
  printf '[preprocessing failed] dataset=%s\n' "$DATASET_KEY" >&2
  exit 2
fi

if ! OUTPUT_ROOT="$($PYTHON_BIN "$RUNNER" --config "$CONFIG_PATH" --print-output-root)"; then
  printf 'Failed to resolve output root for %s\n' "$DATASET_KEY" >&2
  exit 2
fi
mkdir -p "$OUTPUT_ROOT/_status" "$OUTPUT_ROOT/_logs" "$(dirname "$SUMMARY_OUTPUT")"

if ! "$PYTHON_BIN" "$RUNNER" \
  --config "$CONFIG_PATH" \
  --write-catalog "$OUTPUT_ROOT/_status/catalog.json"; then
  printf 'Failed to write model catalog for %s\n' "$DATASET_KEY" >&2
  exit 2
fi

if ! MODEL_LIST="$($PYTHON_BIN "$RUNNER" --config "$CONFIG_PATH" --list-runnable)"; then
  printf 'Failed to enumerate baseline models for %s\n' "$DATASET_KEY" >&2
  exit 2
fi
mapfile -t MODELS <<< "$MODEL_LIST"
overall_status=0

for architecture in "${MODELS[@]}"; do
  log_path="$OUTPUT_ROOT/_logs/${architecture}.log"
  printf '\n[baseline] dataset=%s architecture=%s\n' "$DATASET_KEY" "$architecture"
  {
    printf 'run_started_utc=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    printf 'dataset=%s\n' "$DATASET_KEY"
    printf 'architecture=%s\n' "$architecture"
    printf 'config_path=%s\n' "$CONFIG_PATH"
    printf '%s\n' '===== YAML hyperparameters ====='
    cat "$CONFIG_PATH"
    printf '%s\n' '===== training output ====='
    "$PYTHON_BIN" -u "$RUNNER" \
      --config "$CONFIG_PATH" \
      --architecture "$architecture"
  } 2>&1 | tee "$log_path"
  run_status=${PIPESTATUS[0]}
  if (( run_status != 0 )); then
    overall_status=1
    printf '[failed:%d] dataset=%s architecture=%s (continuing)\n' \
      "$run_status" "$DATASET_KEY" "$architecture" >&2
  else
    printf '[completed] dataset=%s architecture=%s\n' \
      "$DATASET_KEY" "$architecture"
  fi
done

if ! "$PYTHON_BIN" "$SUMMARY_RUNNER" \
  --output-root "$OUTPUT_ROOT" \
  --output "$SUMMARY_OUTPUT"; then
  overall_status=1
  printf 'Failed to build summary for %s\n' "$DATASET_KEY" >&2
fi

printf '\nDataset=%s summary=%s\n' "$DATASET_KEY" "$SUMMARY_OUTPUT"
exit "$overall_status"
