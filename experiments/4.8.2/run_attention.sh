#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATA_ROOT="${DATA_ROOT:-$(dirname "$PROJECT_ROOT")/OpenNeruro_ds003626}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/4.8/4.8.2}"
DEVICE="${DEVICE:-cuda}"
ATTENTION_VARIANTS="${ATTENTION_VARIANTS:-satt,statt}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"

if [[ -n "${GPU_ID:-}" ]]; then export CUDA_VISIBLE_DEVICES="$GPU_ID"; fi

python "$(dirname "$0")/scripts/run_4_8_2.py" \
  --project-root "$PROJECT_ROOT" --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" \
  --device "$DEVICE" --family attention --attention-variants "$ATTENTION_VARIANTS" \
  --shard-index "$SHARD_INDEX" --shard-count "$SHARD_COUNT" "$@"
