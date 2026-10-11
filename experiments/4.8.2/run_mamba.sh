#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATA_ROOT="${DATA_ROOT:-$(dirname "$PROJECT_ROOT")/OpenNeruro_ds003626}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/4.8/4.8.2}"
DEVICE="${DEVICE:-auto}"
MAMBA_VARIANTS="${MAMBA_VARIANTS:-full_mamba_patch_replaced,mamba_offset_only}"
MAMBA_MODE="${MAMBA_MODE:-all}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"

if [[ -n "${GPU_ID:-}" ]]; then export CUDA_VISIBLE_DEVICES="$GPU_ID"; fi

python "$(dirname "$0")/scripts/run_4_8_2.py" \
  --project-root "$PROJECT_ROOT" --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" \
  --device "$DEVICE" --family mamba --mamba-variants "$MAMBA_VARIANTS" \
  --mamba-mode "$MAMBA_MODE" --shard-index "$SHARD_INDEX" --shard-count "$SHARD_COUNT" "$@"
