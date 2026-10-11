#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATA_ROOT="${DATA_ROOT:-$(dirname "$PROJECT_ROOT")/OpenNeruro_ds003626}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/4.8/4.8.1}"
DEVICE="${DEVICE:-auto}"
MODELS="${MODELS:-group4_patchLstm_deformableLstm,group4_patchGRU_deformableGRU,group4_patchRNN_deformableRNN,group4_patchLstm_deformableCfC,group4_patchCfC_deformableCfC}"
REPEATS="${REPEATS:-1-5}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"

if [[ -n "${GPU_ID:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

python "$(dirname "$0")/scripts/run_4_8_1.py" \
  --project-root "$PROJECT_ROOT" \
  --data-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --device "$DEVICE" \
  --models "$MODELS" \
  --repeats "$REPEATS" \
  --shard-index "$SHARD_INDEX" \
  --shard-count "$SHARD_COUNT" \
  "$@"
