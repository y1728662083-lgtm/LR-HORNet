#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/../OpenNeruro_ds003626}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/4.7/4.7.2/group_06_offset_mechanism_rank01_04}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-}"
RANKS="${RANKS:-1,2,3,4}"
export PROJECT_ROOT DATA_ROOT CPU_THREADS="${CPU_THREADS:-8}" BATCH_SIZE="${BATCH_SIZE:-8}" MAX_TRIALS="${MAX_TRIALS:-0}"

lookup_checkpoint() {
  local rank="$1" pattern variable path
  variable="CHECKPOINT_0${rank}"
  path="${!variable:-}"
  if [[ -n "$path" ]]; then printf '%s\n' "$path"; return; fi
  if [[ -z "$CHECKPOINT_ROOT" ]]; then
    echo "Set CHECKPOINT_ROOT or $variable." >&2; return 2
  fi
  case "$rank" in
    1) pattern='*patch_0512*repeat_01_seed_42*best_model.pt' ;;
    2) pattern='*patch_0512*repeat_05_seed_42*best_model.pt' ;;
    3) pattern='*patch_0008*repeat_01_seed_42*best_model.pt' ;;
    4) pattern='*patch_0032*repeat_01_seed_42*best_model.pt' ;;
    5) pattern='*patch_0064*repeat_04_seed_42*best_model.pt' ;;
    *) echo "Unknown rank: $rank" >&2; return 2 ;;
  esac
  path="$(find "$CHECKPOINT_ROOT" -type f -path "$pattern" -print -quit)"
  if [[ -z "$path" ]]; then echo "No checkpoint matched $pattern under $CHECKPOINT_ROOT" >&2; return 2; fi
  printf '%s\n' "$path"
}

rank_fields() {
  case "$1" in
    1) echo 'rank01_patch0512_repeat01_test042 rank01_patch0512_repeat01_test042.yaml' ;;
    2) echo 'rank02_patch0512_repeat05_test041 rank02_patch0512_repeat05_test041.yaml' ;;
    3) echo 'rank03_patch0008_repeat01_test034 rank03_patch0008_repeat01_test034.yaml' ;;
    4) echo 'rank04_patch0032_repeat01_test034 rank04_patch0032_repeat01_test034.yaml' ;;
    5) echo 'rank05_patch0064_repeat04_test034 rank05_patch0064_repeat04_test034.yaml' ;;
    *) return 2 ;;
  esac
}

IFS=',' read -ra selected <<< "$RANKS"
for rank in "${selected[@]}"; do
  rank="${rank//[[:space:]]/}"
  read -r name config_name <<< "$(rank_fields "$rank")"
  config="$SCRIPT_DIR/configs/group06_offset_mechanism/$config_name"
  checkpoint="$(lookup_checkpoint "$rank")"
  output="$OUTPUT_ROOT/$name"
  if [[ -f "$output/causal_ablation_summary.csv" && -f "$output/artificial_shift_summary.csv" ]]; then
    echo "SKIP complete: $name"; continue
  fi
  if [[ -d "$output" && -n "$(find "$output" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Partial output exists; move it aside before retry: $output" >&2; exit 3
  fi
  CONFIG_PATH="$config" CHECKPOINT_PATH="$checkpoint" OUTPUT_DIR="$output" \
    bash "$SCRIPT_DIR/run_offset_mechanism.sh"
done
