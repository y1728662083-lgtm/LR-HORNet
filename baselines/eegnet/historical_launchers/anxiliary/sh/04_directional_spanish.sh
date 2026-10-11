#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/_run_dataset.sh" \
  "directional_words_spanish" \
  "anxiliary/yaml/directional_spanish_covert_words.yaml"
