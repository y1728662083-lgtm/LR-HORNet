#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/_run_dataset.sh" \
  "bimodal_inner_speech" \
  "anxiliary/yaml/bimodal_inner_speech.yaml"
