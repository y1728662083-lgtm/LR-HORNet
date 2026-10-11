"""Run dataset preprocessing before starting paper4 training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_pipeline.data.preprocessing import preprocess_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Training YAML containing data and preprocessing sections.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and list outputs without processing EEG.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or not isinstance(config.get("data"), dict):
        raise ValueError("Config must contain a data mapping")

    outputs = preprocess_from_config(config, dry_run=args.dry_run)
    action = "planned" if args.dry_run else "ready"
    print(
        f"dataset={config['data'].get('dataset', 'ds003626')} "
        f"outputs_{action}={len(outputs)}"
    )
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
