"""Run official FAST on each dataset's audited, fixed input grid."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


ADDITION_ROOT = Path(__file__).resolve().parent
PAPER4_ROOT = ADDITION_ROOT.parent
sys.path.insert(0, str(PAPER4_ROOT / "src"))
sys.path.insert(0, str(PAPER4_ROOT))
sys.path.insert(0, str(ADDITION_ROOT))

import run_official_baselines as base
from common import require_cuda_device
from eeg_pipeline.data.splits import build_loso_subject_folds
from fast_native_adapter import build_fast_native_model


base.build_official_model = build_fast_native_model
SUPPORTED_DATASETS = {"ds003626", "directional_words", "bimodal_inner_speech"}


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"seed", "data", "split", "input_adapter", "models", "training"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing config sections: {sorted(missing)}")
    dataset = config["data"].get("dataset")
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset!r}")
    if set(config["models"]) != {"fast"}:
        raise ValueError("Native-rate rerun config must contain FAST only")
    fast = config["models"]["fast"]
    if int(fast["dim_token"]) != 32:
        raise ValueError("FAST dim_token must be 32 for the official-script setting")
    inputs = config["input_adapter"]
    if int(inputs["source_sampling_rate_hz"]) != int(inputs["target_sampling_rate_hz"]):
        raise ValueError(
            "FAST input adapter must receive the dataset's effective fixed rate"
        )
    subjects = tuple(config["data"]["subjects"])
    expected = list(range(len(subjects)))
    configured = [int(value) for value in config["split"]["fold_indices"]]
    if configured != expected:
        raise ValueError(f"All folds must be listed exactly once: {configured} != {expected}")
    if not bool(config["split"].get("adapt_held_out_subjects", False)):
        raise ValueError("Paper4 calibrated LOSO must remain enabled")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=("fast",), default=("fast",))
    parser.add_argument("--folds", nargs="+", type=int, help="1-based folds")
    parser.add_argument("--plan", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    fold_indices = base.selected_folds(config, args.folds)
    folds = build_loso_subject_folds(tuple(config["data"]["subjects"]))
    print(
        f"dataset={config['data']['dataset']} protocol=calibrated_loso "
        f"model=fast effective_rate={config['input_adapter']['source_sampling_rate_hz']} "
        f"dim_token=32 folds={[index + 1 for index in fold_indices]}",
        flush=True,
    )
    if args.plan:
        for index in fold_indices:
            fold = folds[index]
            print(
                f"fold={index + 1:02d} val_subject={fold.val_subject} "
                f"test_subject={fold.test_subject}"
            )
        return
    device = require_cuda_device(str(config["training"]["device"]))
    for index in fold_indices:
        base.run_one(config, "fast", folds[index], device)


if __name__ == "__main__":
    main()
