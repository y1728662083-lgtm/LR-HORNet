"""Run EEGMamba on the additional calibrated-LOSO datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml


ADDITION_ROOT = Path(__file__).resolve().parent
PAPER4_ROOT = ADDITION_ROOT.parent
sys.path.insert(0, str(PAPER4_ROOT / "src"))
sys.path.insert(0, str(PAPER4_ROOT))
sys.path.insert(0, str(ADDITION_ROOT))

import run_official_baselines as base
from adapters import build_official_model
from common import require_cuda_device
from eeg_pipeline.data.splits import build_loso_subject_folds


SUPPORTED_DATASETS = {
    "directional_words",
    "bimodal_inner_speech",
    "bci2020",
}

# Reuse the split-manifest, resume and prediction implementation.
base.build_official_model = build_official_model


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"seed", "data", "split", "input_adapter", "models", "training"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Configuration is missing sections: {sorted(missing)}")
    dataset_name = config["data"].get("dataset")
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset {dataset_name!r}; expected {sorted(SUPPORTED_DATASETS)}"
        )
    subjects = tuple(config["data"]["subjects"])
    configured_folds = [int(value) for value in config["split"]["fold_indices"]]
    expected_folds = list(range(len(subjects)))
    if configured_folds != expected_folds:
        raise ValueError(
            "All subject folds must be enabled exactly once: "
            f"configured={configured_folds}, expected={expected_folds}"
        )
    if not bool(config["split"].get("adapt_held_out_subjects", False)):
        raise ValueError("This experiment requires Paper4 calibrated LOSO")
    if config["input_adapter"].get("patch_remainder") != "zero_pad":
        raise ValueError("Only the audited zero_pad patch remainder is supported")
    source_rate = int(config["input_adapter"]["source_sampling_rate_hz"])
    if source_rate <= 0:
        raise ValueError("source_sampling_rate_hz must be positive")
    if dataset_name == "directional_words":
        languages = {str(subject).split("/", 1)[0].casefold() for subject in subjects}
        if len(languages) != 1:
            raise ValueError(
                "Russian and Spanish must use separate calibrated-LOSO configurations"
            )
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=("eegmamba",))
    parser.add_argument("--folds", nargs="+", type=int, help="1-based fold numbers")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Write summaries after this invocation (dispatcher summarizes once globally).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    models = base.selected_models(config, args.models)
    fold_indices = base.selected_folds(config, args.folds)
    folds = build_loso_subject_folds(tuple(config["data"]["subjects"]))
    jobs = [(model, index) for index in fold_indices for model in models]
    print(
        f"dataset={config['data']['dataset']} protocol=calibrated_loso "
        f"models={models} folds={[index + 1 for index in fold_indices]} "
        f"fits={len(jobs)} output_root={config['training']['output_root']}",
        flush=True,
    )
    if args.plan:
        for sequence, (model, index) in enumerate(jobs, start=1):
            fold = folds[index]
            print(
                f"{sequence:03d}/{len(jobs):03d} model={model} "
                f"fold={index + 1:02d} val_subject={fold.val_subject} "
                f"test_subject={fold.test_subject}"
            )
        return

    device = require_cuda_device(str(config["training"]["device"]))
    completed = skipped = 0
    for model_name, fold_index in jobs:
        status = base.run_one(config, model_name, folds[fold_index], device)
        completed += status == "completed"
        skipped += status == "skipped"
    if args.summarize:
        from summarize_results import summarize

        summarize(
            Path(config["training"]["output_root"]),
            expected_folds=len(folds),
        )
    print(
        f"FINISHED completed_now={completed} resumed_skips={skipped} "
        f"total={len(jobs)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
