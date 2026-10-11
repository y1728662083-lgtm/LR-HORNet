from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eeg_pipeline.models.baseline_model import (  # noqa: E402
    BASELINE_SPECS,
    build_braindecode_baseline,
)
from eeg_pipeline.utils.summary import build_summary  # noqa: E402


def test_catalog_matches_braindecode_181_classification_registry() -> None:
    from braindecode.models.util import (
        models_mandatory_parameters,
        non_classification_models,
    )

    official = {
        name
        for name, _, _ in models_mandatory_parameters
        if name not in set(non_classification_models)
    }
    configured = {spec.architecture for spec in BASELINE_SPECS}
    assert configured == official
    assert len(configured) == 58
    assert {
        spec.architecture for spec in BASELINE_SPECS if spec.paper_year == 2025
    } >= {"CBraMod", "CodeBrain", "EEGDINO", "PBT", "TCFormer"}
    assert {
        spec.architecture for spec in BASELINE_SPECS if spec.paper_year == 2026
    } == {"MVPFormer", "STEEGFormer", "ZUNA"}


def test_eegnet_builder_returns_the_unwrapped_official_class() -> None:
    from braindecode.models import EEGNet

    model = build_braindecode_baseline(
        n_classes=4,
        architecture="EEGNet",
        n_channels=128,
        n_times=1153,
        sfreq=256.0,
        n_electrode_groups=16,
    )
    assert type(model) is EEGNet
    model.eval()
    with torch.inference_mode():
        assert model(torch.zeros(2, 128, 1153)).shape == (2, 4)


def test_baseline_yaml_is_full_loso_and_requested_training_setup() -> None:
    config_path = PROJECT_ROOT / "configs" / "braindecode_baselines.yaml"
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    assert config["split"]["fold_indices"] == list(range(10))
    assert config["loader"]["batch_size"] == 128
    assert config["training"]["epochs"] == 100
    assert config["training"]["patience"] == 40
    assert config["training"]["scheduler"] == {
        "name": "reduce_on_plateau",
        "monitor": "loss",
        "factor": 0.5,
        "patience": 10,
        "min_learning_rate": 0.00001,
    }


def _write_fold(root: Path, model_key: str, fold: int) -> None:
    fold_dir = root / "imagine" / model_key / f"fold_{fold:02d}"
    fold_dir.mkdir(parents=True)
    with (fold_dir / "training_log.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("epoch", "val_accuracy", "is_best"),
        )
        writer.writeheader()
        writer.writerow({"epoch": 1, "val_accuracy": 0.2, "is_best": 1})
        writer.writerow(
            {
                "epoch": fold + 1,
                "val_accuracy": fold / 10,
                "is_best": 1,
            }
        )
    with (fold_dir / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump({"accuracy": (fold + 1) / 20}, file)


def test_summary_records_parameters_all_folds_and_aggregates(
    tmp_path: Path,
) -> None:
    model_key = "braindecode_eegnet"
    status_dir = tmp_path / "_status"
    status_dir.mkdir()
    catalog = {
        "task": "imagine",
        "expected_folds": list(range(1, 11)),
        "models": [
            {
                "architecture": "EEGNet",
                "model_key": model_key,
                "paper_year": None,
                "runnable": True,
                "reason": "",
            }
        ],
    }
    (status_dir / "catalog.json").write_text(
        json.dumps(catalog), encoding="utf-8"
    )
    status = {
        "status": "completed",
        "task": "imagine",
        "expected_folds": list(range(1, 11)),
        "started_at_utc": "2000-01-01T00:00:00+00:00",
        "total_parameters": 1234,
        "trainable_parameters": 1200,
    }
    (status_dir / f"{model_key}.json").write_text(
        json.dumps(status), encoding="utf-8"
    )
    for fold in range(1, 11):
        _write_fold(tmp_path, model_key, fold)

    output_path = tmp_path / "summary.csv"
    rows = build_summary(tmp_path, output_path)

    row = rows[0]
    assert row["status"] == "completed"
    assert row["expected_folds"] == 10
    assert row["completed_folds"] == 10
    assert row["total_parameters"] == 1234
    assert row["trainable_parameters"] == 1200
    assert row["fold_01_best_epoch"] == 2
    assert row["fold_10_best_epoch"] == 11
    assert float(row["best_val_accuracy_mean"]) == pytest.approx(0.55)
    assert float(row["test_accuracy_at_best_val_mean"]) == pytest.approx(
        0.325
    )
    with output_path.open(encoding="utf-8-sig", newline="") as file:
        saved = next(csv.DictReader(file))
    assert "fold_10_test_accuracy_at_best_val" in saved
    assert saved["total_parameters"] == "1234"


def test_summary_recovers_legacy_fold_outputs_without_model_status(
    tmp_path: Path,
) -> None:
    model_key = "braindecode_eegnet"
    status_dir = tmp_path / "_status"
    status_dir.mkdir()
    catalog = {
        "task": "imagine",
        "expected_folds": [1, 2],
        "models": [
            {
                "architecture": "EEGNet",
                "model_key": model_key,
                "paper_year": None,
                "runnable": True,
                "reason": "",
            }
        ],
    }
    (status_dir / "catalog.json").write_text(
        json.dumps(catalog), encoding="utf-8"
    )
    _write_fold(tmp_path, model_key, 1)
    _write_fold(tmp_path, model_key, 2)

    rows = build_summary(tmp_path, tmp_path / "summary.csv")

    row = rows[0]
    assert row["status"] == "completed"
    assert row["completed_folds"] == 2
    assert row["reason"] == ""
    assert row["fold_01_best_epoch"] == 2
    assert row["fold_02_best_epoch"] == 3
