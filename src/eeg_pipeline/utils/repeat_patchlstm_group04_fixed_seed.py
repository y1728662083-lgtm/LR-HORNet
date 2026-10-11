"""Repeat one four-group Patch-LSTM family fold with the same seed."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from eeg_pipeline.utils.repeat_single_fold import (
    RUN_FIELDS,
    _atomic_write_csv,
    _load_config,
    _mean_and_sample_std,
    _read_completed_metrics,
    _resolve_from_project,
    _select_by_validation,
    _stream_process,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPECTED_MODELS = {
    "causal1_group8patchlstm_eegnet",
    "group4_patchLstm_deformableLstm",
    "group4_patchLstm_deformableCfC",
}
PARAMETER_LINE = re.compile(
    r"parameters:\s*total=([\d,]+),\s*trainable=([\d,]+)"
)


def _read_parameter_counts(log_path: Path) -> tuple[int, int] | None:
    """Read counts emitted after main.py builds from actual dataset metadata."""
    try:
        with log_path.open(encoding="utf-8", errors="replace") as file:
            for line in file:
                match = PARAMETER_LINE.search(line)
                if match is not None:
                    return tuple(
                        int(value.replace(",", ""))
                        for value in match.groups()
                    )
    except OSError:
        return None
    return None


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--fold-index",
        type=int,
        required=True,
        help="Zero-based fold index; displayed fold number is index + 1.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Optional output directory for this complete repeat set.",
    )
    return parser.parse_args()


def _validate(
    config: dict[str, Any],
    *,
    fold_index: int,
    seed: int,
    repeats: int,
) -> None:
    model = config.get("model", {})
    model_name = model.get("name")
    if model_name not in EXPECTED_MODELS:
        raise ValueError(
            "This runner requires one of the four-group Patch-LSTM family "
            f"models {sorted(EXPECTED_MODELS)}; got {model_name!r}."
        )
    kwargs = model.get("kwargs", {}) or {}
    expected_signature = {
        "data_n_electrode_groups": 4,
        "n_electrode_groups": 4,
        "patch_size": 128,
        "patch_stride": 64,
        "label_smoothing": 0.05,
    }
    actual_signature = {
        "data_n_electrode_groups": config.get("data", {}).get(
            "n_electrode_groups"
        ),
        "n_electrode_groups": kwargs.get("n_electrode_groups"),
        "patch_size": kwargs.get("patch_size"),
        "patch_stride": kwargs.get("patch_stride"),
        "label_smoothing": config.get("training", {}).get(
            "label_smoothing"
        ),
    }
    if actual_signature != expected_signature:
        raise ValueError(
            "The YAML is not the expected PatchLSTM group04 configuration. "
            f"Expected {expected_signature}, got {actual_signature}."
        )
    batch_size = config.get("loader", {}).get("batch_size")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("loader.batch_size must be a positive integer.")
    subjects = config.get("data", {}).get("subjects")
    if not isinstance(subjects, list) or len(subjects) < 3:
        raise ValueError("data.subjects must contain at least three subjects.")
    if isinstance(fold_index, bool) or not 0 <= fold_index < len(subjects):
        raise ValueError(
            f"fold-index must be in [0, {len(subjects) - 1}]."
        )
    if isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32).")
    if isinstance(repeats, bool) or repeats < 1:
        raise ValueError("repeats must be a positive integer.")
    if config.get("training", {}).get("monitor") != "accuracy":
        raise ValueError(
            "training.monitor must be accuracy so checkpoint selection uses "
            "the validation set."
        )


def _output_directory(
    repeat_root: Path,
    config: dict[str, Any],
    *,
    model_name: str,
    fold_number: int,
) -> Path:
    """Mirror main.py's dataset-aware output-directory convention."""
    data = config["data"]
    dataset_name = str(data.get("dataset", "ds003626"))
    experiment_name = str(data.get("experiment_name", dataset_name))
    root = repeat_root
    if dataset_name != "ds003626":
        root = root / experiment_name
    return (
        root
        / str(data["task"])
        / model_name
        / f"fold_{fold_number:02d}"
    )


def _write_reports(
    run_root: Path,
    records: list[dict[str, Any]],
    *,
    model_name: str,
    fold_number: int,
    seed: int,
    repeats: int,
) -> None:
    _select_by_validation(records)
    _atomic_write_csv(run_root / "repeat_runs.csv", RUN_FIELDS, records)

    completed = [row for row in records if row["status"] == "completed"]
    val_values = [float(row["best_val_accuracy"]) for row in completed]
    test_values = [
        float(row["test_accuracy_at_best_val"]) for row in completed
    ]
    val_mean, val_std = _mean_and_sample_std(val_values)
    test_mean, test_std = _mean_and_sample_std(test_values)
    selected = next(
        (row for row in records if row["selected_by_validation"]), None
    )
    max_val = (
        max(completed, key=lambda row: float(row["best_val_accuracy"]))
        if completed
        else None
    )
    max_test = (
        max(
            completed,
            key=lambda row: float(row["test_accuracy_at_best_val"]),
        )
        if completed
        else None
    )
    failed_count = sum(row["status"] == "failed" for row in records)
    pending_count = repeats - len(completed) - failed_count
    summary = {
        "model": model_name,
        "fold_number": fold_number,
        "fixed_seed": seed,
        "requested_runs": repeats,
        "completed_runs": len(completed),
        "failed_runs": failed_count,
        "pending_runs": pending_count,
        "best_val_accuracy_mean": val_mean,
        "best_val_accuracy_sample_std": val_std,
        "test_accuracy_at_best_val_mean": test_mean,
        "test_accuracy_at_best_val_sample_std": test_std,
        "max_best_val_accuracy": (
            max_val["best_val_accuracy"] if max_val else ""
        ),
        "max_best_val_repeat": max_val["repeat"] if max_val else "",
        "diagnostic_max_test_accuracy": (
            max_test["test_accuracy_at_best_val"] if max_test else ""
        ),
        "diagnostic_max_test_repeat": (
            max_test["repeat"] if max_test else ""
        ),
        "selected_repeat_by_validation": (
            selected["repeat"] if selected else ""
        ),
        "selected_best_val_accuracy": (
            selected["best_val_accuracy"] if selected else ""
        ),
        "selected_test_accuracy_at_best_val": (
            selected["test_accuracy_at_best_val"] if selected else ""
        ),
        "selection_rule": (
            "highest validation accuracy; test maximum is diagnostic only"
        ),
    }
    _atomic_write_csv(run_root / "repeat_summary.csv", summary.keys(), [summary])


def main() -> int:
    args = _parse_args()
    config_path = _resolve_from_project(args.config)
    base_config = _load_config(config_path)
    _validate(
        base_config,
        fold_index=args.fold_index,
        seed=args.seed,
        repeats=args.repeats,
    )
    model_name = base_config["model"]["name"]
    fold_number = args.fold_index + 1

    if args.run_root is None:
        base_output_root = Path(base_config["training"]["output_root"])
        if not base_output_root.is_absolute():
            base_output_root = PROJECT_ROOT / base_output_root
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_root = (
            base_output_root
            / "six_fixed_seed_repeats"
            / f"fold_{fold_number:02d}_seed_{args.seed}_{run_tag}"
        )
    else:
        run_root = _resolve_from_project(args.run_root)
    # A stable run root makes long experiments resumable after interruption.
    # Completed repeat directories are validated and skipped below.
    run_root.mkdir(parents=True, exist_ok=True)

    parameter_counts: tuple[int, int] | None = None
    manifest = {
        "experiment_type": "same-fold same-seed repeated training",
        "source_config": str(config_path),
        "source_config_sha256": hashlib.sha256(
            config_path.read_bytes()
        ).hexdigest(),
        "model": model_name,
        "fold_index": args.fold_index,
        "fold_number": fold_number,
        "repeat_count": args.repeats,
        "fixed_seed": args.seed,
        "total_parameters": None,
        "trainable_parameters": None,
        "parameter_source": "main.py model built from actual adapted dataset",
        "selection_rule": (
            "select by validation only; test maximum is diagnostic only"
        ),
    }
    manifest_path = run_root / "manifest.json"
    _write_json_atomic(manifest_path, manifest)

    print(f"repeat_set={run_root}")
    print(
        f"fold={fold_number:02d}, fixed_seed={args.seed}, "
        f"repeats={args.repeats}"
    )
    print("parameters: deferred until main.py builds the actual dataset model")

    records: list[dict[str, Any]] = []
    for repeat in range(1, args.repeats + 1):
        repeat_root = (
            run_root / f"repeat_{repeat:02d}_seed_{args.seed}"
        )
        resolved_config = copy.deepcopy(base_config)
        resolved_config["seed"] = args.seed
        resolved_config["split"]["fold_indices"] = [args.fold_index]
        resolved_config["training"]["output_root"] = str(repeat_root)
        generated_config = (
            run_root / "configs" / f"repeat_{repeat:02d}_seed_{args.seed}.yaml"
        )
        generated_config.parent.mkdir(parents=True, exist_ok=True)
        with generated_config.open("w", encoding="utf-8", newline="\n") as file:
            yaml.safe_dump(
                resolved_config,
                file,
                sort_keys=False,
                allow_unicode=True,
            )

        output_dir = _output_directory(
            repeat_root,
            resolved_config,
            model_name=model_name,
            fold_number=fold_number,
        )
        log_path = (
            run_root / "logs" / f"repeat_{repeat:02d}_seed_{args.seed}.log"
        )
        record: dict[str, Any] = {
            "repeat": repeat,
            "seed": args.seed,
            "fold_index": args.fold_index,
            "fold_number": fold_number,
            "status": "running",
            "exit_code": "",
            "elapsed_seconds": "",
            "total_parameters": (
                parameter_counts[0] if parameter_counts is not None else ""
            ),
            "trainable_parameters": (
                parameter_counts[1] if parameter_counts is not None else ""
            ),
            "best_epoch": "",
            "best_val_accuracy": "",
            "val_loss_at_best": "",
            "test_accuracy_at_best_val": "",
            "test_loss_at_best_val": "",
            "selected_by_validation": False,
            "output_dir": str(output_dir),
            "config_path": str(generated_config),
            "log_path": str(log_path),
            "error": "",
        }
        existing_counts = _read_parameter_counts(log_path)
        if existing_counts is not None:
            parameter_counts = existing_counts
            record["total_parameters"], record["trainable_parameters"] = (
                existing_counts
            )
            for previous_record in records:
                previous_record["total_parameters"] = existing_counts[0]
                previous_record["trainable_parameters"] = existing_counts[1]
            manifest["total_parameters"] = existing_counts[0]
            manifest["trainable_parameters"] = existing_counts[1]
            _write_json_atomic(manifest_path, manifest)
        try:
            record.update(_read_completed_metrics(output_dir))
            record["status"] = "completed"
            record["exit_code"] = 0
            records.append(record)
            _write_reports(
                run_root,
                records,
                model_name=model_name,
                fold_number=fold_number,
                seed=args.seed,
                repeats=args.repeats,
            )
            print(
                f"\n[resume fixed-seed repeat {repeat}/{args.repeats}] "
                f"seed={args.seed} fold={fold_number:02d}"
            )
            continue
        except (
            OSError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            pass

        records.append(record)
        _write_reports(
            run_root,
            records,
            model_name=model_name,
            fold_number=fold_number,
            seed=args.seed,
            repeats=args.repeats,
        )

        print(
            f"\n[fixed-seed repeat {repeat}/{args.repeats}] "
            f"seed={args.seed} fold={fold_number:02d}"
        )
        started = time.monotonic()
        exit_code = _stream_process(
            [
                sys.executable,
                "-u",
                str(PROJECT_ROOT / "main.py"),
                "--config",
                str(generated_config),
            ],
            log_path,
        )
        observed_counts = _read_parameter_counts(log_path)
        if observed_counts is not None:
            parameter_counts = observed_counts
            for current_record in records:
                current_record["total_parameters"] = observed_counts[0]
                current_record["trainable_parameters"] = observed_counts[1]
            manifest["total_parameters"] = observed_counts[0]
            manifest["trainable_parameters"] = observed_counts[1]
            _write_json_atomic(manifest_path, manifest)
        record["elapsed_seconds"] = round(time.monotonic() - started, 3)
        record["exit_code"] = exit_code
        if exit_code != 0:
            record["status"] = "failed"
            record["error"] = f"main.py exited with status {exit_code}"
        else:
            try:
                record.update(_read_completed_metrics(output_dir))
                record["status"] = "completed"
            except (
                OSError,
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as error:
                record["status"] = "failed"
                record["error"] = str(error)
        _write_reports(
            run_root,
            records,
            model_name=model_name,
            fold_number=fold_number,
            seed=args.seed,
            repeats=args.repeats,
        )

    completed = [row for row in records if row["status"] == "completed"]
    print(f"\ncompleted={len(completed)}/{args.repeats}")
    print(f"runs_csv={run_root / 'repeat_runs.csv'}")
    print(f"summary_csv={run_root / 'repeat_summary.csv'}")
    if completed:
        max_test = max(
            completed,
            key=lambda row: float(row["test_accuracy_at_best_val"]),
        )
        print(
            "diagnostic_max_test_accuracy="
            f"{float(max_test['test_accuracy_at_best_val']):.4f}, "
            f"repeat={max_test['repeat']}"
        )
        selected = next(
            row for row in records if row["selected_by_validation"]
        )
        print(
            "selected_by_validation: "
            f"repeat={selected['repeat']}, "
            f"best_val_accuracy={float(selected['best_val_accuracy']):.4f}, "
            "corresponding_test_accuracy="
            f"{float(selected['test_accuracy_at_best_val']):.4f}"
        )
    return 0 if len(completed) == args.repeats else 1


if __name__ == "__main__":
    raise SystemExit(main())
