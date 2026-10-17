"""Train LaBraM with the calibrated ds003626 LOSO protocol."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset


ADDITION_ROOT = Path(__file__).resolve().parent
PAPER4_ROOT = ADDITION_ROOT.parent
REPOSITORY_ROOT = ADDITION_ROOT.parents[2]
sys.path.insert(0, str(PAPER4_ROOT / "src"))
sys.path.insert(0, str(PAPER4_ROOT))
sys.path.insert(0, str(ADDITION_ROOT))
sys.path.append(str(REPOSITORY_ROOT))

from portable_paths import resolve_config_data_root
from adapters import build_official_model
from data_cache import materialize_splits
from common import (
    count_parameters,
    require_cuda_device,
    resolve_from_addition,
    set_seed,
    sha256_file,
    stable_hash,
    write_json,
)
from eeg_pipeline.data.splits import (
    SubjectFold,
    build_adapted_fold_datasets,
    build_loso_subject_folds,
)
from training import fit_model, predict_and_score


_RUNTIME_CONFIGURED = False


def configure_runtime(config: Mapping[str, Any]) -> None:
    global _RUNTIME_CONFIGURED
    if _RUNTIME_CONFIGURED:
        return
    cpu_threads = int(config["training"].get("cpu_threads_per_process", 4))
    if cpu_threads <= 0:
        raise ValueError("training.cpu_threads_per_process must be positive")
    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    _RUNTIME_CONFIGURED = True
    print(
        f"runtime cpu_threads={torch.get_num_threads()} "
        f"interop_threads={torch.get_num_interop_threads()} "
        f"ram_cache={bool(config['training'].get('cache_preprocessed_trials', False))}",
        flush=True,
    )


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    resolve_config_data_root(config, repository_root=REPOSITORY_ROOT)
    required = {"seed", "data", "split", "input_adapter", "models", "training"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Configuration is missing sections: {sorted(missing)}")
    if config["data"].get("dataset") != "ds003626":
        raise ValueError("This runner is intentionally restricted to ds003626")
    if config["input_adapter"].get("patch_remainder") != "zero_pad":
        raise ValueError("Only the audited zero_pad patch remainder policy is supported")
    return config


def dataset_attribute(dataset: Dataset, name: str):
    if hasattr(dataset, name):
        return getattr(dataset, name)
    if isinstance(dataset, Subset):
        return dataset_attribute(dataset.dataset, name)
    if isinstance(dataset, ConcatDataset):
        values = [dataset_attribute(item, name) for item in dataset.datasets]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"Dataset components disagree on {name!r}")
        return values[0]
    wrapped = getattr(dataset, "dataset", None)
    if wrapped is not None and wrapped is not dataset:
        return dataset_attribute(wrapped, name)
    raise AttributeError(f"{type(dataset).__name__} has no attribute {name!r}")


def selected_models(config: Mapping[str, Any], requested: Iterable[str] | None) -> list[str]:
    enabled = [name for name, value in config["models"].items() if value.get("enabled", True)]
    if requested is None:
        return enabled
    requested = list(requested)
    invalid = set(requested) - set(config["models"])
    if invalid:
        raise ValueError(f"Unknown models: {sorted(invalid)}")
    disabled = [name for name in requested if name not in enabled]
    if disabled:
        raise ValueError(f"Requested models are disabled in YAML: {disabled}")
    return requested


def selected_folds(config: Mapping[str, Any], requested: Iterable[int] | None) -> list[int]:
    configured = [int(value) for value in config["split"]["fold_indices"]]
    if requested is None:
        return configured
    zero_based = [int(value) - 1 for value in requested]
    invalid = [value + 1 for value in zero_based if value not in configured]
    if invalid:
        raise ValueError(f"Fold numbers are not enabled in YAML: {invalid}")
    return zero_based


def build_split_manifest(
    fold: SubjectFold,
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    *,
    seed: int,
    calibrated: bool,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "protocol": "calibrated_loso" if calibrated else "strict_loso",
        "fold_index_zero_based": int(fold.fold_index),
        "fold_number_one_based": int(fold.fold_index + 1),
        "train_subjects": list(fold.train_subjects),
        "validation_subject": fold.val_subject,
        "test_subject": fold.test_subject,
        "split_seed": int(seed),
        "test_calibration_seed": int(seed + 1),
        "counts": {
            "train": len(train_dataset),
            "validation": len(val_dataset),
            "test": len(test_dataset),
        },
    }
    if calibrated:
        if not isinstance(train_dataset, ConcatDataset) or len(train_dataset.datasets) != 3:
            raise TypeError("Calibrated Paper4 train dataset should contain three components")
        val_calibration = train_dataset.datasets[1]
        test_calibration = train_dataset.datasets[2]
        if not all(
            isinstance(item, Subset)
            for item in (val_calibration, test_calibration, val_dataset, test_dataset)
        ):
            raise TypeError("Expected stratified calibration/evaluation Subset objects")
        manifest.update(
            validation_calibration_indices=list(map(int, val_calibration.indices)),
            validation_evaluation_indices=list(map(int, val_dataset.indices)),
            test_calibration_indices=list(map(int, test_calibration.indices)),
            test_evaluation_indices=list(map(int, test_dataset.indices)),
        )
        for prefix in ("validation", "test"):
            calibration = set(manifest[f"{prefix}_calibration_indices"])
            evaluation = set(manifest[f"{prefix}_evaluation_indices"])
            if calibration & evaluation:
                raise RuntimeError(f"{prefix} calibration/evaluation overlap")
            full_size = len(val_dataset.dataset if prefix == "validation" else test_dataset.dataset)
            if calibration | evaluation != set(range(full_size)):
                raise RuntimeError(f"{prefix} calibration/evaluation coverage mismatch")
    manifest["manifest_sha256"] = stable_hash(manifest)
    return manifest


def make_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    *,
    batch_size: int,
    eval_batch_size: int,
    seed: int,
    pin_memory: bool,
    train_drop_last: bool,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    generator = torch.Generator().manual_seed(seed)
    return (
        DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
            pin_memory=pin_memory,
            drop_last=train_drop_last,
        ),
        DataLoader(
            val_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
        ),
        DataLoader(
            test_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
        ),
    )


def fingerprint_payload(
    config: Mapping[str, Any],
    model_name: str,
    fold_index: int,
) -> dict[str, Any]:
    model_config = config["models"][model_name]
    payload: dict[str, Any] = {
        "seed": config["seed"],
        "data": config["data"],
        "split": config["split"],
        "input_adapter": config["input_adapter"],
        "training": config["training"],
        "model_name": model_name,
        "model": model_config,
        "fold_index": fold_index,
    }
    checkpoint = resolve_from_addition(model_config.get("checkpoint"))
    payload["checkpoint_sha256"] = (
        sha256_file(checkpoint) if checkpoint is not None and checkpoint.is_file() else None
    )
    manifest_path = ADDITION_ROOT / "source_manifest.json"
    payload["source_manifest_sha256"] = sha256_file(manifest_path)
    return payload


def result_is_resumable(output_dir: Path, fingerprint: str) -> bool:
    metadata_path = output_dir / "run_metadata.json"
    metrics_path = output_dir / "test_metrics.json"
    predictions_path = output_dir / "test_predictions.csv"
    checkpoint_path = output_dir / "best_model.pt"
    if not all(path.is_file() for path in (metadata_path, metrics_path, predictions_path, checkpoint_path)):
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete":
        return False
    if metadata.get("run_fingerprint") != fingerprint:
        raise RuntimeError(
            f"Existing completed output has a different fingerprint: {output_dir}. "
            "Use a different output_root or remove/move that exact fold directory deliberately."
        )
    return True


def save_prediction_rows(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    model_name: str,
    fold: SubjectFold,
    test_dataset: Dataset,
    class_names: tuple[str, ...],
) -> None:
    test_indices = (
        list(map(int, test_dataset.indices))
        if isinstance(test_dataset, Subset)
        else list(range(len(test_dataset)))
    )
    if len(test_indices) != len(rows):
        raise RuntimeError("Prediction count does not match test subset indices")
    enriched: list[dict[str, Any]] = []
    for row, subject_index in zip(rows, test_indices, strict=True):
        item = {
            "model": model_name,
            "fold": fold.fold_index + 1,
            "test_subject": fold.test_subject,
            "heldout_subject_trial_index": subject_index,
            **row,
            "target_name": class_names[int(row["target"])],
            "prediction_name": class_names[int(row["prediction"])],
        }
        enriched.append(item)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(enriched[0]))
        writer.writeheader()
        writer.writerows(enriched)


def run_one(
    config: Mapping[str, Any],
    model_name: str,
    fold: SubjectFold,
    device: torch.device,
) -> str:
    configure_runtime(config)
    model_config = config["models"][model_name]
    training_config = dict(model_config["training"])
    if "layer_decay" in model_config:
        training_config["layer_decay"] = model_config["layer_decay"]
    output_root = Path(config["training"]["output_root"])
    output_dir = output_root / model_name / f"fold_{fold.fold_index + 1:02d}"
    fingerprint = stable_hash(
        fingerprint_payload(config, model_name, fold.fold_index)
    )
    if result_is_resumable(output_dir, fingerprint):
        print(f"SKIP model={model_name} fold={fold.fold_index + 1:02d} (matching complete result)")
        return "skipped"

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "run_metadata.json",
        {"status": "running", "run_fingerprint": fingerprint},
    )
    try:
        training_seed = int(training_config.get("seed", config["seed"]))
        deterministic = bool(training_config.get("deterministic", True))
        set_seed(training_seed, deterministic=deterministic)
        torch.backends.cudnn.benchmark = bool(training_config.get("cudnn_benchmark", False))
        data = config["data"]
        train_dataset, val_dataset, test_dataset = build_adapted_fold_datasets(
            dataset_root=data["root"],
            task=data["task"],
            fold=fold,
            sessions=data["sessions"],
            dataset_name=data["dataset"],
            dataset_options=data.get("options"),
            seed=int(config["seed"]),
            standardize=bool(data["standardize"]),
            n_electrode_groups=int(data["n_electrode_groups"]),
            adapt_held_out_subjects=bool(config["split"]["adapt_held_out_subjects"]),
        )
        split_manifest = build_split_manifest(
            fold, train_dataset, val_dataset, test_dataset,
            seed=int(config["seed"]),
            calibrated=bool(config["split"]["adapt_held_out_subjects"]),
        )
        channel_names = tuple(dataset_attribute(train_dataset, "channel_names"))
        class_names = tuple(str(value) for value in dataset_attribute(train_dataset, "class_names"))
        split_manifest.pop("manifest_sha256", None)
        split_manifest["channel_policy"] = (
            "within_language_eeg_intersection"
            if data["dataset"] == "directional_words"
            and bool(data.get("options", {}).get("harmonize_within_language", False))
            else "dataset_native"
        )
        split_manifest["n_channels"] = len(channel_names)
        split_manifest["channel_names"] = list(channel_names)
        split_manifest["manifest_sha256"] = stable_hash(split_manifest)
        write_json(output_dir / "split_manifest.json", split_manifest)
        loader_train = train_dataset
        loader_val = val_dataset
        loader_test = test_dataset
        cache_enabled = bool(config["training"].get("cache_preprocessed_trials", False))
        if cache_enabled:
            loader_train, loader_val, loader_test = materialize_splits(
                train_dataset, val_dataset, test_dataset
            )
        sample_eeg, _ = loader_train[0]
        model = build_official_model(
            model_name,
            model_config,
            channel_names=channel_names,
            n_times=sample_eeg.shape[-1],
            n_classes=len(class_names),
            input_config=config["input_adapter"],
        ).to(device)
        parameters = count_parameters(model)
        batch_size = int(training_config["batch_size"])
        eval_batch_size = max(
            1,
            int(batch_size * float(training_config.get("eval_batch_size_multiplier", 1.0))),
        )
        train_loader, val_loader, test_loader = make_loaders(
            loader_train, loader_val, loader_test,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            seed=training_seed,
            pin_memory=bool(config["training"].get("pin_memory", True)),
            train_drop_last=bool(training_config.get("train_drop_last", False)),
        )
        print(
            f"START model={model_name} fold={fold.fold_index + 1:02d} "
            f"test_subject={fold.test_subject} train/val/test="
            f"{len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)} "
            f"params={parameters['total']:,}",
            flush=True,
        )
        resolved = {
            "model": model_name,
            "model_config": model_config,
            "training_config": training_config,
            "input_adapter": config["input_adapter"],
            "channel_names": list(channel_names),
            "class_names": list(class_names),
            "parameters": parameters,
            "adapter_metadata": getattr(model, "adapter_metadata", {}),
            "checkpoint_report": getattr(model, "checkpoint_report", {}),
            "runtime": {
                "training_seed": training_seed,
                "physical_batch_size": batch_size,
                "eval_batch_size": eval_batch_size,
                "cache_preprocessed_trials": cache_enabled,
                "cpu_threads_per_process": int(
                    config["training"].get("cpu_threads_per_process", 0)
                ),
            },
        }
        with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(resolved, handle, allow_unicode=True, sort_keys=False)
        write_json(output_dir / "checkpoint_report.json", resolved["checkpoint_report"])

        training_result = fit_model(
            model_name, model, train_loader, val_loader, device,
            training_config, output_dir,
            monitor=str(config["training"].get("monitor", "accuracy")),
        )
        metrics, prediction_rows = predict_and_score(
            model, test_loader, device,
            amp=bool(training_config.get("amp", True)),
        )
        metrics.update(
            model=model_name,
            fold=fold.fold_index + 1,
            test_subject=fold.test_subject,
            best_epoch=training_result["best_epoch"],
            best_validation_metric=training_result["best_validation_metric"],
            parameters_total=parameters["total"],
            parameters_trainable=parameters["trainable"],
        )
        write_json(output_dir / "test_metrics.json", metrics)
        save_prediction_rows(
            output_dir / "test_predictions.csv", prediction_rows,
            model_name=model_name, fold=fold, test_dataset=test_dataset,
            class_names=class_names,
        )
        write_json(
            output_dir / "run_metadata.json",
            {
                "status": "complete",
                "run_fingerprint": fingerprint,
                "model": model_name,
                "fold": fold.fold_index + 1,
                "test_subject": fold.test_subject,
                "training": training_result,
                "split_manifest_sha256": split_manifest["manifest_sha256"],
                "checkpoint_report": getattr(model, "checkpoint_report", {}),
            },
        )
        print(
            f"DONE model={model_name} fold={fold.fold_index + 1:02d} "
            f"test_acc={metrics['accuracy']:.4f} "
            f"test_bacc={metrics['balanced_accuracy']:.4f}",
            flush=True,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return "completed"
    except Exception as error:
        write_json(
            output_dir / "failure.json",
            {
                "model": model_name,
                "fold": fold.fold_index + 1,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        write_json(
            output_dir / "run_metadata.json",
            {
                "status": "failed",
                "run_fingerprint": fingerprint,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=ADDITION_ROOT / "configs" / "ds003626_official_baselines.yaml",
    )
    parser.add_argument("--models", nargs="+", choices=("labram",))
    parser.add_argument("--folds", nargs="+", type=int, help="1-based fold numbers")
    parser.add_argument("--plan", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    models = selected_models(config, args.models)
    fold_indices = selected_folds(config, args.folds)
    folds = build_loso_subject_folds(tuple(config["data"]["subjects"]))
    if len(folds) != 10:
        raise RuntimeError(f"Expected 10 ds003626 folds, got {len(folds)}")
    jobs = [(model, index) for model in models for index in fold_indices]
    print(
        f"protocol=calibrated_loso models={models} folds={[i + 1 for i in fold_indices]} "
        f"fits={len(jobs)} output_root={config['training']['output_root']}"
    )
    if args.plan:
        for sequence, (model, index) in enumerate(jobs, start=1):
            fold = folds[index]
            print(
                f"{sequence:02d}/{len(jobs):02d} model={model} fold={index + 1:02d} "
                f"val_subject={fold.val_subject} test_subject={fold.test_subject}"
            )
        return

    device = require_cuda_device(str(config["training"]["device"]))
    completed = skipped = 0
    for model_name, fold_index in jobs:
        status = run_one(config, model_name, folds[fold_index], device)
        completed += status == "completed"
        skipped += status == "skipped"
    from summarize_results import summarize

    summarize(Path(config["training"]["output_root"]), expected_folds=10)
    print(f"FINISHED completed_now={completed} resumed_skips={skipped} total={len(jobs)}")


if __name__ == "__main__":
    main()
