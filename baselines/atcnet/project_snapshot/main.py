"""Main entry point for the EEG classification pipeline."""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler, ReduceLROnPlateau
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_pipeline.data.frequency import CachedFilterBankDataset
from eeg_pipeline.data.loaders import build_dataloaders
from eeg_pipeline.data.splits import (
    SubjectFold,
    build_adapted_fold_datasets,
    build_loso_subject_folds,
)


def _dataset_attribute(dataset: Dataset, name: str):
    """Read one invariant metadata attribute through dataset wrappers."""
    if hasattr(dataset, name):
        return getattr(dataset, name)
    if isinstance(dataset, Subset):
        return _dataset_attribute(dataset.dataset, name)
    if isinstance(dataset, ConcatDataset):
        values = [
            _dataset_attribute(item, name)
            for item in dataset.datasets
        ]
        reference = values[0]
        if any(value != reference for value in values[1:]):
            raise ValueError(
                f"Dataset components disagree on metadata {name!r}: {values}"
            )
        return reference
    wrapped = getattr(dataset, "dataset", None)
    if wrapped is not None and wrapped is not dataset:
        return _dataset_attribute(wrapped, name)
    raise AttributeError(
        f"Dataset {type(dataset).__name__} does not expose {name!r}"
    )


def load_config(config_path: str | Path) -> dict:
    """Load a YAML configuration and check its top-level sections."""
    config_path = Path(config_path)
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")

    required_sections = {"seed", "data", "split", "loader", "model", "training"}
    missing_sections = required_sections - config.keys()
    if missing_sections:
        raise ValueError(
            f"Missing configuration sections: {sorted(missing_sections)}"
        )

    return config


def set_random_seed(seed: int) -> None:
    """Set Python, NumPy and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_output_dir(config: dict, fold_index: int) -> Path:
    """Build the task/model/fold output path."""
    output_root = Path(config["training"]["output_root"])
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    task = config["data"]["task"]
    model_name = config["model"]["name"]
    dataset_name = config["data"].get("dataset", "ds003626")
    experiment_name = config["data"].get("experiment_name", dataset_name)
    if dataset_name == "ds003626":
        return output_root / task / model_name / f"fold_{fold_index + 1:02d}"
    return (
        output_root
        / experiment_name
        / task
        / model_name
        / f"fold_{fold_index + 1:02d}"
    )


def build_fold_loaders(
    config: dict,
    fold: SubjectFold,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, validation and test loaders for one fold."""
    data_config = config["data"]
    train_dataset, val_dataset, test_dataset = build_adapted_fold_datasets(
        dataset_root=data_config["root"],
        task=data_config["task"],
        fold=fold,
        sessions=data_config["sessions"],
        dataset_name=data_config.get("dataset", "ds003626"),
        dataset_options=data_config.get("options"),
        seed=config["seed"],
        standardize=data_config["standardize"],
        n_electrode_groups=data_config["n_electrode_groups"],
        adapt_held_out_subjects=config["split"].get(
            "adapt_held_out_subjects",
            True,
        ),
    )

    tfc_config = config["training"].get("tfc")
    if tfc_config is not None and tfc_config.get("enabled", False):
        model_kwargs = config["model"].get("kwargs", {})
        bands_hz = tfc_config["bands_hz"]
        expected_bands = model_kwargs.get("n_frequency_bands", len(bands_hz))
        if len(bands_hz) != expected_bands:
            raise ValueError(
                "training.tfc.bands_hz count must equal "
                "model.kwargs.n_frequency_bands"
            )
        actual_sampling_rate_hz = float(
            _dataset_attribute(train_dataset, "sampling_rate_hz")
        )
        configured_sampling_rate_hz = tfc_config.get("sampling_rate_hz")
        if configured_sampling_rate_hz is not None and not math.isclose(
            float(configured_sampling_rate_hz),
            actual_sampling_rate_hz,
        ):
            raise ValueError(
                "training.tfc.sampling_rate_hz does not match the selected "
                f"dataset: configured={configured_sampling_rate_hz}, "
                f"actual={actual_sampling_rate_hz}"
            )
        train_dataset = CachedFilterBankDataset(
            train_dataset,
            sampling_rate_hz=actual_sampling_rate_hz,
            bands_hz=bands_hz,
            patch_size=model_kwargs["patch_size"],
            patch_stride=model_kwargs["patch_stride"],
            filter_order=tfc_config.get("filter_order", 4),
            cache_dtype=tfc_config.get("cache_dtype", "float16"),
        )
        print(
            "training frequency cache enabled: "
            f"bands={len(bands_hz)}, "
            f"patch_size={model_kwargs['patch_size']}, "
            f"patch_stride={model_kwargs['patch_stride']}, "
            f"cache_dtype={tfc_config.get('cache_dtype', 'float16')}"
        )

    loader_config = config["loader"]
    return build_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=loader_config["batch_size"],
        seed=config["seed"],
        pin_memory=loader_config["pin_memory"],
    )


def build_scheduler(
    optimizer: Optimizer,
    scheduler_config: dict | None,
    *,
    total_epochs: int,
) -> LRScheduler | ReduceLROnPlateau | None:
    """Build an optional metric-based or epoch-based LR scheduler."""
    if scheduler_config is None:
        return None

    scheduler_name = scheduler_config.get("name")
    if scheduler_name == "reduce_on_plateau":
        if scheduler_config.get("monitor") != "loss":
            raise ValueError(
                "reduce_on_plateau scheduler must monitor validation loss"
            )
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_config["factor"],
            patience=scheduler_config["patience"],
            min_lr=scheduler_config["min_learning_rate"],
        )

    if scheduler_name == "cosine_warmup":
        warmup_epochs = scheduler_config.get("warmup_epochs")
        if isinstance(warmup_epochs, bool) or not isinstance(warmup_epochs, int):
            raise ValueError("scheduler.warmup_epochs must be an integer")
        if not 1 <= warmup_epochs < total_epochs:
            raise ValueError(
                "scheduler.warmup_epochs must satisfy "
                "1 <= warmup_epochs < training.epochs"
            )

        start_factor = scheduler_config.get("warmup_start_factor", 0.1)
        if isinstance(start_factor, bool):
            raise ValueError(
                "scheduler.warmup_start_factor must be in (0, 1]"
            )
        try:
            start_factor = float(start_factor)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "scheduler.warmup_start_factor must be in (0, 1]"
            ) from error
        if not math.isfinite(start_factor) or not 0.0 < start_factor <= 1.0:
            raise ValueError(
                "scheduler.warmup_start_factor must be in (0, 1]"
            )

        min_learning_rate = scheduler_config.get("min_learning_rate", 0.0)
        if isinstance(min_learning_rate, bool):
            raise ValueError(
                "scheduler.min_learning_rate must be a finite non-negative number"
            )
        try:
            min_learning_rate = float(min_learning_rate)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "scheduler.min_learning_rate must be a finite non-negative number"
            ) from error
        base_learning_rates = [
            float(group["lr"]) for group in optimizer.param_groups
        ]
        if (
            not math.isfinite(min_learning_rate)
            or min_learning_rate < 0.0
            or any(
                min_learning_rate > base_learning_rate
                for base_learning_rate in base_learning_rates
            )
        ):
            raise ValueError(
                "scheduler.min_learning_rate must be finite, non-negative, "
                "and no greater than every optimizer learning rate"
            )

        cosine_epochs = total_epochs - warmup_epochs

        def make_lr_lambda(base_learning_rate: float):
            minimum_factor = (
                min_learning_rate / base_learning_rate
                if base_learning_rate > 0.0
                else 0.0
            )

            def lr_lambda(epoch_index: int) -> float:
                if epoch_index < warmup_epochs:
                    warmup_progress = epoch_index / warmup_epochs
                    return start_factor + (1.0 - start_factor) * warmup_progress

                cosine_progress = min(
                    (epoch_index - warmup_epochs) / cosine_epochs,
                    1.0,
                )
                cosine_factor = 0.5 * (
                    1.0 + math.cos(math.pi * cosine_progress)
                )
                return minimum_factor + (
                    1.0 - minimum_factor
                ) * cosine_factor

            return lr_lambda

        return LambdaLR(
            optimizer,
            lr_lambda=[
                make_lr_lambda(base_learning_rate)
                for base_learning_rate in base_learning_rates
            ],
        )

    raise ValueError(f"Unknown scheduler: {scheduler_name!r}")


def build_training_components(
    config: dict,
    dataset: Dataset | None = None,
) -> tuple[
    nn.Module,
    nn.Module,
    Optimizer,
    LRScheduler | ReduceLROnPlateau | None,
    torch.device,
    int,
]:
    """Build the model, loss, optimizer, scheduler, device and class count."""
    from eeg_pipeline.models.factory import build_model

    task = config["data"]["task"]
    if dataset is None:
        task_classes = {"imagine": 4, "speak": 4, "visual": 4, "pattern": 3}
        if task not in task_classes:
            raise ValueError(
                "Dataset metadata is required for non-ds003626 task "
                f"{task!r}"
            )
        n_classes = task_classes[task]
        sample_shape = None
    else:
        class_names = tuple(_dataset_attribute(dataset, "class_names"))
        if len(class_names) < 2:
            raise ValueError(
                f"Classification requires at least two classes: {class_names}"
            )
        n_classes = len(class_names)
        sample = dataset[0]
        if not isinstance(sample, (tuple, list)) or len(sample) < 2:
            raise TypeError("EEG dataset samples must contain EEG and label")
        sample_shape = tuple(sample[0].shape)
        if len(sample_shape) != 2:
            raise ValueError(
                f"Expected EEG sample shape [channels, time], got {sample_shape}"
            )

    training_config = config["training"]
    requested_device = str(training_config["device"])
    if requested_device == "auto":
        requested_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "cuda" and (device.index or 0) >= torch.cuda.device_count():
        raise ValueError(
            f"Requested {device}, but only {torch.cuda.device_count()} CUDA "
            "device(s) are visible to this process. Check CUDA_VISIBLE_DEVICES."
        )
    print(
        f"training_device={device} "
        f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}"
    )

    if training_config["loss"] != "cross_entropy":
        raise ValueError(f"Unknown loss: {training_config['loss']!r}")
    if training_config["optimizer"] != "adamw":
        raise ValueError(f"Unknown optimizer: {training_config['optimizer']!r}")

    label_smoothing_value = training_config.get("label_smoothing", 0.0)
    if isinstance(label_smoothing_value, bool):
        raise ValueError("training.label_smoothing must be in [0, 1].")
    try:
        label_smoothing = float(label_smoothing_value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "training.label_smoothing must be a finite number in [0, 1]."
        ) from error
    if not math.isfinite(label_smoothing) or not 0.0 <= label_smoothing <= 1.0:
        raise ValueError("training.label_smoothing must be in [0, 1].")

    model_config = config["model"]
    model_kwargs = dict(model_config.get("kwargs", {}))
    if sample_shape is not None:
        inferred_dimensions = {
            "n_channels": sample_shape[0],
            "n_times": sample_shape[1],
        }
        for key, inferred_value in inferred_dimensions.items():
            configured_value = model_kwargs.get(key)
            if (
                configured_value is not None
                and int(configured_value) != inferred_value
            ):
                raise ValueError(
                    f"model.kwargs.{key}={configured_value} does not match "
                    f"the selected dataset ({inferred_value})"
                )
            model_kwargs[key] = inferred_value
    model = build_model(
        name=model_config["name"],
        n_classes=n_classes,
        **model_kwargs,
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )
    scheduler = build_scheduler(
        optimizer,
        training_config.get("scheduler"),
        total_epochs=training_config["epochs"],
    )
    return model, criterion, optimizer, scheduler, device, n_classes


def run_fold(
    config: dict,
    fold: SubjectFold,
) -> dict[str, float | list[list[int]]]:
    """Train and evaluate one LOSO fold."""
    from eeg_pipeline.evaluation.evaluator import (
        evaluate_model,
        save_evaluation_result,
    )
    from eeg_pipeline.training.engine import fit_model

    train_loader, val_loader, test_loader = build_fold_loaders(config, fold)
    model, criterion, optimizer, scheduler, device, n_classes = (
        build_training_components(config, train_loader.dataset)
    )
    class_names = tuple(
        str(name)
        for name in _dataset_attribute(train_loader.dataset, "class_names")
    )
    if len(class_names) != n_classes:
        raise ValueError(
            "Dataset class names do not match the model output count: "
            f"{len(class_names)} != {n_classes}"
        )
    eval_criterion = nn.CrossEntropyLoss()
    output_dir = build_output_dir(config, fold.fold_index)

    print(
        f"fold={fold.fold_index + 1:02d} "
        f"train_subjects={fold.train_subjects} "
        f"val_subject={fold.val_subject} test_subject={fold.test_subject}"
    )
    print(
        f"trials: train={len(train_loader.dataset)}, "
        f"val={len(val_loader.dataset)}, test={len(test_loader.dataset)}"
    )
    print(
        f"model={config['model']['name']}, n_classes={n_classes}, "
        f"output_dir={output_dir}"
    )
    print(
        "label_to_class: "
        + ", ".join(
            f"{label}={class_name!r}"
            for label, class_name in enumerate(class_names)
        )
    )

    training_config = config["training"]
    _, best_epoch = fit_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        output_dir=output_dir,
        eval_criterion=eval_criterion,
        scheduler=scheduler,
        contrastive_config=training_config.get("tfc"),
        epochs=training_config["epochs"],
        patience=training_config["patience"],
        monitor=training_config["monitor"],
    )
    result = evaluate_model(model, test_loader, eval_criterion, device)
    result_path = save_evaluation_result(result, output_dir)

    print(
        f"fold={fold.fold_index + 1:02d} best_epoch={best_epoch} "
        f"test_loss={result['loss']:.6f} "
        f"test_accuracy={result['accuracy']:.4f}"
    )
    print(f"test metrics saved to: {result_path}")
    return result


def run_experiment(config: dict) -> list[dict]:
    """Run the LOSO folds selected in the configuration."""
    fold_indices = config["split"]["fold_indices"]
    if len(set(fold_indices)) != len(fold_indices):
        raise ValueError(f"fold_indices contains duplicates: {fold_indices}")

    folds = build_loso_subject_folds(config["data"]["subjects"])
    invalid_indices = [
        index for index in fold_indices if not 0 <= index < len(folds)
    ]
    if invalid_indices:
        raise ValueError(f"Invalid fold indices: {invalid_indices}")

    val_subject = config["split"].get("val_subject")
    if val_subject is not None:
        if len(fold_indices) != 1:
            raise ValueError("split.val_subject requires exactly one fold")
        if val_subject not in config["data"]["subjects"]:
            raise ValueError("split.val_subject is not in data.subjects")
        fold_index = fold_indices[0]
        original = folds[fold_index]
        if val_subject == original.test_subject:
            raise ValueError("Validation and test subjects must differ")
        train_subjects = tuple(
            subject for subject in config["data"]["subjects"]
            if subject not in (original.test_subject, val_subject)
        )
        folds = tuple(
            SubjectFold(fold_index, train_subjects, val_subject, original.test_subject)
            if index == fold_index else fold
            for index, fold in enumerate(folds)
        )
        # Preserve old paths for all pre-existing configurations.
        root = Path(config["training"]["output_root"])
        config = dict(config)
        config["training"] = dict(config["training"])
        config["training"]["output_root"] = str(root / f"val_subject_{val_subject:02d}")

    print(
        f"dataset={config['data'].get('dataset', 'ds003626')}, "
        f"task={config['data']['task']}, model={config['model']['name']}, "
        f"folds={fold_indices}"
    )
    results = []
    for fold_index in fold_indices:
        set_random_seed(config["seed"])
        result = run_fold(config, folds[fold_index])
        results.append({"fold": fold_index + 1, **result})

    print(f"experiment finished: folds_run={len(results)}")
    return results


def parse_args() -> argparse.Namespace:
    """Parse the experiment configuration path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "deformable_patchCfC.yaml",
        help=(
            "YAML configuration path; relative paths are resolved from the "
            "project root."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = load_config(config_path)
    run_experiment(config)


