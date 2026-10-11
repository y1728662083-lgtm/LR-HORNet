"""Training and evaluation for one EEG model epoch."""

from __future__ import annotations

import csv
import math
import time
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Literal

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device | str,
    optimizer: Optimizer | None = None,
    *,
    contrastive_config: Mapping | None = None,
    return_components: bool = False,
) -> tuple[float, float] | tuple[float, float, dict[str, float]]:
    """Run one training or evaluation epoch and return loss and accuracy."""
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_ce_loss = 0.0
    total_tfc_loss = 0.0
    total_time_loss = 0.0
    total_frequency_loss = 0.0
    total_cross_domain_loss = 0.0
    total_memory_loss = 0.0
    total_correct = 0
    total_trials = 0
    tfc_enabled = bool(
        is_training
        and contrastive_config is not None
        and contrastive_config.get("enabled", False)
    )
    if tfc_enabled:
        from eeg_pipeline.training.tfc import (
            augment_frequency,
            augment_time,
            compute_tfc_losses,
            sample_position_indices,
        )

        lambda_nce = float(contrastive_config.get("lambda_nce", 0.1))
        if not math.isfinite(lambda_nce) or lambda_nce < 0.0:
            raise ValueError("training.tfc.lambda_nce must be non-negative")

    with torch.set_grad_enabled(is_training):
        for batch in dataloader:
            if not isinstance(batch, (tuple, list)) or len(batch) not in (2, 3):
                raise TypeError(
                    "dataloader batches must be (eeg, labels) or "
                    "(eeg, labels, frequency)"
                )
            eeg, labels = batch[:2]
            frequency = batch[2] if len(batch) == 3 else None
            eeg = eeg.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            if tfc_enabled:
                if frequency is None:
                    raise RuntimeError(
                        "TF-C training requires cached frequency targets"
                    )
                if not hasattr(model, "forward_contrastive"):
                    raise TypeError(
                        "TF-C model must implement forward_contrastive"
                    )
                frequency = frequency.to(
                    device,
                    dtype=eeg.dtype,
                    non_blocking=True,
                )
                augmented_eeg = augment_time(
                    eeg,
                    jitter_std=contrastive_config.get(
                        "time_jitter_std", 0.05
                    ),
                    scaling_std=contrastive_config.get(
                        "time_scaling_std", 0.0
                    ),
                )
                augmented_frequency = augment_frequency(
                    frequency,
                    perturb_ratio=contrastive_config.get(
                        "frequency_perturb_ratio", 0.1
                    ),
                    add_scale=contrastive_config.get(
                        "frequency_add_scale", 0.1
                    ),
                )
                if frequency.ndim != 4:
                    raise ValueError(
                        "TF-C frequency targets must have shape (B,C,K,P)"
                    )
                positions_per_batch = contrastive_config.get(
                    "positions_per_batch", 1024
                )
                position_indices = sample_position_indices(
                    frequency.shape[1] * frequency.shape[3],
                    positions_per_batch,
                    device=eeg.device,
                )
                output = model.forward_contrastive(
                    eeg,
                    augmented_eeg,
                    frequency,
                    augmented_frequency,
                    time_position_indices=position_indices,
                )
                logits = output.logits
                ce_loss = criterion(logits, labels)
                tfc_losses = compute_tfc_losses(
                    h_time=output.h_time,
                    h_time_augmented=output.h_time_augmented,
                    h_frequency=output.h_frequency,
                    h_frequency_augmented=output.h_frequency_augmented,
                    z_time=output.z_time,
                    z_frequency=output.z_frequency,
                    temperature=contrastive_config.get("temperature", 0.2),
                    intra_domain_weight=contrastive_config.get(
                        "intra_domain_weight", 0.2
                    ),
                    positions_per_batch=positions_per_batch,
                    chunk_size=contrastive_config.get("chunk_size", 256),
                    position_indices=position_indices,
                )
                loss = ce_loss + lambda_nce * tfc_losses.total
            else:
                logits = model(eeg)
                ce_loss = criterion(logits, labels)
                zero = ce_loss.detach().new_zeros(())
                tfc_losses = (zero, zero, zero, zero)
                loss = ce_loss

            memory_loss = ce_loss.new_zeros(())
            memory_contribution = ce_loss.new_zeros(())
            if is_training and hasattr(model, "memory_auxiliary_loss"):
                memory_loss = model.memory_auxiliary_loss()
                if not isinstance(memory_loss, torch.Tensor):
                    raise TypeError("memory_auxiliary_loss() must return a Tensor")
                if memory_loss.ndim != 0:
                    raise ValueError("memory_auxiliary_loss() must return a scalar")
                if not torch.isfinite(memory_loss.detach()).item():
                    raise FloatingPointError("memory auxiliary loss is not finite")
                memory_weight = float(
                    getattr(model, "memory_auxiliary_loss_weight", 1.0)
                )
                if not math.isfinite(memory_weight) or memory_weight < 0.0:
                    raise ValueError(
                        "memory_auxiliary_loss_weight must be finite and "
                        "non-negative"
                    )
                memory_contribution = memory_weight * memory_loss
                loss = loss + memory_contribution

            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.detach().item() * batch_size
            total_ce_loss += ce_loss.detach().item() * batch_size
            total_tfc_loss += tfc_losses[0].detach().item() * batch_size
            total_time_loss += tfc_losses[1].detach().item() * batch_size
            total_frequency_loss += tfc_losses[2].detach().item() * batch_size
            total_cross_domain_loss += tfc_losses[3].detach().item() * batch_size
            total_memory_loss += (
                memory_contribution.detach().item() * batch_size
            )
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_trials += batch_size

    if total_trials == 0:
        raise ValueError("dataloader must contain at least one trial")

    loss_value = total_loss / total_trials
    accuracy_value = total_correct / total_trials
    components = {
        "ce": total_ce_loss / total_trials,
        "tfc": total_tfc_loss / total_trials,
        "time": total_time_loss / total_trials,
        "frequency": total_frequency_loss / total_trials,
        "cross_domain": total_cross_domain_loss / total_trials,
        # Weighted contribution included in train_loss, not the raw model term.
        "memory": total_memory_loss / total_trials,
    }
    if return_components:
        return loss_value, accuracy_value, components
    return loss_value, accuracy_value


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device | str,
    *,
    output_dir: str | Path,
    eval_criterion: nn.Module | None = None,
    scheduler: LRScheduler | ReduceLROnPlateau | None = None,
    contrastive_config: Mapping | None = None,
    epochs: int = 100,
    patience: int = 10,
    monitor: Literal["accuracy", "loss"] = "accuracy",
    verbose: bool = True,
) -> tuple[list[dict[str, int | float]], int]:
    """Train, save epoch logs and the best model, then restore it.

    ``criterion`` is used for optimization. ``eval_criterion`` is used for
    validation metrics, allowing label smoothing to remain a training-only
    regularizer. When omitted, the training criterion is reused for backward
    compatibility.
    """
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if patience < 1:
        raise ValueError("patience must be at least 1")
    if monitor not in ("accuracy", "loss"):
        raise ValueError("monitor must be 'accuracy' or 'loss'")
    if eval_criterion is None:
        eval_criterion = criterion

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training_log.csv"
    best_model_path = output_dir / "best_model.pt"

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(
        f"parameters: total={total_parameters:,}, "
        f"trainable={trainable_parameters:,}"
    )

    history: list[dict[str, int | float]] = []
    best_state = None
    best_epoch = 0
    best_val_loss = float("inf")
    best_val_accuracy = float("-inf")
    epochs_without_improvement = 0

    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start_time = time.perf_counter()

    tfc_enabled = bool(
        contrastive_config is not None
        and contrastive_config.get("enabled", False)
    )
    memory_enabled = bool(
        hasattr(model, "memory_auxiliary_loss")
        and float(getattr(model, "memory_auxiliary_loss_weight", 1.0)) != 0.0
    )

    log_fields = [
        "epoch",
        "learning_rate",
        "train_loss",
        "train_accuracy",
    ]
    if tfc_enabled or memory_enabled:
        log_fields.append("train_ce_loss")
    if tfc_enabled:
        log_fields.extend(
            (
                "train_tfc_loss",
                "train_time_loss",
                "train_frequency_loss",
                "train_cross_domain_loss",
            )
        )
    if memory_enabled:
        log_fields.append("train_memory_loss")
    log_fields.extend(
        (
            "val_loss",
            "val_accuracy",
            "is_best",
        )
    )
    with log_path.open("w", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=log_fields)
        writer.writeheader()

        for epoch in range(1, epochs + 1):
            train_loss, train_accuracy, train_components = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer,
                contrastive_config=contrastive_config,
                return_components=True,
            )
            val_loss, val_accuracy = run_epoch(
                model, val_loader, eval_criterion, device
            )
            improved = (
                val_accuracy > best_val_accuracy
                if monitor == "accuracy"
                else val_loss < best_val_loss
            )
            if improved:
                best_val_loss = val_loss
                best_val_accuracy = val_accuracy
                best_epoch = epoch
                best_state = deepcopy(model.state_dict())
                torch.save(best_state, best_model_path)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            record = {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "is_best": int(improved),
            }
            if tfc_enabled or memory_enabled:
                record["train_ce_loss"] = train_components["ce"]
            if tfc_enabled:
                record.update(
                    train_tfc_loss=train_components["tfc"],
                    train_time_loss=train_components["time"],
                    train_frequency_loss=train_components["frequency"],
                    train_cross_domain_loss=train_components["cross_domain"],
                )
            if memory_enabled:
                record["train_memory_loss"] = train_components["memory"]
            history.append(record)
            writer.writerow(record)
            log_file.flush()

            if verbose:
                marker = " *" if improved else ""
                message = [
                    f"epoch={epoch}",
                    f"lr={record['learning_rate']:.6g}",
                    f"train_loss={train_loss:.6f}",
                    f"train_accuracy={train_accuracy:.4f}",
                ]
                if tfc_enabled or memory_enabled:
                    message.append(f"train_ce={train_components['ce']:.6f}")
                if tfc_enabled:
                    message.extend(
                        (
                            f"train_tfc={train_components['tfc']:.6f}",
                            "train_cross="
                            f"{train_components['cross_domain']:.6f}",
                        )
                    )
                if memory_enabled:
                    message.append(
                        f"train_memory={train_components['memory']:.6f}"
                    )
                message.extend(
                    (
                        f"val_loss={val_loss:.6f}",
                        f"val_accuracy={val_accuracy:.4f}{marker}",
                    )
                )
                print(" ".join(message))

            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"early stopping at epoch {epoch}")
                break

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - start_time

    if best_state is None:
        raise RuntimeError("No best model was found during training")
    model.load_state_dict(best_state)

    hours, remainder = divmod(int(elapsed_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    best_name = "val_accuracy" if monitor == "accuracy" else "val_loss"
    best_value = best_val_accuracy if monitor == "accuracy" else best_val_loss
    print(
        f"training finished: epochs_run={len(history)}, "
        f"total_time={hours:02d}:{minutes:02d}:{seconds:02d}"
    )
    print(f"best {best_name}={best_value:.6f} at epoch {best_epoch}")
    print(f"best model saved to: {best_model_path}")
    return history, best_epoch


if __name__ == "__main__":
    import math
    import sys

    from torch.utils.data import Subset, TensorDataset

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from eeg_pipeline.data.dataset import EEGTaskDataset
    from eeg_pipeline.data.ds003626 import load_task_sessions
    from eeg_pipeline.data.loaders import build_dataloaders
    from eeg_pipeline.models.eegnet import build_eegnet

    dataset_root = Path("data/ds003626")
    task_sessions = load_task_sessions(
        dataset_root,
        task="imagine",
        subjects=(1,),
        sessions=(1,),
        preload=False,
    )
    full_dataset = EEGTaskDataset(task_sessions, standardize=True)
    train_dataset = Subset(full_dataset, range(32))
    val_dataset = Subset(full_dataset, range(32, 48))
    train_loader, val_loader, _ = build_dataloaders(
        train_dataset,
        val_dataset,
        val_dataset,
        batch_size=8,
        seed=42,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_eegnet(n_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    before_train = [
        parameter.detach().clone()
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    train_loss, train_accuracy = run_epoch(
        model,
        train_loader,
        criterion,
        device,
        optimizer,
    )
    after_train = [
        parameter.detach().clone()
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if not any(
        not torch.equal(before, after)
        for before, after in zip(before_train, after_train, strict=True)
    ):
        raise RuntimeError("Training epoch did not update model parameters")

    before_val = [parameter.detach().clone() for parameter in model.parameters()]
    val_loss, val_accuracy = run_epoch(
        model,
        val_loader,
        criterion,
        device,
    )
    if any(
        not torch.equal(before, parameter)
        for before, parameter in zip(
            before_val, model.parameters(), strict=True
        )
    ):
        raise RuntimeError("Evaluation epoch changed model parameters")

    for name, loss, accuracy in (
        ("train", train_loss, train_accuracy),
        ("val", val_loss, val_accuracy),
    ):
        if not math.isfinite(loss):
            raise RuntimeError(f"{name} loss is not finite: {loss}")
        if not 0.0 <= accuracy <= 1.0:
            raise RuntimeError(f"{name} accuracy is invalid: {accuracy}")

    print("run_epoch test passed")
    print(f"device={device}")
    print(f"train: loss={train_loss:.6f}, accuracy={train_accuracy:.4f}")
    print(f"val: loss={val_loss:.6f}, accuracy={val_accuracy:.4f}")
    print("training updated parameters; evaluation preserved parameters")

    synthetic_features = torch.randn(8, 4)
    synthetic_labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    synthetic_loader = DataLoader(
        TensorDataset(synthetic_features, synthetic_labels),
        batch_size=4,
        shuffle=False,
    )
    for monitor in ("accuracy", "loss"):
        output_dir = (
            Path(__file__).resolve().parents[3]
            / "outputs"
            / "engine_smoke_test"
            / monitor
        )
        synthetic_model = nn.Linear(4, 2).to(device)
        synthetic_optimizer = torch.optim.SGD(
            synthetic_model.parameters(), lr=0.0
        )
        history, best_epoch = fit_model(
            synthetic_model,
            synthetic_loader,
            synthetic_loader,
            nn.CrossEntropyLoss(),
            synthetic_optimizer,
            device,
            output_dir=output_dir,
            epochs=3,
            patience=2,
            monitor=monitor,
            verbose=False,
        )
        if len(history) != 3 or best_epoch != 1:
            raise RuntimeError(
                f"Unexpected early stopping for monitor={monitor}: "
                f"epochs={len(history)}, best_epoch={best_epoch}"
            )
        log_path = output_dir / "training_log.csv"
        best_model_path = output_dir / "best_model.pt"
        with log_path.open(newline="", encoding="utf-8") as log_file:
            log_rows = list(csv.DictReader(log_file))
        if len(log_rows) != len(history) or not best_model_path.is_file():
            raise RuntimeError(f"Training outputs are incomplete: {output_dir}")
        print(
            f"fit_model test passed: monitor={monitor}, "
            f"epochs_run={len(history)}, best_epoch={best_epoch}"
        )
