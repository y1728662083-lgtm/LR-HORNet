"""Auditable single-GPU training utilities for LaBraM."""

from __future__ import annotations

import csv
import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torch.utils.data import DataLoader


def build_optimizer(
    model_name: str,
    model: nn.Module,
    config: Mapping[str, Any],
) -> torch.optim.Optimizer:
    if model_name != "labram":
        raise KeyError(f"This bundle only provides LaBraM, got {model_name!r}")
    base_lr = float(config["learning_rate"])
    weight_decay = float(config["weight_decay"])
    layer_decay = float(config["layer_decay"])
    max_layer = 13
    grouped: dict[tuple[float, float], list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if ".blocks." in name:
            layer_id = int(name.split(".blocks.", 1)[1].split(".", 1)[0]) + 1
        elif any(token in name for token in (".head.", ".norm.", ".fc_norm.")):
            layer_id = max_layer
        else:
            layer_id = 0
        scale = layer_decay ** (max_layer - layer_id)
        no_decay = (
            parameter.ndim == 1
            or name.endswith(".bias")
            or any(token in name for token in ("pos_embed", "time_embed", "cls_token"))
        )
        decay = 0.0 if no_decay else weight_decay
        grouped.setdefault((scale, decay), []).append(parameter)
    groups = [
        {"params": params, "lr": base_lr * scale,
         "weight_decay": decay, "lr_scale": scale}
        for (scale, decay), params in sorted(grouped.items())
    ]
    return torch.optim.AdamW(groups, lr=base_lr)



def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    warmup_epochs: int,
    minimum_learning_rate: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]

    def make_lambda(base_lr: float):
        minimum_factor = min(float(minimum_learning_rate) / base_lr, 1.0)

        def lr_lambda(epoch: int) -> float:
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return max((epoch + 1) / warmup_epochs, 1.0 / warmup_epochs)
            progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs - 1, 1)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return minimum_factor + (1.0 - minimum_factor) * cosine

        return lr_lambda

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, [make_lambda(lr) for lr in base_lrs]
    )


class OfficialStepScheduler:
    """Official LaBraM per-update linear-warm-up plus cosine schedule."""

    def __init__(
        self, optimizer: torch.optim.Optimizer, *, mode: str, epochs: int,
        steps_per_epoch: int, base_learning_rate: float,
        minimum_learning_rate: float, warmup_epochs: int,
    ) -> None:
        if mode != "labram_official_warmup_cosine":
            raise ValueError(f"Unknown LaBraM scheduler mode: {mode}")
        if steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be positive")
        self.optimizer = optimizer
        self.update_index = 0
        self.cosine = None
        total_updates = epochs * steps_per_epoch
        warmup_updates = warmup_epochs * steps_per_epoch
        values: list[float] = []
        if warmup_updates:
            values.extend(
                base_learning_rate * index / max(warmup_updates - 1, 1)
                for index in range(warmup_updates)
            )
        cosine_updates = total_updates - warmup_updates
        values.extend(
            minimum_learning_rate
            + 0.5 * (base_learning_rate - minimum_learning_rate)
            * (1.0 + math.cos(math.pi * index / cosine_updates))
            for index in range(cosine_updates)
        )
        if len(values) != total_updates:
            raise RuntimeError("LaBraM learning-rate schedule length mismatch")
        self.values = values

    def before_batch(self) -> None:
        value = self.values[min(self.update_index, len(self.values) - 1)]
        for group in self.optimizer.param_groups:
            group["lr"] = value * float(group.get("lr_scale", 1.0))

    def after_optimizer_step(self) -> None:
        self.update_index += 1



def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    accumulation_steps: int = 1,
    clip_norm: float = 0.0,
    amp: bool = False,
    step_scheduler: OfficialStepScheduler | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    targets_all: list[np.ndarray] = []
    predictions_all: list[np.ndarray] = []

    with torch.set_grad_enabled(training):
        for step, (eeg, labels) in enumerate(loader, start=1):
            if training and step_scheduler is not None:
                step_scheduler.before_batch()
            eeg = eeg.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(eeg)
                loss = criterion(logits, labels)
                backward_loss = loss / accumulation_steps

            if training:
                assert optimizer is not None and scaler is not None
                scaler.scale(backward_loss).backward()
                should_step = step % accumulation_steps == 0 or step == len(loader)
                if should_step:
                    scaler.unscale_(optimizer)
                    if clip_norm > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if step_scheduler is not None:
                        step_scheduler.after_optimizer_step()

            batch_size = labels.shape[0]
            total_loss += float(loss.detach()) * batch_size
            total_correct += int((logits.argmax(dim=1) == labels).sum())
            total_examples += batch_size
            targets_all.append(labels.detach().cpu().numpy())
            predictions_all.append(logits.detach().argmax(dim=1).cpu().numpy())

    if total_examples == 0:
        raise RuntimeError("The data loader is empty")
    targets = np.concatenate(targets_all)
    predictions = np.concatenate(predictions_all)
    kappa = float(cohen_kappa_score(targets, predictions))
    if not math.isfinite(kappa):
        kappa = -1.0
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "cohen_kappa": kappa,
    }


def fit_model(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: Mapping[str, Any],
    output_dir: str | Path,
    *,
    monitor: str = "accuracy",
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = int(config["epochs"])
    patience = int(config["patience"])
    accumulation = int(config.get("gradient_accumulation_steps", 1))
    scheduler_mode = str(config.get("scheduler", "legacy_epoch_cosine"))
    if scheduler_mode == "labram_official_warmup_cosine" and accumulation != 1:
        raise ValueError("Released LaBraM fine-tuning uses accumulation/update_freq=1")
    clip_norm = float(config.get("gradient_clip_norm", 0.0))
    amp = bool(config.get("amp", True) and device.type == "cuda")
    criterion_train = nn.CrossEntropyLoss(
        label_smoothing=float(config.get("label_smoothing", 0.0))
    )
    criterion_eval = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model_name, model, config)
    step_scheduler: OfficialStepScheduler | None = None
    epoch_scheduler = None
    if scheduler_mode == "labram_official_warmup_cosine":
        step_scheduler = OfficialStepScheduler(
            optimizer,
            mode=scheduler_mode,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            base_learning_rate=float(config["learning_rate"]),
            minimum_learning_rate=float(config.get("minimum_learning_rate", 0.0)),
            warmup_epochs=int(config.get("warmup_epochs", 0)),
        )
    else:
        epoch_scheduler = build_warmup_cosine_scheduler(
            optimizer,
            epochs=epochs,
            warmup_epochs=int(config.get("warmup_epochs", 0)),
            minimum_learning_rate=float(config.get("minimum_learning_rate", 0.0)),
        )
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    monitor = str(config.get("selection_metric", monitor))
    if monitor not in {"accuracy", "cohen_kappa", "loss"}:
        raise ValueError(f"Unsupported selection metric: {monitor}")
    best_value = math.inf if monitor == "loss" else -math.inf
    early_stopping = bool(config.get("early_stopping", False))
    best_epoch = 0
    best_state = None
    stale = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    fields = (
        "epoch", "learning_rate", "train_loss", "train_accuracy",
        "val_loss", "val_accuracy", "val_cohen_kappa", "is_best",
    )
    log_path = output_dir / "training_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            train = _run_epoch(
                model, train_loader, criterion_train, device,
                optimizer=optimizer, scaler=scaler,
                accumulation_steps=accumulation, clip_norm=clip_norm, amp=amp,
                step_scheduler=step_scheduler,
            )
            validation = _run_epoch(
                model, val_loader, criterion_eval, device, amp=amp
            )
            value = validation[monitor]
            improved = value < best_value if monitor == "loss" else value > best_value
            if improved:
                best_value = value
                best_epoch = epoch
                best_state = deepcopy(model.state_dict())
                torch.save(best_state, output_dir / "best_model.pt")
                stale = 0
            else:
                stale += 1
            record = {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[-1]["lr"],
                "train_loss": train["loss"],
                "train_accuracy": train["accuracy"],
                "val_loss": validation["loss"],
                "val_accuracy": validation["accuracy"],
                "val_cohen_kappa": validation["cohen_kappa"],
                "is_best": int(improved),
            }
            writer.writerow(record)
            handle.flush()
            history.append(record)
            marker = " *" if improved else ""
            print(
                f"epoch={epoch:03d} lr={record['learning_rate']:.7g} "
                f"train_loss={train['loss']:.6f} train_acc={train['accuracy']:.4f} "
                f"val_loss={validation['loss']:.6f} val_acc={validation['accuracy']:.4f} "
                f"val_kappa={validation['cohen_kappa']:.4f}{marker}",
                flush=True,
            )
            if epoch_scheduler is not None:
                epoch_scheduler.step()
            if early_stopping and stale >= patience:
                print(f"early stopping at epoch {epoch}", flush=True)
                break

    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected")
    model.load_state_dict(best_state)
    elapsed = time.perf_counter() - started
    return {
        "best_epoch": best_epoch,
        "best_validation_metric": best_value,
        "epochs_run": len(history),
        "elapsed_seconds": elapsed,
        "amp": amp,
        "gradient_accumulation_steps": accumulation,
        "scheduler": scheduler_mode,
        "selection_metric": monitor,
        "early_stopping": early_stopping,
    }


def predict_and_score(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    amp: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    labels_all: list[np.ndarray] = []
    predictions_all: list[np.ndarray] = []
    probabilities_all: list[np.ndarray] = []
    with torch.inference_mode():
        for eeg, labels in loader:
            eeg = eeg.to(device, non_blocking=True)
            labels_device = labels.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                logits = model(eeg)
                loss = criterion(logits, labels_device)
            probabilities = logits.float().softmax(dim=1).cpu().numpy()
            predictions = probabilities.argmax(axis=1)
            total_loss += float(loss)
            labels_all.append(labels.numpy())
            predictions_all.append(predictions)
            probabilities_all.append(probabilities)
    y_true = np.concatenate(labels_all)
    y_pred = np.concatenate(predictions_all)
    probabilities = np.concatenate(probabilities_all)
    classes = list(range(probabilities.shape[1]))
    result = {
        "loss": total_loss / len(y_true),
        "accuracy": float(np.mean(y_true == y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        "n_test_trials": int(len(y_true)),
    }
    rows = []
    for index, (target, prediction, probability) in enumerate(
        zip(y_true, y_pred, probabilities, strict=True)
    ):
        row: dict[str, Any] = {
            "evaluation_order": index,
            "target": int(target),
            "prediction": int(prediction),
            "correct": int(target == prediction),
        }
        row.update({f"probability_{i}": float(value) for i, value in enumerate(probability)})
        rows.append(row)
    return result, rows
