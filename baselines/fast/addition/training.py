"""Auditable single-GPU training utilities for the official baselines."""

from __future__ import annotations

import csv
import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader


def build_optimizer(
    model_name: str,
    model: nn.Module,
    config: Mapping[str, Any],
) -> torch.optim.Optimizer:
    base_lr = float(config["learning_rate"])
    weight_decay = float(config["weight_decay"])
    head_multiplier = float(config.get("head_learning_rate_multiplier", 1.0))

    if model_name == "labram" and "layer_decay" in config:
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
        parameter_groups = [
            {
                "params": parameters,
                "lr": base_lr * scale,
                "weight_decay": decay,
                "lr_scale": scale,
            }
            for (scale, decay), parameters in sorted(grouped.items())
        ]
        return torch.optim.AdamW(parameter_groups, lr=base_lr)

    backbone: list[nn.Parameter] = []
    head: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if model_name == "eegmamba" and ".backbone." not in name:
            head.append(parameter)
        else:
            backbone.append(parameter)
    groups: list[dict[str, Any]] = []
    if backbone:
        groups.append({"params": backbone, "lr": base_lr})
    if head:
        groups.append({"params": head, "lr": base_lr * head_multiplier})
    return torch.optim.AdamW(groups, lr=base_lr, weight_decay=weight_decay)


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
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.set_grad_enabled(training):
        for step, (eeg, labels) in enumerate(loader, start=1):
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

            batch_size = labels.shape[0]
            total_loss += float(loss.detach()) * batch_size
            total_correct += int((logits.argmax(dim=1) == labels).sum())
            total_examples += batch_size

    if total_examples == 0:
        raise RuntimeError("The data loader is empty")
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
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
    clip_norm = float(config.get("gradient_clip_norm", 0.0))
    amp = bool(config.get("amp", True) and device.type == "cuda")
    criterion_train = nn.CrossEntropyLoss(
        label_smoothing=float(config.get("label_smoothing", 0.0))
    )
    criterion_eval = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model_name, model, config)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        epochs=epochs,
        warmup_epochs=int(config.get("warmup_epochs", 0)),
        minimum_learning_rate=float(config.get("minimum_learning_rate", 0.0)),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    best_value = -math.inf if monitor == "accuracy" else math.inf
    best_epoch = 0
    best_state = None
    stale = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    fields = (
        "epoch", "learning_rate", "train_loss", "train_accuracy",
        "val_loss", "val_accuracy", "is_best",
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
            )
            validation = _run_epoch(
                model, val_loader, criterion_eval, device, amp=amp
            )
            value = validation[monitor]
            improved = value > best_value if monitor == "accuracy" else value < best_value
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
                "is_best": int(improved),
            }
            writer.writerow(record)
            handle.flush()
            history.append(record)
            marker = " *" if improved else ""
            print(
                f"epoch={epoch:03d} lr={record['learning_rate']:.7g} "
                f"train_loss={train['loss']:.6f} train_acc={train['accuracy']:.4f} "
                f"val_loss={validation['loss']:.6f} val_acc={validation['accuracy']:.4f}{marker}",
                flush=True,
            )
            scheduler.step()
            if stale >= patience:
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
