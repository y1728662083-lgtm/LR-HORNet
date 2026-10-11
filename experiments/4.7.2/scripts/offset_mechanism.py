#!/usr/bin/env python3
"""Inference-only causal and synthetic-shift tests for temporal offsets.

The script loads an already trained ``group4_patchLstm_deformableLstm``
checkpoint.  It never updates model parameters.  Positive synthetic shift
means a rightward delay, ``x_delta[t] = x[t - delta]``.  Because the model
samples at ``t + offset``, the ideal compensating sampling offset has the
same sign as ``delta``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


MODES = ("learned", "zero", "shuffled", "random_matched")
MODE_LABELS = {
    "learned": "Learned offset",
    "zero": "Zero offset",
    "shuffled": "Trial-shuffled offset",
    "random_matched": "Phase-randomized matched offset",
}
MODE_COLORS = {
    "learned": "#1F77B4",
    "zero": "#7F7F7F",
    "shuffled": "#FF7F0E",
    "random_matched": "#2CA02C",
}


def find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "main.py").is_file() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError(
        "Could not locate paper4 root. Put this script below a directory "
        "containing main.py and src/."
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_shifts(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("--shifts must contain at least one number")
    if 0.0 not in values:
        values.append(0.0)
    return sorted(set(values))


def load_state_dict(path: Path, device: torch.device) -> dict[str, Tensor]:
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict) or not payload:
        raise TypeError(f"Unsupported checkpoint payload in {path}")
    if all(str(key).startswith("module.") for key in payload):
        payload = {str(key)[7:]: value for key, value in payload.items()}
    return payload


def find_deformation(model: nn.Module) -> tuple[str, nn.Module]:
    candidates = []
    for name, module in model.named_modules():
        if all(hasattr(module, attr) for attr in ("lstm", "offset_head", "gate_head")):
            candidates.append((name, module))
    if len(candidates) != 1:
        names = [name for name, _ in candidates]
        raise RuntimeError(f"Expected one LSTM deformation module, found {names}")
    return candidates[0]


def replace_named_module(model: nn.Module, name: str, replacement: nn.Module) -> None:
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], replacement)


class OffsetInterventionWarp(nn.Module):
    """Exact trained warp with controllable applied offsets and capture."""

    def __init__(self, trained_warp: nn.Module) -> None:
        super().__init__()
        self.trained_warp = trained_warp
        self.mode = "learned"
        self.override: Tensor | None = None
        self.capture = True
        self.last_pre: Tensor | None = None
        self.last_predicted_offset: Tensor | None = None
        self.last_applied_offset: Tensor | None = None
        self.last_post: Tensor | None = None

    def set_intervention(self, mode: str, override: Tensor | None = None) -> None:
        if mode not in ("learned", "zero", "override"):
            raise ValueError(f"Unknown controller mode: {mode}")
        self.mode = mode
        self.override = override

    def predict_offset(self, input: Tensor) -> Tensor:
        batch, maps, spatial, times = input.shape
        if spatial != 1:
            raise ValueError(f"Expected spatial dimension 1, got {input.shape}")
        sequences = input[:, :, 0, :].reshape(batch * maps, times, 1)
        hidden, _ = self.trained_warp.lstm(sequences)
        raw = self.trained_warp.offset_head(hidden).squeeze(-1)
        gate = torch.sigmoid(self.trained_warp.gate_head(hidden)).squeeze(-1)
        return (gate * raw).reshape(batch, maps, times)

    def warp(self, input: Tensor, offset: Tensor) -> Tensor:
        batch, maps, _, times = input.shape
        if offset.shape != (batch, maps, times):
            raise ValueError(
                f"Offset shape {tuple(offset.shape)} does not match "
                f"{(batch, maps, times)}"
            )
        signal = input.reshape(batch * maps, 1, 1, times)
        flattened = offset.reshape(batch * maps, times)
        base = torch.arange(times, device=input.device, dtype=input.dtype)[None]
        sample_time = base + flattened
        normalized = (
            torch.zeros_like(sample_time)
            if times == 1
            else 2.0 * sample_time / (times - 1) - 1.0
        )
        grid = torch.stack(
            (normalized.unsqueeze(1), torch.zeros_like(normalized).unsqueeze(1)),
            dim=-1,
        )
        warped = F.grid_sample(
            signal,
            grid,
            mode="bilinear",
            padding_mode=self.trained_warp.padding_mode,
            align_corners=self.trained_warp.align_corners,
        )
        return warped.reshape(batch, maps, 1, times)

    def forward(self, input: Tensor) -> Tensor:
        predicted = self.predict_offset(input)
        if self.mode == "learned":
            applied = predicted
        elif self.mode == "zero":
            applied = torch.zeros_like(predicted)
        else:
            if self.override is None:
                raise RuntimeError("override mode requires an offset tensor")
            applied = self.override.to(device=input.device, dtype=input.dtype)
        post = self.warp(input, applied)
        if self.capture:
            self.last_pre = input[:, :, 0, :].detach().cpu()
            self.last_predicted_offset = predicted.detach().cpu()
            self.last_applied_offset = applied.detach().cpu()
            self.last_post = post[:, :, 0, :].detach().cpu()
        return post


@dataclass
class InferenceResult:
    logits: Tensor
    probabilities: Tensor
    predictions: Tensor
    labels: Tensor
    pre: Tensor
    predicted_offset: Tensor
    applied_offset: Tensor
    post: Tensor


def cache_test_loader(loader, max_trials: int | None) -> tuple[Tensor, Tensor]:
    eeg_parts, label_parts = [], []
    for batch in loader:
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            raise TypeError("Test loader must return at least (eeg, label)")
        eeg_parts.append(batch[0].detach().cpu())
        label_parts.append(batch[1].detach().cpu())
    eeg = torch.cat(eeg_parts)
    labels = torch.cat(label_parts).long()
    if max_trials is not None:
        eeg, labels = eeg[:max_trials], labels[:max_trials]
    if len(eeg) == 0:
        raise RuntimeError("Test set is empty")
    return eeg.contiguous(), labels.contiguous()


def infer(
    model: nn.Module,
    controller: OffsetInterventionWarp,
    eeg: Tensor,
    labels: Tensor,
    *,
    batch_size: int,
    device: torch.device,
    mode: str,
    override_offsets: Tensor | None = None,
) -> InferenceResult:
    fields = {key: [] for key in (
        "logits", "pre", "predicted_offset", "applied_offset", "post"
    )}
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(eeg), batch_size):
            stop = min(start + batch_size, len(eeg))
            if mode == "learned":
                controller.set_intervention("learned")
            elif mode == "zero":
                controller.set_intervention("zero")
            else:
                if override_offsets is None:
                    raise ValueError(f"{mode} requires override_offsets")
                controller.set_intervention(
                    "override", override_offsets[start:stop]
                )
            logits = model(eeg[start:stop].to(device))
            fields["logits"].append(logits.detach().cpu())
            for key in ("pre", "predicted_offset", "applied_offset", "post"):
                value = getattr(controller, f"last_{key}")
                if value is None:
                    raise RuntimeError(f"Controller did not capture {key}")
                fields[key].append(value)
    logits = torch.cat(fields["logits"])
    probabilities = logits.softmax(dim=1)
    return InferenceResult(
        logits=logits,
        probabilities=probabilities,
        predictions=logits.argmax(dim=1),
        labels=labels,
        pre=torch.cat(fields["pre"]),
        predicted_offset=torch.cat(fields["predicted_offset"]),
        applied_offset=torch.cat(fields["applied_offset"]),
        post=torch.cat(fields["post"]),
    )


def classification_metrics(labels: Tensor, predictions: Tensor, n_classes: int) -> dict:
    labels_np = labels.numpy()
    predictions_np = predictions.numpy()
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for target, predicted in zip(labels_np, predictions_np):
        confusion[int(target), int(predicted)] += 1
    recalls, f1s = [], []
    for cls in range(n_classes):
        tp = confusion[cls, cls]
        fn = confusion[cls].sum() - tp
        fp = confusion[:, cls].sum() - tp
        recalls.append(tp / (tp + fn) if tp + fn else 0.0)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = recalls[-1]
        f1s.append(
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
    return {
        "accuracy": float((labels_np == predictions_np).mean()),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "confusion_matrix": confusion.tolist(),
    }


def derangement(n: int, generator: torch.Generator) -> Tensor:
    if n < 2:
        raise ValueError("Trial-shuffled offsets require at least two trials")
    order = torch.randperm(n, generator=generator)
    permutation = torch.empty_like(order)
    permutation[order] = order.roll(1)
    if torch.any(permutation == torch.arange(n)):
        raise RuntimeError("Failed to create a derangement")
    return permutation


def phase_randomized_matched(offsets: Tensor, seed: int) -> Tensor:
    """Randomize temporal phase while preserving each offset power spectrum."""
    if offsets.ndim != 3:
        raise ValueError(f"Expected (N,M,T) offsets, got {offsets.shape}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    spectrum = torch.fft.rfft(offsets.float(), dim=-1)
    phase = torch.rand(
        spectrum.shape,
        generator=generator,
        dtype=offsets.dtype,
    ) * (2.0 * math.pi)
    factor = torch.polar(torch.ones_like(phase), phase)
    factor[..., 0] = 1.0 + 0.0j
    if offsets.shape[-1] % 2 == 0:
        signs = torch.randint(
            0, 2, spectrum[..., -1].shape, generator=generator
        ).to(offsets.dtype) * 2 - 1
        factor[..., -1] = torch.complex(signs, torch.zeros_like(signs))
    randomized = torch.fft.irfft(spectrum * factor, n=offsets.shape[-1], dim=-1)
    # Numerical guard: exactly restore sequence-wise mean and standard deviation.
    original_mean = offsets.mean(dim=-1, keepdim=True)
    original_std = offsets.std(dim=-1, keepdim=True, unbiased=False)
    randomized = randomized - randomized.mean(dim=-1, keepdim=True)
    randomized_std = randomized.std(dim=-1, keepdim=True, unbiased=False)
    randomized = randomized * (
        original_std / randomized_std.clamp_min(torch.finfo(offsets.dtype).eps)
    ) + original_mean
    return randomized.to(offsets.dtype)


def bootstrap_paired_accuracy_delta(
    learned_correct: np.ndarray,
    intervention_correct: np.ndarray,
    *,
    seed: int,
    draws: int = 5000,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    delta = intervention_correct.astype(float) - learned_correct.astype(float)
    samples = rng.integers(0, len(delta), size=(draws, len(delta)))
    boot = delta[samples].mean(axis=1)
    return float(delta.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def causal_ablation(
    model: nn.Module,
    controller: OffsetInterventionWarp,
    eeg: Tensor,
    labels: Tensor,
    *,
    batch_size: int,
    device: torch.device,
    n_classes: int,
    seed: int,
    output_dir: Path,
    feature_map: int,
    example_trial: int,
) -> dict[str, InferenceResult]:
    print("[causal] learned offsets")
    learned = infer(
        model, controller, eeg, labels, batch_size=batch_size,
        device=device, mode="learned"
    )
    generator = torch.Generator().manual_seed(seed + 101)
    permutation = derangement(len(eeg), generator)
    shuffled = learned.applied_offset[permutation]
    matched = phase_randomized_matched(learned.applied_offset, seed + 202)
    offsets = {"shuffled": shuffled, "random_matched": matched}
    results = {"learned": learned}
    print("[causal] zero offsets")
    results["zero"] = infer(
        model, controller, eeg, labels, batch_size=batch_size,
        device=device, mode="zero"
    )
    for mode in ("shuffled", "random_matched"):
        print(f"[causal] {mode} offsets")
        results[mode] = infer(
            model, controller, eeg, labels, batch_size=batch_size,
            device=device, mode=mode, override_offsets=offsets[mode]
        )

    summary, per_trial = [], []
    learned_correct = (learned.predictions == labels).numpy()
    for mode in MODES:
        result = results[mode]
        metrics = classification_metrics(labels, result.predictions, n_classes)
        correct = (result.predictions == labels).numpy()
        delta, ci_low, ci_high = bootstrap_paired_accuracy_delta(
            learned_correct, correct, seed=seed + MODES.index(mode)
        )
        applied = result.applied_offset
        summary.append({
            "mode": mode,
            "mode_label": MODE_LABELS[mode],
            "n_trials": len(labels),
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "accuracy_delta_vs_learned": delta,
            "paired_bootstrap_ci95_low": ci_low,
            "paired_bootstrap_ci95_high": ci_high,
            "mean_signed_applied_offset_samples": float(applied.mean()),
            "mean_abs_applied_offset_samples": float(applied.abs().mean()),
            "applied_offset_std_samples": float(applied.std(unbiased=False)),
            "confusion_matrix_json": json.dumps(metrics["confusion_matrix"]),
        })
        true_probability = result.probabilities.gather(1, labels[:, None])[:, 0]
        for index in range(len(labels)):
            current = applied[index]
            per_trial.append({
                "mode": mode,
                "trial_index": index,
                "label": int(labels[index]),
                "prediction": int(result.predictions[index]),
                "correct": int(correct[index]),
                "true_class_probability": float(true_probability[index]),
                "mean_signed_applied_offset_samples": float(current.mean()),
                "mean_abs_applied_offset_samples": float(current.abs().mean()),
                "rms_applied_offset_samples": float(current.square().mean().sqrt()),
                "shuffle_source_trial": int(permutation[index]) if mode == "shuffled" else "",
            })
    write_csv(output_dir / "causal_ablation_summary.csv", summary)
    write_csv(output_dir / "causal_ablation_per_trial.csv", per_trial)
    plot_causal_metrics(summary, output_dir / "causal_ablation_metrics.png")
    plot_offset_examples(
        results,
        output_dir / "causal_offset_examples.png",
        feature_map=feature_map,
        trial=example_trial,
    )
    return results


def plot_causal_metrics(rows: list[dict], path: Path) -> None:
    labels = [row["mode_label"] for row in rows]
    colors = [MODE_COLORS[row["mode"]] for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for ax, key, title in (
        (axes[0], "accuracy", "Classification accuracy"),
        (axes[1], "macro_f1", "Macro-F1"),
    ):
        values = [row[key] for row in rows]
        bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.6)
        ax.axhline(0.25, color="#B22222", linestyle="--", linewidth=1, label="Chance")
        ax.set_xticks(x, labels, rotation=18, ha="right")
        ax.set_ylim(0, max(0.5, max(values) + 0.08))
        ax.set_ylabel(key.replace("_", " "))
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.bar_label(bars, labels=[f"{value:.1%}" for value in values], padding=3)
    axes[0].legend(frameon=False)
    fig.suptitle("Inference-only causal interventions on learned temporal offsets")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_offset_examples(
    results: dict[str, InferenceResult],
    path: Path,
    *,
    feature_map: int,
    trial: int,
) -> None:
    trial = min(max(trial, 0), len(next(iter(results.values())).labels) - 1)
    maps = next(iter(results.values())).applied_offset.shape[1]
    feature_map = min(max(feature_map, 0), maps - 1)
    fig, axes = plt.subplots(4, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
    for ax, mode in zip(axes, MODES):
        curve = results[mode].applied_offset[trial, feature_map].numpy()
        ax.plot(curve, color=MODE_COLORS[mode], linewidth=1.2)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_ylabel("samples")
        ax.set_title(MODE_LABELS[mode], loc="left", fontsize=10)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Pre-pooling feature time index")
    fig.suptitle(
        f"Applied offset controls: trial {trial}, feature map {feature_map}"
    )
    fig.savefig(path, dpi=220)
    plt.close(fig)


def fractional_delay(eeg: Tensor, delay_samples: float) -> Tensor:
    """Right-shift EEG by delay_samples using border-padded linear sampling."""
    if delay_samples == 0:
        return eeg.clone()
    if eeg.ndim != 3:
        raise ValueError(f"Expected EEG (N,C,T), got {eeg.shape}")
    trials, channels, times = eeg.shape
    signal = eeg.reshape(trials * channels, 1, 1, times)
    base = torch.arange(times, dtype=eeg.dtype)[None]
    # delayed[t] = original[t - delay]; positive delay is a right shift.
    sample_time = base - float(delay_samples)
    normalized = (
        torch.zeros_like(sample_time)
        if times == 1 else 2.0 * sample_time / (times - 1) - 1.0
    )
    normalized = normalized.expand(trials * channels, -1)
    grid = torch.stack(
        (normalized.unsqueeze(1), torch.zeros_like(normalized).unsqueeze(1)),
        dim=-1,
    )
    shifted = F.grid_sample(
        signal,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return shifted.reshape(trials, channels, times)


def feature_similarity(reference: Tensor, current: Tensor, margin: int) -> tuple[Tensor, Tensor]:
    if reference.shape != current.shape:
        raise ValueError("Feature tensors must have identical shapes")
    times = reference.shape[-1]
    if margin * 2 >= times:
        margin = 0
    ref = reference[..., margin: times - margin if margin else times].flatten(1)
    cur = current[..., margin: times - margin if margin else times].flatten(1)
    ref_centered = ref - ref.mean(dim=1, keepdim=True)
    cur_centered = cur - cur.mean(dim=1, keepdim=True)
    correlation = (
        (ref_centered * cur_centered).sum(dim=1)
        / (
            ref_centered.square().sum(dim=1).sqrt()
            * cur_centered.square().sum(dim=1).sqrt()
        ).clamp_min(1e-12)
    )
    nrmse = (
        (ref - cur).square().mean(dim=1).sqrt()
        / ref.square().mean(dim=1).sqrt().clamp_min(1e-12)
    )
    return correlation, nrmse


def artificial_shift_recovery(
    model: nn.Module,
    controller: OffsetInterventionWarp,
    eeg: Tensor,
    labels: Tensor,
    *,
    batch_size: int,
    device: torch.device,
    n_classes: int,
    shifts: Sequence[float],
    output_dir: Path,
    feature_map: int,
    example_trial: int,
    causal_results: dict[str, InferenceResult] | None,
) -> None:
    references: dict[str, InferenceResult] = {}
    if causal_results is not None:
        references = {
            "learned": causal_results["learned"],
            "zero": causal_results["zero"],
        }
    else:
        for mode in ("learned", "zero"):
            references[mode] = infer(
                model, controller, eeg, labels, batch_size=batch_size,
                device=device, mode=mode
            )

    summary, per_trial = [], []
    cached_results: dict[tuple[str, float], InferenceResult] = {}
    for delay in shifts:
        shifted_eeg = fractional_delay(eeg, delay)
        for mode in ("learned", "zero"):
            print(f"[shift] delay={delay:+g} samples mode={mode}")
            result = (
                references[mode]
                if delay == 0 else infer(
                    model, controller, shifted_eeg, labels,
                    batch_size=batch_size, device=device, mode=mode
                )
            )
            cached_results[(mode, delay)] = result
            reference = references[mode]
            margin = int(math.ceil(abs(delay))) + 2
            corr, nrmse = feature_similarity(reference.post, result.post, margin)
            consistent = result.predictions == reference.predictions
            metrics = classification_metrics(labels, result.predictions, n_classes)
            offset_change = (
                result.predicted_offset - references["learned"].predicted_offset
                if mode == "learned"
                else torch.zeros_like(result.predicted_offset)
            )
            trial_offset_change = offset_change.mean(dim=(1, 2))
            compensation_error = trial_offset_change - float(delay)
            summary.append({
                "mode": mode,
                "delay_samples": delay,
                "ideal_sampling_offset_samples": delay,
                "n_trials": len(labels),
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "prediction_consistency_vs_unshifted": float(consistent.float().mean()),
                "postwarp_feature_correlation_mean": float(corr.mean()),
                "postwarp_feature_correlation_std": float(corr.std(unbiased=False)),
                "postwarp_feature_nrmse_mean": float(nrmse.mean()),
                "mean_predicted_offset_samples": float(result.predicted_offset.mean()),
                "mean_offset_response_samples": float(trial_offset_change.mean()),
                "offset_compensation_mae_samples": float(compensation_error.abs().mean()),
            })
            true_probability = result.probabilities.gather(1, labels[:, None])[:, 0]
            for index in range(len(labels)):
                per_trial.append({
                    "mode": mode,
                    "delay_samples": delay,
                    "ideal_sampling_offset_samples": delay,
                    "trial_index": index,
                    "label": int(labels[index]),
                    "prediction": int(result.predictions[index]),
                    "correct": int(result.predictions[index] == labels[index]),
                    "true_class_probability": float(true_probability[index]),
                    "prediction_consistent_vs_unshifted": int(consistent[index]),
                    "postwarp_feature_correlation": float(corr[index]),
                    "postwarp_feature_nrmse": float(nrmse[index]),
                    "mean_predicted_offset_samples": float(result.predicted_offset[index].mean()),
                    "mean_offset_response_samples": float(trial_offset_change[index]),
                    "offset_compensation_error_samples": float(compensation_error[index]),
                })

    learned_nonzero = [
        row for row in summary if row["mode"] == "learned" and row["delay_samples"] != 0
    ]
    x = np.array([row["ideal_sampling_offset_samples"] for row in learned_nonzero])
    y = np.array([row["mean_offset_response_samples"] for row in learned_nonzero])
    if len(np.unique(x)) >= 2:
        slope, intercept = np.polyfit(x, y, deg=1)
        correlation = float(np.corrcoef(x, y)[0, 1])
    else:
        slope = intercept = correlation = float("nan")
    for row in summary:
        row["learned_response_slope_all_nonzero_shifts"] = float(slope)
        row["learned_response_intercept_all_nonzero_shifts"] = float(intercept)
        row["learned_response_correlation_all_nonzero_shifts"] = correlation

    write_csv(output_dir / "artificial_shift_summary.csv", summary)
    write_csv(output_dir / "artificial_shift_per_trial.csv", per_trial)
    plot_shift_summary(summary, output_dir / "artificial_shift_recovery.png")
    plot_shift_example(
        eeg,
        shifts,
        cached_results,
        output_dir / "artificial_shift_example.png",
        feature_map=feature_map,
        trial=example_trial,
    )


def plot_shift_summary(rows: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    panels = (
        ("accuracy", "Accuracy", axes[0, 0]),
        ("postwarp_feature_correlation_mean", "Post-warp feature correlation", axes[0, 1]),
        ("prediction_consistency_vs_unshifted", "Prediction consistency", axes[1, 0]),
    )
    for key, title, ax in panels:
        for mode in ("learned", "zero"):
            rr = sorted((r for r in rows if r["mode"] == mode), key=lambda r: r["delay_samples"])
            ax.plot(
                [r["delay_samples"] for r in rr], [r[key] for r in rr],
                marker="o", linewidth=1.8, label=MODE_LABELS[mode],
                color=MODE_COLORS[mode]
            )
        if key == "accuracy":
            ax.axhline(0.25, color="#B22222", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Artificial rightward delay (samples)")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    ax = axes[1, 1]
    rr = sorted((r for r in rows if r["mode"] == "learned"), key=lambda r: r["delay_samples"])
    delays = np.array([r["delay_samples"] for r in rr])
    responses = np.array([r["mean_offset_response_samples"] for r in rr])
    ax.plot(delays, delays, "--", color="black", label="Ideal response")
    ax.plot(delays, responses, "o-", color=MODE_COLORS["learned"], label="Learned response")
    ax.set_title("Offset response to known delay")
    ax.set_xlabel("Artificial rightward delay (samples)")
    ax.set_ylabel("Predicted offset change (samples)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.suptitle("Synthetic temporal-shift recovery using a frozen checkpoint")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_shift_example(
    eeg: Tensor,
    shifts: Sequence[float],
    results: dict[tuple[str, float], InferenceResult],
    path: Path,
    *,
    feature_map: int,
    trial: int,
) -> None:
    candidates = [value for value in shifts if value > 0]
    delay = min(candidates, key=lambda value: abs(value - 4.0)) if candidates else max(shifts)
    trial = min(max(trial, 0), len(eeg) - 1)
    learned_ref = results[("learned", 0.0)]
    learned_shift = results[("learned", delay)]
    zero_shift = results[("zero", delay)]
    maps = learned_ref.pre.shape[1]
    feature_map = min(max(feature_map, 0), maps - 1)
    shifted_eeg = fractional_delay(eeg[trial:trial + 1], delay)[0]
    fig, axes = plt.subplots(4, 1, figsize=(13, 9), constrained_layout=True)
    axes[0].plot(eeg[trial, 0], label="Original", linewidth=1)
    axes[0].plot(shifted_eeg[0], label=f"Delayed by {delay:g} samples", linewidth=1, alpha=0.8)
    axes[0].set_title("Raw EEG channel 0")
    axes[0].legend(frameon=False)
    axes[1].plot(learned_ref.pre[trial, feature_map], label="Unshifted pre-warp")
    axes[1].plot(learned_shift.pre[trial, feature_map], label="Shifted pre-warp", alpha=0.8)
    axes[1].set_title(f"Pre-warp spatial feature map {feature_map}")
    axes[1].legend(frameon=False)
    axes[2].plot(learned_ref.post[trial, feature_map], label="Unshifted learned")
    axes[2].plot(learned_shift.post[trial, feature_map], label="Shifted learned", alpha=0.8)
    axes[2].plot(zero_shift.post[trial, feature_map], label="Shifted zero-offset", alpha=0.7)
    axes[2].set_title("Post-warp feature recovery")
    axes[2].legend(frameon=False)
    axes[3].plot(learned_ref.predicted_offset[trial, feature_map], label="Unshifted")
    axes[3].plot(learned_shift.predicted_offset[trial, feature_map], label="Shifted")
    axes[3].axhline(delay, linestyle="--", color="black", label="Ideal extra offset")
    axes[3].set_title("Predicted sampling offset")
    axes[3].set_ylabel("samples")
    axes[3].set_xlabel("Time index")
    axes[3].legend(frameon=False)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle(f"Representative synthetic-delay response: trial {trial}")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument(
        "--data-root", type=Path,
        help="Override data.root from the archived training YAML",
    )
    parser.add_argument("--fold", type=int, help="One-based Fold; defaults to the first config Fold")
    parser.add_argument(
        "--experiment", choices=("both", "causal", "shift"), default="both"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-trials", type=int, default=0, help="0 means the full test set")
    parser.add_argument(
        "--shifts",
        type=parse_shifts,
        default=parse_shifts("-8,-4,-2,-1,-0.5,0,0.5,1,2,4,8"),
        help="Comma-separated rightward delays in raw EEG samples",
    )
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--feature-map", type=int, default=0)
    parser.add_argument("--example-trial", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cpu_threads < 1 or args.batch_size < 1 or args.max_trials < 0:
        raise ValueError("cpu-threads and batch-size must be positive; max-trials >= 0")
    script = Path(__file__).resolve()
    project_root = (
        args.project_root.resolve()
        if args.project_root else find_project_root(script.parent)
    )
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "src"))
    import main as training_main
    from eeg_pipeline.data.splits import build_loso_subject_folds
    from eeg_pipeline.models.factory import build_model

    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use a new directory."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Missing config or checkpoint: {config_path}, {checkpoint_path}"
        )

    os.environ.setdefault("OMP_NUM_THREADS", str(args.cpu_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.cpu_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.cpu_threads))
    torch.set_num_threads(args.cpu_threads)
    try:
        torch.set_num_interop_threads(max(1, min(4, args.cpu_threads)))
    except RuntimeError:
        pass
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    config = training_main.load_config(config_path)
    if args.data_root is not None:
        config["data"]["root"] = str(args.data_root.resolve())
    config["training"]["device"] = str(device)
    config["loader"]["batch_size"] = args.batch_size
    config["loader"]["pin_memory"] = device.type == "cuda"
    configured_folds = config["split"]["fold_indices"]
    fold_number = args.fold or (int(configured_folds[0]) + 1)
    folds = build_loso_subject_folds(config["data"]["subjects"])
    if not 1 <= fold_number <= len(folds):
        raise ValueError(f"Fold must be in 1..{len(folds)}, got {fold_number}")
    fold = folds[fold_number - 1]
    print(
        f"Loading Fold{fold_number}: val={fold.val_subject}, "
        f"test={fold.test_subject}; device={device}; threads={args.cpu_threads}"
    )
    _, _, test_loader = training_main.build_fold_loaders(config, fold)
    eeg, labels = cache_test_loader(
        test_loader, args.max_trials or None
    )
    print(f"Cached test set on CPU: eeg={tuple(eeg.shape)}, labels={tuple(labels.shape)}")

    task_classes = {"imagine": 4, "speak": 4, "visual": 4, "pattern": 3}
    n_classes = task_classes[config["data"]["task"]]
    model = build_model(
        name=config["model"]["name"],
        n_classes=n_classes,
        **config["model"].get("kwargs", {}),
    ).to(device)
    state = load_state_dict(checkpoint_path, device)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: {incompatible}")
    deformation_name, deformation = find_deformation(model)
    controller = OffsetInterventionWarp(deformation).to(device)
    replace_named_module(model, deformation_name, controller)
    model.eval()
    print(f"Checkpoint loaded; controlled module={deformation_name}")

    metadata = {
        "project_root": str(project_root),
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "model": config["model"]["name"],
        "task": config["data"]["task"],
        "fold": fold_number,
        "validation_subject": fold.val_subject,
        "test_subject": fold.test_subject,
        "n_test_trials": len(labels),
        "device": str(device),
        "cpu_threads": args.cpu_threads,
        "batch_size": args.batch_size,
        "intervention_seed": args.seed,
        "shifts_samples": args.shifts,
        "shift_definition": "positive delay: x_delta[t]=x[t-delta]",
        "ideal_offset_definition": "sample_time=t+offset, therefore ideal offset=delta",
        "torch_version": torch.__version__,
        "experiments": args.experiment,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    causal_results = None
    if args.experiment in ("both", "causal"):
        causal_results = causal_ablation(
            model, controller, eeg, labels,
            batch_size=args.batch_size, device=device, n_classes=n_classes,
            seed=args.seed, output_dir=output_dir,
            feature_map=args.feature_map, example_trial=args.example_trial,
        )
    if args.experiment in ("both", "shift"):
        artificial_shift_recovery(
            model, controller, eeg, labels,
            batch_size=args.batch_size, device=device, n_classes=n_classes,
            shifts=args.shifts, output_dir=output_dir,
            feature_map=args.feature_map, example_trial=args.example_trial,
            causal_results=causal_results,
        )
    print(f"Done. Results written to {output_dir}")


if __name__ == "__main__":
    main()
