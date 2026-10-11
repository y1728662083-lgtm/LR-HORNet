#!/usr/bin/env python3
"""Visualize learned warps for one fixed test trial across five checkpoints."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint-root", type=Path)
    source.add_argument(
        "--checkpoint", action="append",
        help="Repeat mapping REPEAT=PATH; pass five times for repeats 1..5",
    )
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trial-index", type=int, default=0)
    parser.add_argument("--feature-map", type=int, default=0)
    return parser.parse_args()


def resolve_checkpoints(args: argparse.Namespace) -> dict[int, Path]:
    if args.checkpoint:
        result = {}
        for item in args.checkpoint:
            repeat_text, separator, path_text = item.partition("=")
            if not separator:
                raise ValueError(f"Expected REPEAT=PATH, got {item!r}")
            repeat = int(repeat_text)
            if repeat in result:
                raise ValueError(f"Duplicate repeat: {repeat}")
            result[repeat] = Path(path_text).expanduser().resolve()
    else:
        paths = sorted(args.checkpoint_root.resolve().rglob("best_model.pt"))
        result = {}
        for repeat in range(1, 6):
            marker = f"repeat_{repeat:02d}_seed_42"
            matches = [path for path in paths if marker in path.parts and "patch_0128" in path.parts]
            if len(matches) != 1:
                raise ValueError(f"Expected one patch_0128 {marker} checkpoint, found {matches}")
            result[repeat] = matches[0]
    if set(result) != set(range(1, 6)):
        raise ValueError(f"Exactly repeats 1..5 are required, got {sorted(result)}")
    missing = [path for path in result.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    return result


def load_state(path: Path, device: torch.device):
    try:
        value = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        value = torch.load(path, map_location=device)
    for key in ("state_dict", "model_state_dict", "model"):
        if isinstance(value, dict) and isinstance(value.get(key), dict):
            value = value[key]
            break
    if value and all(str(key).startswith("module.") for key in value):
        value = {str(key)[7:]: tensor for key, tensor in value.items()}
    return value


def offsets(warp, features: torch.Tensor) -> np.ndarray:
    batch, maps, _, times = features.shape
    sequences = features[:, :, 0, :].reshape(batch * maps, times, 1)
    hidden, _ = warp.lstm(sequences)
    offset = (torch.sigmoid(warp.gate_head(hidden)) * warp.offset_head(hidden)).squeeze(-1)
    return offset.reshape(batch, maps, times).detach().cpu().numpy()


def trend_statistics(feature: np.ndarray, offset: np.ndarray) -> dict:
    t = np.arange(len(feature), dtype=float)
    sample = np.clip(t + offset, 0, len(feature) - 1)
    slope = np.gradient(feature)
    curve = np.gradient(slope)
    valid = np.abs(offset) >= 0.5
    shifted_slope = np.interp(sample, t, slope)
    shifted_curve = np.interp(sample, t, curve)
    opposite = np.clip(t - offset, 0, len(feature) - 1)
    opposite_slope = np.interp(opposite, t, slope)
    result = {
        "mean_abs_offset_samples": float(np.mean(np.abs(offset))),
        "fraction_abs_offset_ge_0_5": float(np.mean(valid)),
        "n_meaningful_positions": int(np.count_nonzero(valid)),
        "mean_abs_first_diff": float(np.mean(np.abs(slope))),
        "mean_abs_second_diff": float(np.mean(np.abs(curve))),
    }
    if valid.any():
        result.update(
            same_trend_rate=float(np.mean(np.sign(slope[valid]) == np.sign(shifted_slope[valid]))),
            opposite_direction_control_rate=float(np.mean(np.sign(slope[valid]) == np.sign(opposite_slope[valid]))),
            sampled_slope_distance=float(np.mean(np.abs(slope[valid] - shifted_slope[valid]))),
            opposite_direction_slope_distance=float(np.mean(np.abs(slope[valid] - opposite_slope[valid]))),
            sampled_curvature_distance=float(np.mean(np.abs(curve[valid] - shifted_curve[valid]))),
        )
    return result


def plot(path: Path, raw: np.ndarray, feature: np.ndarray, warped: np.ndarray,
         offset: np.ndarray, repeat: int, trial_index: int, feature_map: int) -> None:
    t = np.arange(len(feature))
    sample = np.clip(t + offset, 0, len(feature) - 1)
    slope = np.gradient(feature)
    curve = np.gradient(slope)
    fig, axes = plt.subplots(5, 1, figsize=(15, 12), sharex=False)
    axes[0].plot(np.arange(len(raw)), raw, lw=0.8)
    axes[0].set_title("Raw electrode 0 EEG (context; not the warp input)")
    axes[1].plot(t, feature, label="spatial feature before warp", lw=0.9)
    axes[1].plot(t, warped, label="feature after warp", lw=0.9, alpha=0.8)
    chosen = np.arange(0, len(feature), max(1, len(feature) // 24))
    axes[1].scatter(sample[chosen], np.interp(sample[chosen], t, feature), s=13, c="red", label="sampled positions")
    axes[1].legend()
    axes[2].plot(t, slope, lw=0.8)
    axes[2].set_title("First derivative of the pre-warp spatial feature")
    axes[3].plot(t, curve, lw=0.8)
    axes[3].set_title("Second derivative of the pre-warp spatial feature")
    axes[4].plot(t, offset, lw=0.9, label="learned offset in samples")
    axes[4].plot(t, sample - t, lw=0.6, alpha=0.7, label="effective shift after border clamp")
    axes[4].axhline(0, color="black", lw=0.5)
    axes[4].legend()
    axes[4].set_title("Temporal offset; sampling index = t + offset")
    axes[4].set_xlabel("sample index (256 Hz)")
    fig.suptitle(f"Fold 1, test trial {trial_index}, repeat {repeat}, feature map {feature_map}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "src"))
    from eeg_pipeline.data.splits import build_loso_subject_folds, build_adapted_fold_datasets
    from eeg_pipeline.models.factory import build_model

    checkpoints = resolve_checkpoints(args)
    configs = {}
    for repeat in range(1, 6):
        path = args.config_dir.resolve() / f"repeat_{repeat:02d}_seed_42.yaml"
        with path.open(encoding="utf-8") as stream:
            configs[repeat] = yaml.safe_load(stream)
        configs[repeat]["data"]["root"] = str(args.data_root.resolve())
    base = configs[1]
    fold = build_loso_subject_folds(base["data"]["subjects"])[0]
    _, _, test = build_adapted_fold_datasets(
        dataset_root=base["data"]["root"], task=base["data"]["task"], fold=fold,
        sessions=base["data"]["sessions"], seed=base["seed"],
        standardize=base["data"]["standardize"],
        n_electrode_groups=base["data"]["n_electrode_groups"],
    )
    raw, label = test[args.trial_index][:2]
    eeg = torch.as_tensor(raw, dtype=torch.float32)[None]
    requested = args.device
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    rows = []
    output = args.output_dir.resolve()
    for repeat in range(1, 6):
        config = configs[repeat]
        model = build_model(
            config["model"]["name"], n_classes=4,
            n_channels=eeg.shape[1], n_times=eeg.shape[2], **config["model"]["kwargs"],
        ).to(device)
        model.load_state_dict(load_state(checkpoints[repeat], device), strict=True)
        model.eval()
        captured = []
        hook = model.pool_1.deformation.register_forward_hook(
            lambda _module, inputs, result: captured.append((inputs[0].detach(), result.detach()))
        )
        with torch.inference_mode():
            logits = model(eeg.to(device))
        hook.remove()
        pre, post = captured[0]
        offset = offsets(model.pool_1.deformation, pre)[0, args.feature_map]
        feature = pre[0, args.feature_map, 0].cpu().numpy()
        warped = post[0, args.feature_map, 0].cpu().numpy()
        result = trend_statistics(feature, offset)
        result.update(
            repeat=repeat, seed=config["seed"], fold=1,
            validation_subject=fold.val_subject, test_subject=fold.test_subject,
            test_trial=args.trial_index, test_label=int(label),
            predicted_label=int(logits.argmax(1)), feature_map=args.feature_map,
            checkpoint=str(checkpoints[repeat]),
        )
        rows.append(result)
        plot(output / f"repeat_{repeat:02d}.png", eeg[0, 0].numpy(), feature,
             warped, offset, repeat, args.trial_index, args.feature_map)
    destination = output / "offset_trend_diagnostics.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with destination.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
