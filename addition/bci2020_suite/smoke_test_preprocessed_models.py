"""CPU smoke test for BCI2020 data plus four locally dependency-safe models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml


PAPER4_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAPER4_ROOT / "src"))
sys.path.insert(0, str(PAPER4_ROOT))


def count(model: torch.nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--derivatives-root",
        default="derivatives/paper4/bci2020_standard",
        help="Relative to dataset root, or an absolute local smoke-output path.",
    )
    parser.add_argument("--subject", type=int, default=1)
    args = parser.parse_args()

    from eeg_pipeline.data.dataset import EEGTaskDataset
    from eeg_pipeline.data.registry import load_task_sessions
    from eeg_pipeline.models.baseline_model import build_braindecode_baseline
    from eeg_pipeline.models.factory_lr_hornet import build_model

    sessions = load_task_sessions(
        "bci2020",
        args.dataset_root,
        "imagined_speech",
        [args.subject],
        [0, 1, 2],
        preload=True,
        options={"derivatives_root": args.derivatives_root},
    )
    dataset = EEGTaskDataset(sessions, standardize=True, n_electrode_groups=4)
    x, _ = dataset[0]
    x = x.unsqueeze(0)
    if tuple(x.shape[1:]) != (64, 795):
        raise RuntimeError(f"Unexpected BCI2020 shape: {tuple(x.shape)}")
    reports: dict[str, dict] = {}

    lr_config = yaml.safe_load((PAPER4_ROOT / "configs/main/bci2020.yaml").read_text(encoding="utf-8"))
    lr = build_model(
        lr_config["model"]["name"],
        n_classes=5,
        n_channels=64,
        n_times=795,
        **lr_config["model"]["kwargs"],
    ).eval()
    with torch.inference_mode():
        output = lr(x)
    if tuple(output.shape) != (1, 5):
        raise RuntimeError(f"LR-HORNet output shape: {tuple(output.shape)}")
    reports["lr_hornet"] = {**count(lr), "output_shape": list(output.shape)}

    for architecture, key in (("EEGNet", "eegnet"), ("ATCNet", "atcnet")):
        model = build_braindecode_baseline(
            n_classes=5,
            architecture=architecture,
            n_channels=64,
            n_times=795,
            sfreq=256.0,
            n_electrode_groups=16,
        ).eval()
        with torch.inference_mode():
            output = model(x)
        if tuple(output.shape) != (1, 5):
            raise RuntimeError(f"{architecture} output shape: {tuple(output.shape)}")
        reports[key] = {**count(model), "output_shape": list(output.shape)}

    fast_root = PAPER4_ROOT / "baselines/fast"
    for path in (fast_root / "addition", fast_root / "src", fast_root):
        sys.path.insert(0, str(path))
    from fast_native_adapter import build_fast_native_model

    fast_config = yaml.safe_load(
        (fast_root / "addition/configs/fast_native_bci2020.yaml").read_text(encoding="utf-8")
    )
    fast = build_fast_native_model(
        "fast",
        fast_config["models"]["fast"],
        channel_names=dataset.channel_names,
        n_times=795,
        n_classes=5,
        input_config=fast_config["input_adapter"],
    ).eval()
    with torch.inference_mode():
        output = fast(x)
    if tuple(output.shape) != (1, 5):
        raise RuntimeError(f"FAST output shape: {tuple(output.shape)}")
    reports["fast"] = {
        **count(fast),
        "output_shape": list(output.shape),
        "right_padding": fast.right_padding,
        "right_crop": fast.right_crop,
    }

    print(
        json.dumps(
            {
                "status": "passed",
                "records": len(sessions),
                "trials": len(dataset),
                "input_shape": list(x.shape),
                "models": reports,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
