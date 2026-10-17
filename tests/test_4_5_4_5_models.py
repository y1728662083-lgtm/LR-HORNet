"""Smoke-test the model definitions used by the Section 4.5 package."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "main.py").is_file():
    raise RuntimeError("Run this smoke test from the paper4 repository root")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_pipeline.models.factory import build_model  # noqa: E402


def build(name: str):
    common = {
        "n_channels": 8,
        "n_times": 129,
    }
    if name == "braindecode_eegnet":
        return build_model(
            name,
            n_classes=4,
            **common,
            sfreq=256.0,
            n_electrode_groups=4,
        )
    if name == "causal1_eegnet":
        return build_model(
            name,
            n_classes=4,
            **common,
            n_electrode_groups=4,
            F1=4,
            D=1,
            F2=4,
        )
    if name == "group4_causal1_deformableLstm":
        return build_model(
            name,
            n_classes=4,
            **common,
            n_electrode_groups=4,
            F1=4,
            D=1,
            F2=4,
            offset_lstm_hidden_size=8,
            offset_lstm_layers=1,
            offset_lstm_dropout=0.0,
        )
    kwargs = {
        **common,
        "patch_size": 64,
        "patch_stride": 32,
        "n_electrode_groups": 4,
        "normalization": "batchnorm",
        "F1": 4,
        "D": 1,
        "F2": 4,
    }
    if name == "group4_patchLstm_deformableLstm":
        kwargs.update(
            offset_lstm_hidden_size=8,
            offset_lstm_layers=1,
            offset_lstm_dropout=0.0,
        )
    if name == "group4_patchLstm_groupedConvDeformation":
        kwargs["offset_conv_kernel_size"] = 3
    return build_model(name, n_classes=4, **kwargs)


def main() -> None:
    torch.set_num_threads(1)
    names = (
        "braindecode_eegnet",
        "causal1_eegnet",
        "group4_causal1_deformableLstm",
        "group4_patchLstm_noDeformation",
        "group4_patchLstm_deformableLstm",
        # Supplementary control requested with 04_grouped_conv_deformation.yaml.
        "group4_patchLstm_groupedConvDeformation",
    )
    models = {name: build(name).eval() for name in names}
    eeg = torch.zeros(1, 8, 129)
    with torch.inference_mode():
        for name, model in models.items():
            logits = model(eeg)
            if logits.shape != (1, 4):
                raise RuntimeError(f"{name}: unexpected output {tuple(logits.shape)}")
            if not torch.isfinite(logits).all():
                raise RuntimeError(f"{name}: non-finite logits")

    causal = models["causal1_eegnet"]
    if causal.conv_temporal.n_electrode_groups != 4:
        raise RuntimeError("causal1_eegnet must use four temporal filter groups")

    no_patch = models["group4_causal1_deformableLstm"]
    if hasattr(no_patch, "patch_size") or not hasattr(
        no_patch.pool_1, "deformation"
    ):
        raise RuntimeError("The no-Patch control has the wrong components")

    no_deformation = models["group4_patchLstm_noDeformation"]
    if not hasattr(no_deformation, "patch_size") or hasattr(
        no_deformation.pool_1, "deformation"
    ):
        raise RuntimeError("The no-deformation control has the wrong components")

    full = models["group4_patchLstm_deformableLstm"]
    if not hasattr(full, "patch_size") or not hasattr(
        full.pool_1, "deformation"
    ):
        raise RuntimeError("The full model has the wrong components")

    grouped = models["group4_patchLstm_groupedConvDeformation"]
    if grouped.pool_1.deformation.offset_head.groups != 4:
        raise RuntimeError("Grouped-convolution offset head must use four groups")

    print("Section 4.5 model smoke test passed")
    for name, model in models.items():
        count = sum(parameter.numel() for parameter in model.parameters())
        print(f"{name}: parameters={count:,}")


def test_section_4_5_models() -> None:
    main()


if __name__ == "__main__":
    main()
