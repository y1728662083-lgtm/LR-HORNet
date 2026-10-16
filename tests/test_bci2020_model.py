"""Regression test for the BCI2020 LR-HORNet model contract."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_pipeline.models._patch_boundary import (  # noqa: E402
    fit_right_boundary,
    patch_layout,
)
from eeg_pipeline.models.factory_lr_hornet import build_model  # noqa: E402


def _load_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "main" / "bci2020.yaml"
    with config_path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def test_bci2020_patch_boundary() -> None:
    config = _load_config()
    patch_timing = config["patch_timing"]
    model_kwargs = config["model"]["kwargs"]

    assert patch_timing["sampling_rate_hz"] == 256
    assert patch_timing["patch_seconds"] == 0.5
    assert patch_timing["stride_seconds"] == 0.25
    assert patch_timing["right_pad_incomplete_patch"] is True
    assert model_kwargs["patch_size"] == 128
    assert model_kwargs["patch_stride"] == 64
    assert model_kwargs["pad_incomplete_patch"] is True

    patch_count, effective_n_times = patch_layout(
        795,
        model_kwargs["patch_size"],
        model_kwargs["patch_stride"],
        pad_incomplete_patch=model_kwargs["pad_incomplete_patch"],
    )
    assert patch_count == 12
    assert effective_n_times == 832
    assert effective_n_times - 795 == 37

    signal = torch.arange(795, dtype=torch.float32).reshape(1, 1, 1, 795)
    padded = fit_right_boundary(signal, effective_n_times)
    assert padded.shape == (1, 1, 1, 832)
    torch.testing.assert_close(padded[..., :795], signal)
    torch.testing.assert_close(padded[..., 795:], torch.zeros(1, 1, 1, 37))


def test_bci2020_model() -> None:
    config = _load_config()

    model_config = config["model"]
    model_kwargs = dict(model_config["kwargs"])
    assert model_kwargs["pad_incomplete_patch"] is True

    model = build_model(
        model_config["name"],
        n_classes=5,
        n_channels=64,
        n_times=795,
        **model_kwargs,
    ).eval()
    assert model.pad_incomplete_patch is True
    assert model.patch_count == 12
    assert model.effective_n_times == 832
    assert model.right_padding == 37
    assert model.discarded_tail == 0
    assert sum(parameter.numel() for parameter in model.parameters()) == 5479

    torch.manual_seed(42)
    with torch.inference_mode():
        logits = model(torch.randn(1, 64, 795))
    assert logits.shape == (1, 5)
    assert torch.isfinite(logits).all()


def main() -> None:
    torch.set_num_threads(1)
    test_bci2020_patch_boundary()
    test_bci2020_model()
    print("BCI2020 LR-HORNet model test passed")


if __name__ == "__main__":
    main()
