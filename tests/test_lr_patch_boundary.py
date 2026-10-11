"""Verify LR Patch boundary handling without loading the full model stack."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_pipeline.models._patch_boundary import (  # noqa: E402
    fit_right_boundary,
    patch_layout,
)


def main() -> None:
    signal = torch.arange(32, dtype=torch.float32).reshape(1, 1, 4, 8)

    crop_count, crop_length = patch_layout(
        8, 4, 3, pad_incomplete_patch=False
    )
    if (crop_count, crop_length) != (2, 7):
        raise RuntimeError(f"Unexpected crop layout: {(crop_count, crop_length)}")
    cropped = fit_right_boundary(signal, crop_length)
    if cropped.shape != (1, 1, 4, 7):
        raise RuntimeError(f"Unexpected cropped shape: {tuple(cropped.shape)}")
    torch.testing.assert_close(cropped, signal[..., :7])

    pad_count, pad_length = patch_layout(
        8, 4, 3, pad_incomplete_patch=True
    )
    if (pad_count, pad_length) != (3, 10):
        raise RuntimeError(f"Unexpected pad layout: {(pad_count, pad_length)}")
    padded = fit_right_boundary(signal, pad_length)
    if padded.shape != (1, 1, 4, 10):
        raise RuntimeError(f"Unexpected padded shape: {tuple(padded.shape)}")
    torch.testing.assert_close(padded[..., :8], signal)
    torch.testing.assert_close(padded[..., 8:], torch.zeros(1, 1, 4, 2))

    exact_signal = torch.arange(40, dtype=torch.float32).reshape(1, 1, 4, 10)
    exact_count, exact_length = patch_layout(
        10, 4, 3, pad_incomplete_patch=True
    )
    if (exact_count, exact_length) != (3, 10):
        raise RuntimeError(
            f"Unexpected exact layout: {(exact_count, exact_length)}"
        )
    exact = fit_right_boundary(exact_signal, exact_length)
    if exact.shape != (1, 1, 4, 10):
        raise RuntimeError(f"Unexpected exact-boundary shape: {tuple(exact.shape)}")
    torch.testing.assert_close(exact, exact_signal)

    print("LR Patch boundary tests passed")


if __name__ == "__main__":
    main()
