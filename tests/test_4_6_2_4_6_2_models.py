"""Smoke-test every patch setting used in Section 4.6.2."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "main.py").is_file():
    raise RuntimeError("Run this smoke test from the paper4 repository root")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_pipeline.models.factory import build_model  # noqa: E402


def main() -> None:
    torch.set_num_threads(1)
    eeg = torch.zeros(1, 8, 1153)
    settings = [(1153, 1153)] + [(size, size // 2) for size in (8, 16, 32, 64, 128, 256, 512)]
    with torch.inference_mode():
        for patch_size, patch_stride in settings:
            model = build_model(
                "group4_patchLstm_deformableLstm", n_classes=4,
                n_channels=8, n_times=1153, patch_size=patch_size,
                patch_stride=patch_stride, n_electrode_groups=4,
                offset_lstm_hidden_size=8, offset_lstm_layers=1,
                offset_lstm_dropout=0.0, normalization="batchnorm",
                F1=4, D=1, F2=4,
            ).eval()
            logits = model(eeg)
            if logits.shape != (1, 4) or not torch.isfinite(logits).all():
                raise RuntimeError(f"patch={patch_size}: invalid output {tuple(logits.shape)}")
            if model.patch_size != patch_size or model.patch_stride != patch_stride:
                raise RuntimeError(f"patch={patch_size}: model ignored patch settings")
    print("Section 4.6.2 model smoke test passed")


def test_section_4_6_2_models() -> None:
    main()


if __name__ == "__main__":
    main()
