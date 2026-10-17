"""Smoke-test every recurrent electrode-group model used in Section 4.6.3."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "main.py").is_file():
    raise RuntimeError("Run this smoke test from the paper4 repository root")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_pipeline.models.factory import build_model  # noqa: E402


MODEL_NAMES = {
    1: "group1_patchLstm_deformableLstm",
    2: "group2_patchLstm_deformableLstm",
    4: "group4_patchLstm_deformableLstm",
    8: "group8_patchLstm_deformableLstm",
    16: "group16_patchLstm_deformableLstm",
}


def main() -> None:
    torch.set_num_threads(1)
    eeg = torch.zeros(1, 16, 129)
    counts = {}
    with torch.inference_mode():
        for groups, name in MODEL_NAMES.items():
            model = build_model(
                name, n_classes=4, n_channels=16, n_times=129,
                patch_size=64, patch_stride=32,
                n_electrode_groups=groups,
                offset_lstm_hidden_size=8, offset_lstm_layers=1,
                offset_lstm_dropout=0.0, normalization="batchnorm",
                F1=4, D=1, F2=4,
            ).eval()
            logits = model(eeg)
            if logits.shape != (1, 4) or not torch.isfinite(logits).all():
                raise RuntimeError(f"groups={groups}: invalid output {tuple(logits.shape)}")
            if model.n_electrode_groups != groups:
                raise RuntimeError(f"groups={groups}: model ignored group setting")
            if not hasattr(model.pool_1, "deformation"):
                raise RuntimeError(f"groups={groups}: deformation module is missing")
            counts[groups] = sum(parameter.numel() for parameter in model.parameters())
    if len(set(counts.values())) != len(counts):
        raise RuntimeError(f"Expected capacity to vary with groups: {counts}")
    print(f"Section 4.6.3 model smoke test passed; parameters={counts}")


def test_section_4_6_3_models() -> None:
    main()


if __name__ == "__main__":
    main()
