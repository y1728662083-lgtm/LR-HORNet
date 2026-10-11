"""Smoke-test the three local feature extractors used in Section 4.6.1."""

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
    common = {"n_classes": 4, "n_channels": 8, "n_times": 129}
    models = {
        "braindecode_eegnet": build_model(
            "braindecode_eegnet", **common, sfreq=256.0, n_electrode_groups=4
        ),
        "causal1_eegnet": build_model(
            "causal1_eegnet", **common, n_electrode_groups=4, F1=4, D=1, F2=4
        ),
        "group4_patchLstm_noDeformation": build_model(
            "group4_patchLstm_noDeformation", **common, patch_size=64,
            patch_stride=32, n_electrode_groups=4, normalization="batchnorm",
            F1=4, D=1, F2=4,
        ),
    }
    eeg = torch.zeros(1, 8, 129)
    with torch.inference_mode():
        for name, model in models.items():
            logits = model.eval()(eeg)
            if logits.shape != (1, 4) or not torch.isfinite(logits).all():
                raise RuntimeError(f"{name}: invalid output {tuple(logits.shape)}")
    patch_model = models["group4_patchLstm_noDeformation"]
    if not hasattr(patch_model, "patch_size") or hasattr(patch_model.pool_1, "deformation"):
        raise RuntimeError("Patch-LSTM control unexpectedly contains deformation")
    print("Section 4.6.1 model smoke test passed")


if __name__ == "__main__":
    main()
