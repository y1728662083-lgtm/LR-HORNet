from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml


ARCHIVE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SNAPSHOT = ARCHIVE_ROOT / "project_snapshot"
for path in (PROJECT_SNAPSHOT / "src", PROJECT_SNAPSHOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eeg_pipeline.models.baseline_model import (  # noqa: E402
    build_braindecode_baseline,
)


def test_effective_configs_select_eegnet_and_expected_folds() -> None:
    expected = {
        "ds003626.yaml": 10,
        "bimodal_inner_speech.yaml": 4,
        "directional_words_russian.yaml": 12,
        "directional_words_spanish.yaml": 10,
    }
    for filename, n_folds in expected.items():
        config = yaml.safe_load(
            (ARCHIVE_ROOT / "configs" / filename).read_text(encoding="utf-8")
        )
        assert config["model"]["name"] == "braindecode_eegnet"
        assert len(config["split"]["fold_indices"]) == n_folds


def test_builds_unwrapped_official_eegnet() -> None:
    from braindecode.models import EEGNet

    model = build_braindecode_baseline(
        n_classes=4,
        architecture="EEGNet",
        n_channels=128,
        n_times=1153,
        sfreq=256.0,
        n_electrode_groups=16,
    )
    assert type(model) is EEGNet
    model.eval()
    with torch.inference_mode():
        assert model(torch.zeros(2, 128, 1153)).shape == (2, 4)

