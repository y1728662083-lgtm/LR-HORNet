"""Ensure every released five-dataset LR-HORNet checkpoint remains callable."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "smoke_test_main_checkpoints.py"
SPEC = importlib.util.spec_from_file_location("smoke_test_main_checkpoints", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_all_released_main_checkpoints() -> None:
    passed = MODULE.smoke_check(
        device="cpu",
        forward_each_checkpoint=False,
        backward_per_dataset=False,
    )
    assert passed == {
        "ds003626": 10,
        "bimodal_inner_speech": 4,
        "directional_words_russian": 12,
        "directional_words_spanish": 10,
        "bci2020": 15,
    }
