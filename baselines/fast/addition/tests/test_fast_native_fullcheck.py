from __future__ import annotations

import sys
from pathlib import Path
from zipfile import ZipFile

import yaml


ADDITION_ROOT = Path(__file__).resolve().parents[1]
PAPER4_ROOT = ADDITION_ROOT.parent
sys.path.insert(0, str(ADDITION_ROOT))

from dispatch_fast_native import prepare_jobs
from fast_native_adapter import nearest_window_grid_length


def test_complete_fast_plan_has_four_datasets_and_36_fits():
    jobs = prepare_jobs()
    assert len(jobs) == 36
    assert sum(job.dataset == "ds003626" for job in jobs) == 10
    assert sum(job.dataset == "russian" for job in jobs) == 12
    assert sum(job.dataset == "spanish" for job in jobs) == 10
    assert sum(job.dataset == "bimodal" for job in jobs) == 4


def test_all_configs_keep_calibrated_loso_native_rate_and_official_dim_token():
    for path in sorted((ADDITION_ROOT / "configs").glob("fast_native_*.yaml")):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        inputs = config["input_adapter"]
        fast = config["models"]["fast"]
        assert inputs["source_sampling_rate_hz"] == inputs["target_sampling_rate_hz"]
        assert fast["dim_token"] == 32
        assert fast["source"] == "official_sources/FAST/FAST.py"
        assert fast["window_remainder"] == "nearest"
        assert config["split"]["adapt_held_out_subjects"] is True


def test_directional_configs_pin_the_audited_intersection_and_time_grid():
    expected = {
        "fast_native_russian.yaml": (34, 250, [0.0, 1.0]),
        "fast_native_spanish.yaml": (38, 500, [-0.5, 1.0]),
    }
    for filename, (channels, rate, window) in expected.items():
        path = ADDITION_ROOT / "configs" / filename
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        options = config["data"]["options"]
        assert options["harmonize_within_language"] is True
        assert options["expected_common_channels"] == channels
        assert options["target_sampling_rate_hz"] == rate
        assert options["time_window_seconds"] == window


def test_native_tail_policy_is_deterministic():
    assert nearest_window_grid_length(1153, 256, 128) == (1152, 0, 1)
    assert nearest_window_grid_length(250, 250, 125) == (250, 0, 0)
    assert nearest_window_grid_length(750, 500, 250) == (750, 0, 0)
    assert nearest_window_grid_length(359, 256, 128) == (384, 25, 0)


def test_bimodal_official_preprocessing_source_is_bundled():
    archive = ADDITION_ROOT / "official_preprocessing" / "bimodal_official.zip"
    member = "Inner_Speech_EEG_fMRI-main/EEG_preprocessing/preprocess.py"
    with ZipFile(archive) as handle:
        assert member in handle.namelist()


def test_self_contained_payload_has_multidataset_runtime():
    required = (
        ADDITION_ROOT / "adapters.py",
        ADDITION_ROOT / "adapters_multidataset.py",
        ADDITION_ROOT / "run_official_baselines.py",
        ADDITION_ROOT / "official_sources" / "FAST" / "FAST.py",
        PAPER4_ROOT / "src" / "eeg_pipeline" / "data" / "registry.py",
        PAPER4_ROOT / "src" / "eeg_pipeline" / "data" / "directional_words.py",
        PAPER4_ROOT / "src" / "eeg_pipeline" / "data" / "bimodal_inner_speech.py",
        PAPER4_ROOT / "src" / "eeg_pipeline" / "data" / "preprocessing.py",
    )
    assert all(path.is_file() for path in required)
