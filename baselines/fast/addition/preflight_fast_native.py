"""Dependency, code-source, configuration, and plan checks for FAST reruns."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from zipfile import ZipFile

import yaml


ADDITION_ROOT = Path(__file__).resolve().parent
PAPER4_ROOT = ADDITION_ROOT.parent
CONFIG_ROOT = ADDITION_ROOT / "configs"
BIMODAL_MEMBER = "Inner_Speech_EEG_fMRI-main/EEG_preprocessing/preprocess.py"
EXPECTED = {
    "fast_native_ds003626.yaml": (256, 256, 128, 10),
    "fast_native_russian.yaml": (250, 250, 125, 12),
    "fast_native_spanish.yaml": (500, 500, 250, 10),
    "fast_native_bimodal.yaml": (512, 256, 128, 4),
}
EXPECTED_DIRECTIONAL_OPTIONS = {
    "fast_native_russian.yaml": (34, 250, [0.0, 1.0]),
    "fast_native_spanish.yaml": (38, 500, [-0.5, 1.0]),
}


def main() -> None:
    for module in (
        "torch",
        "mne",
        "numpy",
        "pandas",
        "sklearn",
        "scipy",
        "matplotlib",
        "tqdm",
        "yaml",
    ):
        loaded = importlib.import_module(module)
        print(f"DEPENDENCY_OK {module}={getattr(loaded, '__version__', 'available')}")

    required = (
        ADDITION_ROOT / "adapters.py",
        ADDITION_ROOT / "adapters_multidataset.py",
        ADDITION_ROOT / "common.py",
        ADDITION_ROOT / "training.py",
        ADDITION_ROOT / "run_official_baselines.py",
        ADDITION_ROOT / "source_manifest.json",
        ADDITION_ROOT / "run_fast_native_all_datasets.py",
        ADDITION_ROOT / "official_sources" / "FAST" / "FAST.py",
        PAPER4_ROOT / "src" / "eeg_pipeline" / "data" / "registry.py",
        PAPER4_ROOT / "src" / "eeg_pipeline" / "data" / "dataset.py",
        PAPER4_ROOT / "src" / "eeg_pipeline" / "data" / "splits.py",
        PAPER4_ROOT / "src" / "eeg_pipeline" / "data" / "directional_words.py",
        PAPER4_ROOT / "src" / "eeg_pipeline" / "data" / "bimodal_inner_speech.py",
        PAPER4_ROOT / "src" / "eeg_pipeline" / "data" / "preprocessing.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Installed runtime files are missing:\n" + "\n".join(missing))

    archive = ADDITION_ROOT / "official_preprocessing" / "bimodal_official.zip"
    with ZipFile(archive) as handle:
        if BIMODAL_MEMBER not in handle.namelist():
            raise FileNotFoundError(f"{archive} lacks {BIMODAL_MEMBER}")
    print(f"OFFICIAL_CODE_OK archive={archive} member={BIMODAL_MEMBER}")

    total = 0
    for filename, (rate, window, stride, folds) in EXPECTED.items():
        config = yaml.safe_load((CONFIG_ROOT / filename).read_text(encoding="utf-8"))
        inputs = config["input_adapter"]
        fast = config["models"]["fast"]
        assert int(inputs["source_sampling_rate_hz"]) == rate
        assert int(inputs["target_sampling_rate_hz"]) == rate
        assert int(fast["window_samples"]) == window
        assert int(fast["stride_samples"]) == stride
        assert int(fast["dim_token"]) == 32
        assert fast["source"] == "official_sources/FAST/FAST.py"
        assert len(config["split"]["fold_indices"]) == folds
        assert bool(config["split"]["adapt_held_out_subjects"])
        if filename in EXPECTED_DIRECTIONAL_OPTIONS:
            expected_channels, target_rate, time_window = (
                EXPECTED_DIRECTIONAL_OPTIONS[filename]
            )
            options = config["data"]["options"]
            assert bool(options["harmonize_within_language"])
            assert int(options["expected_common_channels"]) == expected_channels
            assert int(options["target_sampling_rate_hz"]) == target_rate
            assert options["time_window_seconds"] == time_window
        total += folds
        print(
            f"CONFIG_OK {filename} effective_rate={rate}Hz window={window} "
            f"stride={stride} dim_token=32 folds={folds}"
        )
    if total != 36:
        raise RuntimeError(f"Expected 36 FAST fits, got {total}")

    sys.path.insert(0, str(PAPER4_ROOT / "src"))
    sys.path.insert(0, str(PAPER4_ROOT))
    sys.path.insert(0, str(ADDITION_ROOT))
    from eeg_pipeline.data.registry import normalize_dataset_name
    from fast_native_adapter import nearest_window_grid_length

    assert normalize_dataset_name("directional_words") == "directional_words"
    assert normalize_dataset_name("bimodal_inner_speech") == "bimodal_inner_speech"
    assert nearest_window_grid_length(1153, 256, 128) == (1152, 0, 1)
    assert nearest_window_grid_length(359, 256, 128) == (384, 25, 0)
    print(
        "PREFLIGHT OK: official FAST source, audited dataset grids, "
        "dim_token=32, 36 fits."
    )


if __name__ == "__main__":
    main()
