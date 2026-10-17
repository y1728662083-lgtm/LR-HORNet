"""Dependency-light checks for repository-relative dataset paths."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path, PurePosixPath, PureWindowsPath
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portable_paths import resolve_config_data_root  # noqa: E402


DATA_ROOTS = {
    "ds003626": "data/ds003626",
    "bimodal_inner_speech": "data/bimodal_inner_speech",
    "directional_words_russian": "data/directional_words",
    "directional_words_spanish": "data/directional_words",
    "bci2020": "data/bci2020",
}


def _baseline_configs() -> tuple[tuple[str, str], ...]:
    cases: list[tuple[str, str]] = []
    for model in ("eegnet", "atcnet"):
        cases.extend(
            (dataset, f"baselines/{model}/configs/{dataset}.yaml")
            for dataset in DATA_ROOTS
        )
    for model in ("eegmamba", "labram"):
        cases.extend(
            (
                dataset,
                "baselines/"
                f"{model}/addition/configs/{dataset}_official_baselines.yaml",
            )
            for dataset in DATA_ROOTS
        )
    fast_names = {
        "ds003626": "ds003626",
        "bimodal_inner_speech": "bimodal",
        "directional_words_russian": "russian",
        "directional_words_spanish": "spanish",
        "bci2020": "bci2020",
    }
    cases.extend(
        (
            dataset,
            f"baselines/fast/addition/configs/fast_native_{suffix}.yaml",
        )
        for dataset, suffix in fast_names.items()
    )
    return tuple(cases)


CONFIGS = tuple(
    (dataset, f"configs/main/{dataset}.yaml") for dataset in DATA_ROOTS
) + _baseline_configs()


@pytest.mark.parametrize(("dataset", "relative"), CONFIGS)
def test_all_dataset_roots_have_deterministic_resolution(
    dataset: str,
    relative: str,
) -> None:
    path = ROOT / relative
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    original = str(config["data"]["root"])
    resolved = deepcopy(config)
    resolve_config_data_root(resolved, repository_root=ROOT)

    is_absolute = (
        Path(original).is_absolute()
        or PurePosixPath(original).is_absolute()
        or PureWindowsPath(original).is_absolute()
    )
    if is_absolute:
        assert resolved["data"]["root"] == original
    else:
        assert original.replace("\\", "/") == DATA_ROOTS[dataset]
        assert Path(resolved["data"]["root"]) == ROOT / DATA_ROOTS[dataset]


def test_all_python_baseline_runners_apply_repository_resolution() -> None:
    runners = (
        "src/eeg_pipeline/models/baseline_model.py",
        "baselines/eegmamba/addition/run_official_baselines.py",
        "baselines/eegmamba/addition/run_multidataset_official_baselines.py",
        "baselines/labram/addition/run_official_baselines.py",
        "baselines/labram/addition/run_multidataset_official_baselines.py",
        "baselines/fast/addition/run_official_baselines.py",
        "baselines/fast/addition/run_fast_native_all_datasets.py",
        "baselines/fast/addition/check_fast_dataset_inputs.py",
    )
    for relative in runners:
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "resolve_config_data_root(config" in source, relative


def test_braindecode_launchers_start_from_repository_root() -> None:
    for model in ("eegnet", "atcnet"):
        source = (
            ROOT / "baselines" / model / "scripts" / "run_all_datasets.sh"
        ).read_text(encoding="utf-8")
        assert 'cd "$PAPER4_ROOT"' in source


def test_braindecode_bimodal_preprocessing_reference_exists() -> None:
    for model in ("eegnet", "atcnet"):
        path = ROOT / "baselines" / model / "configs" / "bimodal_inner_speech.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        preprocessing = config["preprocessing"]
        source = ROOT / preprocessing["official_code_source"]
        assert source.is_dir(), source
        assert (source / "preprocess.py").is_file()
        assert "official_code_zip" not in preprocessing


def test_foreign_absolute_roots_are_not_rewritten() -> None:
    historical_linux_root = "/root/" + "autodl-tmp/BCI_2020"
    for value in (historical_linux_root, r"F:\\EEG\\BCI_2020"):
        config = {"data": {"root": value}}
        resolve_config_data_root(config, repository_root=ROOT)
        assert config["data"]["root"] == value
