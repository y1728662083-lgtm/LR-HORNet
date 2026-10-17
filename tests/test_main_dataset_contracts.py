"""Cross-dataset configuration, split and adapter-path contracts."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eeg_pipeline.data.bci2020 import CLASS_NAMES as BCI2020_CLASSES  # noqa: E402
from eeg_pipeline.data.bimodal_inner_speech import (  # noqa: E402
    WORD_CLASS_NAMES as BIMODAL_CLASSES,
)
from eeg_pipeline.data.directional_words import (  # noqa: E402
    COMMON_WORDS as DIRECTIONAL_CLASSES,
)
from eeg_pipeline.data.ds003626 import DIRECTION_CLASS_NAMES  # noqa: E402
from eeg_pipeline.data.registry import load_task_sessions  # noqa: E402
from eeg_pipeline.data.splits import build_loso_subject_folds  # noqa: E402


@dataclass(frozen=True)
class DatasetContract:
    config_name: str
    folds: int
    channels: int
    times: int
    classes: int
    patch_size: int
    patch_stride: int
    missing_fragments: tuple[str, ...]


CONTRACTS = (
    DatasetContract(
        "ds003626.yaml",
        10,
        128,
        1153,
        4,
        128,
        64,
        (
            "derivatives/sub-01/ses-01/sub-01_ses-01_eeg-epo.fif",
            "derivatives/sub-01/ses-01/sub-01_ses-01_events.dat",
        ),
    ),
    DatasetContract(
        "bimodal_inner_speech.yaml",
        4,
        64,
        359,
        8,
        256,
        128,
        ("derivatives/paper4/bimodal/sub-01/sub-01_task-inner_epo.fif",),
    ),
    DatasetContract(
        "directional_words_russian.yaml",
        12,
        34,
        250,
        6,
        125,
        63,
        ("preprocessed/russian/sub1/sub1_epochs.fif",),
    ),
    DatasetContract(
        "directional_words_spanish.yaml",
        10,
        38,
        750,
        6,
        250,
        125,
        ("preprocessed/spanish/sub0/sub0_epochs.fif",),
    ),
    DatasetContract(
        "bci2020.yaml",
        15,
        64,
        795,
        5,
        128,
        64,
        (
            "derivatives/paper4/bci2020_standard/sub-01/eeg/"
            "sub-01_task-imaginedSpeech_run-00_desc-preproc_epo.fif",
        ),
    ),
)


def _read_config(contract: DatasetContract) -> dict:
    with (ROOT / "configs" / "main" / contract.config_name).open(
        encoding="utf-8"
    ) as stream:
        return yaml.safe_load(stream)


def _first_session(config: dict):
    sessions = config["data"]["sessions"]
    return "auto" if sessions == "auto" else [sessions[0]]


def _load_from_root(config: dict, root: Path):
    data = config["data"]
    return load_task_sessions(
        data.get("dataset"),
        root,
        data["task"],
        subjects=[data["subjects"][0]],
        sessions=_first_session(config),
        preload=False,
        options=data.get("options"),
    )


@pytest.mark.parametrize("contract", CONTRACTS, ids=lambda item: item.config_name)
def test_main_configuration_and_loso_contract(contract: DatasetContract) -> None:
    config = _read_config(contract)
    subjects = tuple(config["data"]["subjects"])

    assert len(subjects) == contract.folds
    assert config["split"]["fold_indices"] == list(range(contract.folds))
    assert config["split"]["adapt_held_out_subjects"] is True
    kwargs = config["model"]["kwargs"]
    assert (kwargs["patch_size"], kwargs["patch_stride"]) == (
        contract.patch_size,
        contract.patch_stride,
    )

    folds = build_loso_subject_folds(subjects)
    assert len(folds) == contract.folds
    assert tuple(fold.test_subject for fold in folds) == subjects
    assert tuple(fold.val_subject for fold in folds) == subjects[1:] + subjects[:1]
    assert all(len(fold.train_subjects) == contract.folds - 2 for fold in folds)


def test_dataset_class_counts() -> None:
    assert len(DIRECTION_CLASS_NAMES) == CONTRACTS[0].classes
    assert len(BIMODAL_CLASSES) == CONTRACTS[1].classes
    assert len(DIRECTIONAL_CLASSES) == CONTRACTS[2].classes
    assert len(DIRECTIONAL_CLASSES) == CONTRACTS[3].classes
    assert len(BCI2020_CLASSES) == CONTRACTS[4].classes


@pytest.mark.parametrize("contract", CONTRACTS, ids=lambda item: item.config_name)
def test_missing_dataset_root_is_explicit(
    contract: DatasetContract, tmp_path: Path
) -> None:
    config = _read_config(contract)
    missing_root = tmp_path / "missing-dataset-root"
    with pytest.raises(FileNotFoundError) as error:
        _load_from_root(config, missing_root)
    assert str(missing_root) in str(error.value)


@pytest.mark.parametrize("contract", CONTRACTS, ids=lambda item: item.config_name)
def test_missing_derivative_path_is_explicit(
    contract: DatasetContract, tmp_path: Path
) -> None:
    config = _read_config(contract)
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    with pytest.raises(FileNotFoundError) as error:
        _load_from_root(config, dataset_root)
    normalized_message = str(error.value).replace("\\", "/")
    for fragment in contract.missing_fragments:
        assert fragment in normalized_message


@pytest.mark.parametrize(
    ("channels", "times", "message"),
    ((63, 359, "64 EEG channels"), (64, 358, "359 samples")),
)
def test_bimodal_rejects_incompatible_preprocessing(
    tmp_path: Path,
    channels: int,
    times: int,
    message: str,
) -> None:
    config = _read_config(CONTRACTS[1])
    output = (
        tmp_path
        / "derivatives"
        / "paper4"
        / "bimodal"
        / "sub-01"
        / "sub-01_task-inner_epo.fif"
    )
    output.parent.mkdir(parents=True)
    info = mne.create_info(
        [f"EEG{index:03d}" for index in range(channels)],
        sfreq=512.0,
        ch_types="eeg",
    )
    epochs = mne.EpochsArray(
        np.zeros((1, channels, times), dtype=np.float64),
        info,
        events=np.asarray([[0, 0, 1]], dtype=np.int64),
        event_id={"child": 1},
        tmin=0.0,
        baseline=None,
        verbose="ERROR",
    )
    epochs.save(output, overwrite=True, verbose="ERROR")

    with pytest.raises(RuntimeError, match=message):
        _load_from_root(config, tmp_path)
