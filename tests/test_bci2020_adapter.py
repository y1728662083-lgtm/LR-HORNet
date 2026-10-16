"""Regression test for the standard-preprocessed BCI2020 adapter."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import mne
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_pipeline.data.bci2020 import CLASS_NAMES, EVENT_ID  # noqa: E402
from eeg_pipeline.data.dataset import EEGTaskDataset  # noqa: E402
from eeg_pipeline.data.registry import (  # noqa: E402
    load_task_sessions,
    normalize_dataset_name,
)
from eeg_pipeline.data.splits import build_loso_subject_folds  # noqa: E402


def _write_minimal_preprocessed_fif(dataset_root: Path) -> Path:
    montage = mne.channels.make_standard_montage("biosemi64")
    info = mne.create_info(
        ch_names=montage.ch_names,
        sfreq=256.0,
        ch_types="eeg",
    )
    info.set_montage(montage)

    rng = np.random.default_rng(42)
    data = rng.normal(0.0, 1e-6, size=(5, 64, 795))
    events = np.column_stack(
        (
            np.arange(5, dtype=np.int64) * 795,
            np.zeros(5, dtype=np.int64),
            np.arange(1, 6, dtype=np.int64),
        )
    )
    epochs = mne.EpochsArray(
        data,
        info,
        events=events,
        event_id=EVENT_ID,
        tmin=0.0,
        baseline=None,
        verbose="ERROR",
    )

    output = (
        dataset_root
        / "derivatives"
        / "paper4"
        / "bci2020_standard"
        / "sub-01"
        / "eeg"
        / "sub-01_task-imaginedSpeech_run-00_desc-preproc_epo.fif"
    )
    output.parent.mkdir(parents=True)
    epochs.save(output, overwrite=True, verbose="ERROR")
    return output


def _check_adapter(dataset_root: Path) -> None:
    _write_minimal_preprocessed_fif(dataset_root)

    assert normalize_dataset_name("bcic2020_track3") == "bci2020"
    task_sessions = load_task_sessions(
        "bci2020",
        dataset_root,
        "imagined_speech",
        subjects=[1],
        sessions=[0],
        preload=True,
    )

    assert len(task_sessions) == 1
    task_data = task_sessions[0]
    assert task_data.class_names == CLASS_NAMES
    np.testing.assert_array_equal(task_data.labels, np.arange(5))
    assert len(task_data.epochs.ch_names) == 64
    assert len(task_data.epochs.times) == 795
    assert task_data.sampling_rate_hz == 256.0

    dataset = EEGTaskDataset(
        task_sessions,
        standardize=False,
        n_electrode_groups=4,
    )
    assert len(dataset) == 5
    assert dataset.class_names == CLASS_NAMES
    assert dataset.labels == (0, 1, 2, 3, 4)
    assert dataset.num_channels == 64
    assert dataset.num_times == 795

    for index in range(5):
        eeg, label = dataset[index]
        assert eeg.shape == (64, 795)
        assert eeg.dtype == torch.float32
        assert label.dtype == torch.long
        assert label.item() == index


def test_bci2020_adapter(tmp_path: Path) -> None:
    _check_adapter(tmp_path)


def test_bci2020_loso_contract() -> None:
    subjects = tuple(range(1, 16))
    folds = build_loso_subject_folds(subjects)

    assert len(folds) == 15
    assert tuple(fold.fold_index for fold in folds) == tuple(range(15))
    assert tuple(fold.test_subject for fold in folds) == subjects
    assert tuple(fold.val_subject for fold in folds) == subjects[1:] + (1,)

    expected_subjects = set(subjects)
    for fold in folds:
        assert len(fold.train_subjects) == 13
        assert fold.test_subject not in fold.train_subjects
        assert fold.val_subject not in fold.train_subjects
        assert fold.test_subject != fold.val_subject
        assert (
            set(fold.train_subjects)
            | {fold.val_subject, fold.test_subject}
        ) == expected_subjects


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="bci2020-adapter-") as directory:
        _check_adapter(Path(directory))
    test_bci2020_loso_contract()
    print("BCI2020 adapter test passed")


if __name__ == "__main__":
    main()
