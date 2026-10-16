"""Subject-level split definitions shared by all supported EEG datasets."""

from __future__ import annotations

import operator
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, Dataset, Subset

if TYPE_CHECKING:
    from .dataset import EEGTaskDataset


@dataclass(frozen=True)
class SubjectFold:
    """Subject roles for one outer LOSO-based evaluation fold."""

    fold_index: int
    train_subjects: tuple[object, ...]
    val_subject: object
    test_subject: object

'''旧版LOSO留一验证
def build_loso_subject_folds(
    subjects: Sequence[int],
) -> tuple[SubjectFold, ...]:
    """Build deterministic train/validation/test subject roles.

    Each subject is used once as the test subject. The next subject in the
    provided order is used for validation, with cyclic wraparound. All other
    subjects form the base training group.
    """
    if isinstance(subjects, (str, bytes)) or not isinstance(subjects, Sequence):
        raise TypeError("subjects must be a sequence of integer identifiers")

    subject_ids: list[int] = []
    for subject in subjects:
        if isinstance(subject, bool):
            raise TypeError("Boolean values are not valid subject identifiers")
        try:
            subject_id = operator.index(subject)
        except TypeError as error:
            raise TypeError(
                f"Subject identifier must be an integer, got {subject!r}"
            ) from error
        if not 1 <= subject_id <= 10:
            raise ValueError(
                f"ds003626 subject must be between 1 and 10, got {subject_id}"
            )
        subject_ids.append(subject_id)

    ordered_subjects = tuple(subject_ids)
    if len(ordered_subjects) < 3:
        raise ValueError("At least three subjects are required for train/val/test")
    if len(set(ordered_subjects)) != len(ordered_subjects):
        raise ValueError(f"subjects contains duplicates: {ordered_subjects}")

    all_subjects = set(ordered_subjects)
    folds: list[SubjectFold] = []
    for fold_index, test_subject in enumerate(ordered_subjects):
        val_subject = ordered_subjects[(fold_index + 1) % len(ordered_subjects)]
        excluded_subjects = {val_subject, test_subject}
        train_subjects = tuple(
            subject
            for subject in ordered_subjects
            if subject not in excluded_subjects
        )

        train_set = set(train_subjects)
        if train_set & excluded_subjects:
            raise RuntimeError(f"Subject overlap detected in fold {fold_index}")
        if train_set | excluded_subjects != all_subjects:
            raise RuntimeError(f"Subject coverage mismatch in fold {fold_index}")

        folds.append(
            SubjectFold(
                fold_index=fold_index,
                train_subjects=train_subjects,
                val_subject=val_subject,
                test_subject=test_subject,
            )
        )

    return tuple(folds)
'''

def build_loso_subject_folds(
    subjects: Sequence[int],
) -> tuple[SubjectFold, ...]:
    """Build cyclic train/validation/test subject folds."""
    subjects = tuple(subjects)

    if len(subjects) < 3 or len(set(subjects)) != len(subjects):
        raise ValueError("subjects must contain at least three unique values")

    folds = []
    for index, test_subject in enumerate(subjects):
        val_subject = subjects[(index + 1) % len(subjects)]
        train_subjects = tuple(
            subject
            for subject in subjects
            if subject not in (test_subject, val_subject)
        )
        folds.append(
            SubjectFold(index, train_subjects, val_subject, test_subject)
        )

    return tuple(folds)


def split_calibration_dataset(
    dataset: EEGTaskDataset,
    *,
    seed: int = 42,
) -> tuple[Subset, Subset]:
    """Split a dataset equally by label for calibration and evaluation."""
    indices = np.arange(len(dataset))
    calibration_indices, evaluation_indices = train_test_split(
        indices,
        test_size=0.5,
        random_state=seed,
        stratify=dataset.labels,
    )
    return (
        Subset(dataset, calibration_indices.tolist()),
        Subset(dataset, evaluation_indices.tolist()),
    )


def build_adapted_fold_datasets(
    dataset_root,
    task: str,
    fold: SubjectFold,
    *,
    dataset_name: str = "ds003626",
    sessions: Sequence = (1, 2, 3),
    dataset_options: dict | None = None,
    seed: int = 42,
    standardize: bool = False,
    n_electrode_groups: int = 1,
    adapt_held_out_subjects: bool = True,
    transform=None,
) -> tuple[Dataset, Dataset, Dataset]:
    """Build one fold with a shared spatial electrode grouping layout."""
    from eeg_pipeline.data.dataset import EEGTaskDataset
    from eeg_pipeline.data.registry import load_task_sessions

    def load_dataset(subjects):
        task_sessions = load_task_sessions(
            dataset_name,
            dataset_root,
            task,
            subjects=subjects,
            sessions=sessions,
            preload=False,
            options=dataset_options,
        )
        return EEGTaskDataset(
            task_sessions,
            standardize=standardize,
            n_electrode_groups=n_electrode_groups,
            transform=transform,
        )

    base_train = load_dataset(fold.train_subjects)
    val_full = load_dataset((fold.val_subject,))
    test_full = load_dataset((fold.test_subject,))

    for split_name, split_dataset in (
        ("validation", val_full),
        ("test", test_full),
    ):
        if split_dataset.channel_names != base_train.channel_names:
            raise RuntimeError(
                f"{split_name} channel order differs from training order"
            )
        if split_dataset.electrode_group_sizes != (
            base_train.electrode_group_sizes
        ):
            raise RuntimeError(
                f"{split_name} electrode groups differ from training groups"
            )

    if not adapt_held_out_subjects:
        return base_train, val_full, test_full

    val_calibration, val_dataset = split_calibration_dataset(
        val_full, seed=seed
    )
    test_calibration, test_dataset = split_calibration_dataset(
        test_full, seed=seed + 1
    )

    train_dataset = ConcatDataset(
        [base_train, val_calibration, test_calibration]
    )
    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from eeg_pipeline.data.dataset import EEGTaskDataset
    from eeg_pipeline.data.ds003626 import load_task_sessions

    dataset_subjects = tuple(range(1, 11))
    subject_folds = build_loso_subject_folds(dataset_subjects)

    if len(subject_folds) != 10:
        raise RuntimeError(f"Expected 10 folds, got {len(subject_folds)}")
    if tuple(fold.test_subject for fold in subject_folds) != dataset_subjects:
        raise RuntimeError("Each subject must be the test subject exactly once")
    expected_val_subjects = dataset_subjects[1:] + dataset_subjects[:1]
    if tuple(fold.val_subject for fold in subject_folds) != expected_val_subjects:
        raise RuntimeError("Validation subjects do not follow cyclic order")
    if any(len(fold.train_subjects) != 8 for fold in subject_folds):
        raise RuntimeError("Each ds003626 fold must have eight train subjects")

    print("LOSO subject-fold test passed")
    print(f"folds={len(subject_folds)}")
    print(f"first={subject_folds[0]}")
    print(f"last={subject_folds[-1]}")

    dataset_root = Path(r"F:\脑电数据集\dataset\OpenNeruro_ds003626")
    task_sessions = load_task_sessions(
        dataset_root,
        task="imagine",
        subjects=(1,),
        sessions=(1, 2, 3),
        preload=False,
    )
    subject_dataset = EEGTaskDataset(task_sessions)
    calibration_dataset, evaluation_dataset = split_calibration_dataset(
        subject_dataset,
        seed=42,
    )

    total_trials = len(subject_dataset)
    split_indices = set(calibration_dataset.indices) | set(
        evaluation_dataset.indices
    )
    if len(split_indices) != total_trials:
        raise RuntimeError("Split contains missing or overlapping trials")
    if abs(len(calibration_dataset) - len(evaluation_dataset)) > 1:
        raise RuntimeError("Dataset was not split approximately in half")

    calibration_labels = [
        subject_dataset.labels[index]
        for index in calibration_dataset.indices
    ]
    evaluation_labels = [
        subject_dataset.labels[index]
        for index in evaluation_dataset.indices
    ]
    expected_counts = [25, 25, 25, 25]
    if np.bincount(calibration_labels).tolist() != expected_counts:
        raise RuntimeError("Calibration labels are not balanced")
    if np.bincount(evaluation_labels).tolist() != expected_counts:
        raise RuntimeError("Evaluation labels are not balanced")

    calibration_eeg, calibration_label = calibration_dataset[0]
    evaluation_eeg, evaluation_label = evaluation_dataset[0]
    print("calibration/evaluation split test passed")
    print(
        f"calibration={len(calibration_dataset)}, "
        f"evaluation={len(evaluation_dataset)}"
    )
    print(
        f"calibration sample={tuple(calibration_eeg.shape)}, "
        f"label={calibration_label.item()}"
    )
    print(
        f"evaluation sample={tuple(evaluation_eeg.shape)}, "
        f"label={evaluation_label.item()}"
    )

    train_dataset, val_dataset, test_dataset = build_adapted_fold_datasets(
        dataset_root,
        task="imagine",
        fold=subject_folds[0],
        seed=42,
        n_electrode_groups=8,
    )
    for name, dataset in (
        ("train", train_dataset),
        ("val", val_dataset),
        ("test", test_dataset),
    ):
        eeg, label = dataset[0]
        if eeg.shape != (128, 1153) or label.ndim != 0:
            raise RuntimeError(
                f"Unexpected {name} sample: eeg={eeg.shape}, "
                f"label={label.shape}"
            )

    print("adapted LOSO fold dataset test passed")
    print(
        f"train={len(train_dataset)}, val={len(val_dataset)}, "
        f"test={len(test_dataset)}"
    )
