"""Reader for the LTU bimodal EEG/fMRI inner-speech dataset."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from .ds003626 import DS003626TaskData


WORD_CLASS_NAMES = (
    "child",
    "daughter",
    "father",
    "wife",
    "four",
    "three",
    "ten",
    "six",
)
SOCIAL_WORDS = frozenset(WORD_CLASS_NAMES[:4])
CATEGORY_CLASS_NAMES = ("social", "number")


def _normalize_subject(subject: str | int) -> str:
    value = str(subject).strip().lower()
    if value.startswith("sub-"):
        value = value[4:]
    if not value.isdecimal() or int(value) not in {1, 2, 3, 5}:
        raise ValueError(
            "Bimodal inner-speech subjects must be one of 1, 2, 3, 5; "
            f"got {subject!r}"
        )
    return f"sub-{int(value):02d}"


def _derivatives_root(root: Path, options: Mapping | None) -> Path:
    configured = (options or {}).get("derivatives_root")
    if configured is None:
        return root / "derivatives" / "paper4" / "bimodal"
    path = Path(configured).expanduser()
    return path if path.is_absolute() else root / path


def _normalize_task(task: str) -> str:
    aliases = {
        "inner_word": "inner_word",
        "word": "inner_word",
        "inner_category": "inner_category",
        "category": "inner_category",
    }
    try:
        return aliases[task.strip().lower()]
    except KeyError:
        raise ValueError(
            "Bimodal task must be 'inner_word' or 'inner_category'"
        ) from None


def load_task_sessions(
    dataset_root: str | Path,
    task: str,
    subjects: Sequence[str | int],
    sessions: Sequence[str | int] | str = ("EEG",),
    *,
    preload: bool = False,
    options: Mapping | None = None,
) -> tuple[DS003626TaskData, ...]:
    """Load official-code-derived epochs for the requested classification."""
    try:
        import mne
    except ImportError as exc:
        raise RuntimeError("MNE is required to read bimodal EEG epochs") from exc

    root = Path(dataset_root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    normalized_task = _normalize_task(task)
    normalized_subjects = tuple(_normalize_subject(item) for item in subjects)
    if not normalized_subjects:
        raise ValueError("subjects must contain at least one subject")
    if len(set(normalized_subjects)) != len(normalized_subjects):
        raise ValueError("subjects contains duplicates after normalization")
    if sessions != "auto":
        normalized_sessions = {
            str(item).strip().lower().removeprefix("ses-")
            for item in sessions
        }
        if normalized_sessions not in (set(), {"eeg"}):
            raise ValueError(
                "The bimodal dataset has one EEG session; use sessions: [EEG]"
            )

    derivatives = _derivatives_root(root, options)
    records: list[DS003626TaskData] = []
    word_to_label = {
        name: index for index, name in enumerate(WORD_CLASS_NAMES)
    }
    for subject in normalized_subjects:
        epochs_path = (
            derivatives
            / subject
            / f"{subject}_task-inner_epo.fif"
        )
        if not epochs_path.is_file():
            raise FileNotFoundError(
                f"Missing preprocessed epochs: {epochs_path}. Run "
                "'python preprocessing.py --config <bimodal-config>' first."
            )
        epochs = mne.read_epochs(
            epochs_path,
            preload=preload,
            verbose="ERROR",
        )
        code_to_name = {
            int(code): name.casefold()
            for name, code in epochs.event_id.items()
        }
        event_names = np.asarray(
            [code_to_name.get(int(code), "") for code in epochs.events[:, 2]]
        )
        mask = np.isin(event_names, WORD_CLASS_NAMES)
        trial_indices = np.flatnonzero(mask).astype(np.int64, copy=False)
        if trial_indices.size == 0:
            raise ValueError(f"No inner-word epochs found in {epochs_path}")
        selected_names = event_names[trial_indices]
        if normalized_task == "inner_word":
            class_names = WORD_CLASS_NAMES
            labels = np.asarray(
                [word_to_label[name] for name in selected_names],
                dtype=np.int64,
            )
        else:
            class_names = CATEGORY_CLASS_NAMES
            labels = np.asarray(
                [0 if name in SOCIAL_WORDS else 1 for name in selected_names],
                dtype=np.int64,
            )
        selected_epochs = epochs[trial_indices]
        records.append(
            DS003626TaskData(
                task=normalized_task,
                subject=subject,
                session="ses-EEG",
                epochs=selected_epochs,
                labels=labels,
                trial_indices=trial_indices,
                sample_indices=selected_epochs.events[:, 0].astype(
                    np.int64,
                    copy=True,
                ),
                class_names=class_names,
            )
        )
    return tuple(records)
