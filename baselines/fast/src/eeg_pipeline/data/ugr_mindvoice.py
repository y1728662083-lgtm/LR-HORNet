"""Reader for official-code-derived UGR-MINDVOICE EEG epochs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from .ds003626 import DS003626TaskData


CONDITION_CLASS_NAMES = ("overt", "covert", "rest")


def _normalize_subject(subject: str | int) -> str:
    value = str(subject).strip().lower()
    if value.startswith("sub-"):
        value = value[4:]
    if not value.isdecimal() or not 1 <= int(value) <= 15:
        raise ValueError(f"UGR-MINDVOICE subject must be in 1..15: {subject!r}")
    return f"sub-{int(value):02d}"


def _normalize_session(session: str | int) -> str:
    value = str(session).strip().lower()
    if value.startswith("ses-"):
        value = value[4:]
    if not value.isdecimal() or int(value) < 1:
        raise ValueError(f"Invalid UGR-MINDVOICE session: {session!r}")
    return f"ses-{int(value):02d}"


def _derivatives_root(root: Path, options: Mapping | None) -> Path:
    configured = (options or {}).get("derivatives_root")
    if configured is None:
        return root / "derivatives" / "paper4" / "ugr_mindvoice"
    path = Path(configured).expanduser()
    return path if path.is_absolute() else root / path


def _subject_sessions(
    derivatives: Path,
    subject: str,
    sessions: Sequence[str | int] | str,
) -> tuple[str, ...]:
    if sessions != "auto":
        normalized = tuple(_normalize_session(item) for item in sessions)
        if not normalized:
            raise ValueError("sessions must not be empty")
        return normalized
    subject_dir = derivatives / subject
    discovered = tuple(
        path.name
        for path in sorted(subject_dir.glob("ses-*"))
        if path.is_dir()
    )
    if not discovered:
        raise FileNotFoundError(
            f"No preprocessed sessions found for {subject} under {subject_dir}"
        )
    return discovered


def load_task_sessions(
    dataset_root: str | Path,
    task: str,
    subjects: Sequence[str | int],
    sessions: Sequence[str | int] | str = "auto",
    *,
    preload: bool = False,
    options: Mapping | None = None,
) -> tuple[DS003626TaskData, ...]:
    """Load overt/covert/rest epochs generated through the official code."""
    try:
        import mne
    except ImportError as exc:
        raise RuntimeError("MNE is required to read UGR-MINDVOICE epochs") from exc

    normalized_task = task.strip().lower()
    if normalized_task not in {"speech_condition", "condition"}:
        raise ValueError("UGR-MINDVOICE task must be 'speech_condition'")
    normalized_task = "speech_condition"

    root = Path(dataset_root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    derivatives = _derivatives_root(root, options)
    normalized_subjects = tuple(_normalize_subject(item) for item in subjects)
    if not normalized_subjects:
        raise ValueError("subjects must contain at least one subject")
    if len(set(normalized_subjects)) != len(normalized_subjects):
        raise ValueError("subjects contains duplicates after normalization")

    records: list[DS003626TaskData] = []
    for subject in normalized_subjects:
        for session in _subject_sessions(derivatives, subject, sessions):
            session_dir = derivatives / subject / session
            for label, condition in enumerate(CONDITION_CLASS_NAMES):
                epochs_path = (
                    session_dir
                    / f"{subject}_{session}_task-{condition}_epo.fif"
                )
                if not epochs_path.is_file():
                    raise FileNotFoundError(
                        f"Missing preprocessed epochs: {epochs_path}. Run "
                        "'python preprocessing.py --config <ugr-config>' first."
                    )
                epochs = mne.read_epochs(
                    epochs_path,
                    preload=preload,
                    verbose="ERROR",
                )
                labels = np.full(len(epochs), label, dtype=np.int64)
                trial_indices = np.arange(len(epochs), dtype=np.int64)
                records.append(
                    DS003626TaskData(
                        task=normalized_task,
                        subject=subject,
                        session=f"{session}/{condition}",
                        epochs=epochs,
                        labels=labels,
                        trial_indices=trial_indices,
                        sample_indices=epochs.events[:, 0].astype(
                            np.int64,
                            copy=True,
                        ),
                        class_names=CONDITION_CLASS_NAMES,
                    )
                )
    return tuple(records)
