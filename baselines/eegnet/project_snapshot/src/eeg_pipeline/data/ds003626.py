"""Low-level readers for the OpenNeuro ds003626 derivatives."""

from __future__ import annotations

import _codecs
import pickle
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

import numpy as np
from numpy.core.multiarray import _reconstruct

if TYPE_CHECKING:
    from mne import Epochs


CLASS_ID_TO_NAME = {
    0: "up",
    1: "down",
    2: "right",
    3: "left",
}

CONDITION_ID_TO_NAME = {
    0: "pronounced",
    1: "inner",
    2: "visualized",
}

TASK_TO_CONDITION = {
    "speak": 0,
    "imagine": 1,
    "visual": 2,
}
DIRECTION_CLASS_NAMES = ("up", "down", "right", "left")
PATTERN_CLASS_NAMES = ("speak", "imagine", "visual")
BIOSEMI128_MONTAGE = "biosemi128"


@dataclass(frozen=True)
class ElectrodeGroupLayout:
    """Spatially compact, balanced electrode groups and their permutation."""

    original_channel_names: tuple[str, ...]
    reordered_channel_names: tuple[str, ...]
    permutation: tuple[int, ...]
    group_channel_names: tuple[tuple[str, ...], ...]
    group_sizes: tuple[int, ...]
    group_centroids: tuple[tuple[float, float, float], ...]

    @property
    def n_electrode_groups(self) -> int:
        """Return the number of spatial electrode groups."""
        return len(self.group_sizes)


def _validate_electrode_group_count(
    n_electrode_groups: int,
    n_channels: int,
) -> None:
    """Validate an electrode group count against a channel count."""
    if isinstance(n_electrode_groups, bool) or not isinstance(
        n_electrode_groups, int
    ):
        raise TypeError("n_electrode_groups must be an integer")
    if not 1 <= n_electrode_groups <= n_channels:
        raise ValueError(
            "n_electrode_groups must be between 1 and the number of "
            f"channels ({n_channels}), got {n_electrode_groups}"
        )


def _balanced_group_sizes(
    n_channels: int,
    n_electrode_groups: int,
) -> tuple[int, ...]:
    """Return near-equal group sizes matching grouped causal EEGNet."""
    base_size, remainder = divmod(n_channels, n_electrode_groups)
    return tuple(
        base_size + int(group_index < remainder)
        for group_index in range(n_electrode_groups)
    )


def _partition_spatial_indices(
    channel_indices: np.ndarray,
    normalized_xy: np.ndarray,
    target_sizes: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    """Recursively divide scalp coordinates into balanced local regions."""
    if len(target_sizes) == 1:
        if len(channel_indices) != target_sizes[0]:
            raise RuntimeError("Spatial partition produced an invalid size")
        return (tuple(int(index) for index in channel_indices),)

    group_split = len(target_sizes) // 2
    channel_split = sum(target_sizes[:group_split])
    local_xy = normalized_xy[channel_indices]
    split_axis = int(np.argmax(np.ptp(local_xy, axis=0)))
    secondary_axis = 1 - split_axis
    spatial_order = np.lexsort(
        (
            channel_indices,
            local_xy[:, secondary_axis],
            local_xy[:, split_axis],
        )
    )
    ordered_indices = channel_indices[spatial_order]

    return _partition_spatial_indices(
        ordered_indices[:channel_split],
        normalized_xy,
        target_sizes[:group_split],
    ) + _partition_spatial_indices(
        ordered_indices[channel_split:],
        normalized_xy,
        target_sizes[group_split:],
    )


def build_spatial_electrode_layout(
    channel_names: Sequence[str],
    channel_positions: Sequence[Sequence[float]] | np.ndarray,
    n_electrode_groups: int,
) -> ElectrodeGroupLayout:
    """Build balanced groups of physically nearby scalp electrodes.

    Recursive median partitioning is performed in the left-right/anterior-
    posterior plane. Group sizes intentionally use the same quotient/remainder
    rule as :class:`causal1_EEGNet`, so flattened groups align exactly with the
    model's contiguous grouped causal convolution.
    """
    names = tuple(channel_names)
    if not names:
        raise ValueError("channel_names must contain at least one channel")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("Every channel name must be a non-empty string")
    if len(set(names)) != len(names):
        raise ValueError("channel_names contains duplicate names")

    n_channels = len(names)
    _validate_electrode_group_count(n_electrode_groups, n_channels)
    positions = np.asarray(channel_positions, dtype=np.float64)
    if positions.shape != (n_channels, 3):
        raise ValueError(
            "channel_positions must have shape "
            f"({n_channels}, 3), got {positions.shape}"
        )
    if not np.isfinite(positions).all():
        raise ValueError("channel_positions contains NaN or infinite values")
    if np.any(np.linalg.norm(positions, axis=1) <= np.finfo(float).eps):
        raise ValueError("Every channel must have a non-zero 3-D position")

    target_sizes = _balanced_group_sizes(
        n_channels,
        n_electrode_groups,
    )
    if n_electrode_groups == 1:
        grouped_indices = (tuple(range(n_channels)),)
    else:
        xy = positions[:, :2]
        xy_scale = np.ptp(xy, axis=0)
        xy_scale[xy_scale <= np.finfo(float).eps] = 1.0
        normalized_xy = (xy - xy.mean(axis=0, keepdims=True)) / xy_scale
        grouped_indices = _partition_spatial_indices(
            np.arange(n_channels, dtype=np.int64),
            normalized_xy,
            target_sizes,
        )

    permutation = tuple(index for group in grouped_indices for index in group)
    if tuple(sorted(permutation)) != tuple(range(n_channels)):
        raise RuntimeError("Electrode grouping is not a complete permutation")
    if tuple(len(group) for group in grouped_indices) != target_sizes:
        raise RuntimeError("Electrode groups do not match their target sizes")

    group_channel_names = tuple(
        tuple(names[index] for index in group)
        for group in grouped_indices
    )
    group_centroids = tuple(
        tuple(float(value) for value in positions[list(group)].mean(axis=0))
        for group in grouped_indices
    )
    return ElectrodeGroupLayout(
        original_channel_names=names,
        reordered_channel_names=tuple(names[index] for index in permutation),
        permutation=permutation,
        group_channel_names=group_channel_names,
        group_sizes=target_sizes,
        group_centroids=group_centroids,
    )


def build_biosemi128_electrode_layout(
    channel_names: Sequence[str],
    n_electrode_groups: int,
) -> ElectrodeGroupLayout:
    """Build a spatial layout from MNE's standard BioSemi-128 montage."""
    try:
        import mne
    except ImportError as exc:
        raise RuntimeError(
            "MNE is required to build the BioSemi-128 electrode layout"
        ) from exc

    montage = mne.channels.make_standard_montage(BIOSEMI128_MONTAGE)
    montage_positions = montage.get_positions()["ch_pos"]
    position_by_name = {
        name.casefold(): np.asarray(position, dtype=np.float64)
        for name, position in montage_positions.items()
    }
    missing_channels = [
        name for name in channel_names if name.casefold() not in position_by_name
    ]
    if missing_channels:
        raise ValueError(
            "Channels are absent from the standard BioSemi-128 montage: "
            f"{missing_channels}"
        )
    positions = np.stack(
        [position_by_name[name.casefold()] for name in channel_names],
        axis=0,
    )
    return build_spatial_electrode_layout(
        channel_names,
        positions,
        n_electrode_groups,
    )


@dataclass(frozen=True)
class DS003626Paths:
    """Resolved derivative files for one subject and session."""

    subject: str
    session: str
    eeg_path: Path
    events_path: Path


@dataclass(frozen=True)
class DS003626Session:
    """Aligned EEG epochs and events for one subject and session."""

    subject: str
    session: str
    paths: DS003626Paths
    epochs: Epochs
    events: np.ndarray


@dataclass(frozen=True)
class DS003626TaskData:
    """EEG epochs and labels selected for one classification task."""

    task: str
    subject: str
    session: str
    epochs: Epochs
    labels: np.ndarray
    trial_indices: np.ndarray
    sample_indices: np.ndarray
    class_names: tuple[str, ...]
    channel_indices: tuple[int, ...] | None = None
    channel_names: tuple[str, ...] | None = None
    time_slice: tuple[int, int, int] | None = None
    sampling_rate_hz: float | None = None


def _normalize_subject(subject: str | int) -> str:
    """Normalize a subject identifier to the BIDS form ``sub-XX``."""

    value = str(subject).strip().lower()
    if value.startswith("sub-"):
        value = value[4:]
    if not value.isdecimal():
        raise ValueError(f"Invalid subject identifier: {subject!r}")

    index = int(value)
    if not 1 <= index <= 10:
        raise ValueError(f"Subject must be between 1 and 10, got {subject!r}")
    return f"sub-{index:02d}"


def _normalize_session(session: str | int) -> str:
    """Normalize a session identifier to the BIDS form ``ses-XX``."""

    value = str(session).strip().lower()
    if value.startswith("ses-"):
        value = value[4:]
    if not value.isdecimal():
        raise ValueError(f"Invalid session identifier: {session!r}")

    index = int(value)
    if not 1 <= index <= 3:
        raise ValueError(f"Session must be between 1 and 3, got {session!r}")
    return f"ses-{index:02d}"


def resolve_derivative_paths(
    root: str | Path, subject: str | int, session: str | int
) -> DS003626Paths:
    """Resolve and validate the EEG and events files for one session."""

    dataset_root = Path(root).expanduser()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    normalized_subject = _normalize_subject(subject)
    normalized_session = _normalize_session(session)
    session_dir = (
        dataset_root / "derivatives" / normalized_subject / normalized_session
    )
    eeg_path = (
        session_dir / f"{normalized_subject}_{normalized_session}_eeg-epo.fif"
    )
    events_path = (
        session_dir / f"{normalized_subject}_{normalized_session}_events.dat"
    )

    missing = [path for path in (eeg_path, events_path) if not path.is_file()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing ds003626 derivative file(s): {missing_text}")

    return DS003626Paths(
        subject=normalized_subject,
        session=normalized_session,
        eeg_path=eeg_path,
        events_path=events_path,
    )

_ALLOWED_PICKLE_GLOBALS = {
    ("_codecs", "encode"): _codecs.encode,
    ("numpy", "dtype"): np.dtype,
    ("numpy", "ndarray"): np.ndarray,
    ("numpy.core.multiarray", "_reconstruct"): _reconstruct,
}


class _RestrictedNumpyUnpickler(pickle.Unpickler):
    """Unpickle legacy NumPy arrays without allowing arbitrary globals."""

    def find_class(self, module: str, name: str) -> object:
        try:
            return _ALLOWED_PICKLE_GLOBALS[(module, name)]
        except KeyError as exc:
            raise pickle.UnpicklingError(
                f"Unsupported object in events file: {module}.{name}"
            ) from exc


def _load_legacy_numpy_array(stream: BinaryIO) -> object:
    """Load the Python 2 compatible pickle used by ds003626 events files."""

    return _RestrictedNumpyUnpickler(stream, encoding="latin1").load()


def read_events(events_path: str | Path) -> np.ndarray:
    """Read and validate a ds003626 four-column events array.

    The columns are raw sample index, class ID, condition ID, and session ID.
    """

    path = Path(events_path)
    if not path.is_file():
        raise FileNotFoundError(f"Events file does not exist: {path}")

    try:
        with path.open("rb") as stream:
            events = np.asarray(_load_legacy_numpy_array(stream))
    except (OSError, pickle.UnpicklingError, ValueError, TypeError) as exc:
        raise ValueError(f"Unable to read ds003626 events file: {path}") from exc

    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError(
            f"Expected events shape [trials, 4], got {events.shape} from {path}"
        )
    if not np.issubdtype(events.dtype, np.integer):
        raise ValueError(f"Expected integer events, got dtype={events.dtype} from {path}")

    events = np.asarray(events, dtype=np.int64)
    if not set(np.unique(events[:, 1])).issubset(CLASS_ID_TO_NAME):
        raise ValueError(f"Unexpected class IDs in events file: {path}")
    if not set(np.unique(events[:, 2])).issubset(CONDITION_ID_TO_NAME):
        raise ValueError(f"Unexpected condition IDs in events file: {path}")
    if not set(np.unique(events[:, 3])).issubset({1, 2, 3}):
        raise ValueError(f"Unexpected session IDs in events file: {path}")

    return events


def read_eeg_epochs(fif_path: str | Path, *, preload: bool = False) -> Epochs:
    """Read a ds003626 EEG epochs FIF while preserving its MNE metadata."""

    try:
        import mne
    except ImportError as exc:
        raise RuntimeError("MNE is required to read ds003626 EEG FIF files") from exc

    path = Path(fif_path)
    if not path.is_file():
        raise FileNotFoundError(f"EEG epochs file does not exist: {path}")

    return mne.read_epochs(path, preload=preload, verbose="ERROR")


def read_derivative_session(
    root: str | Path,
    subject: str | int,
    session: str | int,
    *,
    preload: bool = False,
) -> DS003626Session:
    """Read one derivative session and validate EEG/events alignment."""

    paths = resolve_derivative_paths(root, subject, session)
    events = read_events(paths.events_path)
    epochs = read_eeg_epochs(paths.eeg_path, preload=preload)

    if len(epochs) != events.shape[0]:
        raise RuntimeError(
            f"Trial count mismatch for {paths.subject}/{paths.session}: "
            f"EEG={len(epochs)}, events={events.shape[0]}"
        )
    if not np.array_equal(epochs.events[:, 0], events[:, 0]):
        raise RuntimeError(
            f"Sample index mismatch for {paths.subject}/{paths.session}"
        )
    if not np.array_equal(epochs.events[:, 2] - 31, events[:, 1]):
        raise RuntimeError(
            f"Class label mismatch for {paths.subject}/{paths.session}"
        )

    expected_session = int(paths.session.removeprefix("ses-"))
    if not np.all(events[:, 3] == expected_session):
        raise RuntimeError(
            f"Session label mismatch for {paths.subject}/{paths.session}"
        )

    if len(epochs.ch_names) != 128:
        raise RuntimeError(f"Expected 128 EEG channels, got {len(epochs.ch_names)}")
    if not np.isclose(epochs.info["sfreq"], 256.0):
        raise RuntimeError(
            f"Expected sampling rate 256 Hz, got {epochs.info['sfreq']} Hz"
        )
    if len(epochs.times) != 1153:
        raise RuntimeError(f"Expected 1153 time points, got {len(epochs.times)}")
    if not np.isclose(epochs.tmin, -0.5) or not np.isclose(epochs.tmax, 4.0):
        raise RuntimeError(
            f"Expected time range -0.5 to 4.0 s, got "
            f"{epochs.tmin} to {epochs.tmax} s"
        )

    return DS003626Session(
        subject=paths.subject,
        session=paths.session,
        paths=paths,
        epochs=epochs,
        events=events,
    )


def select_task_trials(
    session_data: DS003626Session,
    task: str,
) -> DS003626TaskData:
    """Select trials and labels for one of the four supported tasks.

    ``speak``, ``imagine`` and ``visual`` are four-class direction tasks.
    ``pattern`` is a three-class condition task using all trials.
    """
    normalized_task = task.strip().lower()

    if normalized_task in TASK_TO_CONDITION:
        condition_id = TASK_TO_CONDITION[normalized_task]
        mask = session_data.events[:, 2] == condition_id
        class_names = DIRECTION_CLASS_NAMES
        label_column = 1
    elif normalized_task == "pattern":
        mask = np.ones(len(session_data.events), dtype=bool)
        class_names = PATTERN_CLASS_NAMES
        label_column = 2
    else:
        allowed = (*TASK_TO_CONDITION, "pattern")
        raise ValueError(
            f"Unsupported task {task!r}; expected one of {allowed}"
        )

    trial_indices = np.flatnonzero(mask).astype(np.int64, copy=False)
    if trial_indices.size == 0:
        raise ValueError(
            f"No trials found for task {normalized_task!r} in "
            f"{session_data.subject}/{session_data.session}"
        )

    labels = session_data.events[trial_indices, label_column].astype(
        np.int64, copy=True
    )
    sample_indices = session_data.events[trial_indices, 0].astype(
        np.int64, copy=True
    )
    epochs = session_data.epochs[trial_indices]

    expected_labels = set(range(len(class_names)))
    observed_labels = set(np.unique(labels).tolist())
    if not observed_labels.issubset(expected_labels):
        raise ValueError(
            f"Task {normalized_task!r} contains invalid labels: "
            f"{sorted(observed_labels - expected_labels)}"
        )
    if not (len(epochs) == len(labels) == len(trial_indices) == len(sample_indices)):
        raise RuntimeError("Selected EEG epochs and task metadata are misaligned")

    return DS003626TaskData(
        task=normalized_task,
        subject=session_data.subject,
        session=session_data.session,
        epochs=epochs,
        labels=labels,
        trial_indices=trial_indices,
        sample_indices=sample_indices,
        class_names=class_names,
    )


def load_task_sessions(
    dataset_root: str | Path,
    task: str,
    subjects: Sequence[str | int],
    sessions: Sequence[str | int],
    *,
    preload: bool = False,
) -> tuple[DS003626TaskData, ...]:
    """Load one task from every requested subject/session combination."""
    if isinstance(subjects, (str, bytes)):
        raise TypeError("subjects must be a sequence such as [1] or [1, 2]")
    if isinstance(sessions, (str, bytes)):
        raise TypeError("sessions must be a sequence such as [1] or [1, 2]")

    normalized_subjects = tuple(_normalize_subject(value) for value in subjects)
    normalized_sessions = tuple(_normalize_session(value) for value in sessions)
    if not normalized_subjects:
        raise ValueError("subjects must contain at least one subject")
    if not normalized_sessions:
        raise ValueError("sessions must contain at least one session")
    if len(set(normalized_subjects)) != len(normalized_subjects):
        raise ValueError(
            "subjects contains duplicate identifiers after normalization: "
            f"{normalized_subjects}"
        )
    if len(set(normalized_sessions)) != len(normalized_sessions):
        raise ValueError(
            "sessions contains duplicate identifiers after normalization: "
            f"{normalized_sessions}"
        )

    normalized_task = task.strip().lower()
    allowed_tasks = (*TASK_TO_CONDITION, "pattern")
    if normalized_task not in allowed_tasks:
        raise ValueError(
            f"Unsupported task {task!r}; expected one of {allowed_tasks}"
        )

    task_sessions: list[DS003626TaskData] = []
    for subject in normalized_subjects:
        for session in normalized_sessions:
            try:
                session_data = read_derivative_session(
                    dataset_root,
                    subject=subject,
                    session=session,
                    preload=preload,
                )
                task_data = select_task_trials(session_data, normalized_task)
            except Exception as error:
                raise RuntimeError(
                    f"Failed to load task {normalized_task!r} for "
                    f"{subject}/{session}"
                ) from error
            task_sessions.append(task_data)

    return tuple(task_sessions)


if __name__ == "__main__":
    dataset_root = Path(r"F:\脑电数据集\dataset\OpenNeruro_ds003626")
    session_data = read_derivative_session(
        dataset_root, subject=1, session=1, preload=False
    )
    events = session_data.events
    epochs = session_data.epochs
    first_trial = epochs.get_data(item=slice(0, 1), copy=True)
    if not np.isfinite(first_trial).all():
        raise RuntimeError("The first EEG trial contains NaN or infinite values")
    electrode_layout = build_biosemi128_electrode_layout(
        epochs.ch_names,
        n_electrode_groups=8,
    )
    if electrode_layout.group_sizes != (16,) * 8:
        raise RuntimeError(
            f"Unexpected electrode groups: {electrode_layout.group_sizes}"
        )

    print("ds003626 read test passed")
    print(f"events: shape={events.shape}, dtype={events.dtype}")
    print(
        "EEG: "
        f"trials={len(epochs)}, channels={len(epochs.ch_names)}, "
        f"time_points={len(epochs.times)}, sfreq={epochs.info['sfreq']} Hz"
    )
    print(f"time range: {epochs.tmin:.3f} to {epochs.tmax:.3f} s")
    print(f"first trial: shape={first_trial.shape}, dtype={first_trial.dtype}")
    print(f"electrode groups: {electrode_layout.group_channel_names}")

    for task_name in ("imagine", "speak", "visual", "pattern"):
        task_data = select_task_trials(session_data, task_name)
        class_counts = dict.fromkeys(task_data.class_names, 0)
        label_ids, counts = np.unique(task_data.labels, return_counts=True)
        for label_id, count in zip(label_ids, counts, strict=True):
            class_counts[task_data.class_names[int(label_id)]] = int(count)
        print(
            f"task={task_name}: trials={len(task_data.labels)}, "
            f"classes={class_counts}"
        )

    multi_session_data = load_task_sessions(
        dataset_root,
        task="imagine",
        subjects=(1, 2),
        sessions=(1, 2),
        preload=False,
    )
    recording_keys = tuple(
        (item.subject, item.session) for item in multi_session_data
    )
    expected_keys = (
        ("sub-01", "ses-01"),
        ("sub-01", "ses-02"),
        ("sub-02", "ses-01"),
        ("sub-02", "ses-02"),
    )
    if recording_keys != expected_keys:
        raise RuntimeError(
            f"Unexpected multi-session order: {recording_keys}"
        )
    print(
        "multi-session task=imagine: "
        f"recordings={len(multi_session_data)}, "
        f"trials={sum(len(item.labels) for item in multi_session_data)}, "
        f"keys={recording_keys}"
    )
