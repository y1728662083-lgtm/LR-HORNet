"""Reader for the published preprocessed directional-word FIF files."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np

from .ds003626 import DS003626TaskData


COMMON_WORDS = ("back", "down", "forward", "left", "right", "up")
RUSSIAN_WORDS = (*COMMON_WORDS, "next")
_SUBJECT_PATTERN = re.compile(r"^(russian|spanish)[/\\](sub\d+)$", re.I)
_WORD_EVENT_PATTERN = re.compile(
    r"^(BACK|DOWN|FORWARD|LEFT|NEXT|RIGHT|UP)([12])$",
    re.I,
)
_LANGUAGE_SUBJECTS = {
    "russian": tuple(f"sub{index}" for index in range(1, 13)),
    "spanish": tuple(f"sub{index}" for index in range(10)),
}


class _LazyDirectionalView:
    """Expose one fixed channel/time grid without rewriting the source FIF."""

    def __init__(
        self,
        epochs,
        actual_order: Sequence[str],
        canonical_order: Sequence[str],
        *,
        target_sampling_rate_hz: float,
        time_window_seconds: tuple[float, float],
    ) -> None:
        import mne

        self._epochs = epochs
        self._actual_order = tuple(actual_order)
        self._source_sampling_rate_hz = float(epochs.info["sfreq"])
        self._target_sampling_rate_hz = float(target_sampling_rate_hz)
        self._time_window_seconds = tuple(map(float, time_window_seconds))
        window_start, window_stop = self._time_window_seconds
        if not np.isfinite((window_start, window_stop)).all():
            raise ValueError("time_window_seconds must contain finite values")
        if window_stop <= window_start:
            raise ValueError("time_window_seconds must satisfy start < stop")
        if self._target_sampling_rate_hz <= 0:
            raise ValueError("target_sampling_rate_hz must be positive")

        source_times = np.asarray(epochs.times, dtype=np.float64)
        tolerance = np.finfo(np.float64).eps * 32
        crop_indices = np.flatnonzero(
            (source_times >= window_start - tolerance)
            & (source_times < window_stop - tolerance)
        )
        expected_source_times = int(
            round((window_stop - window_start) * self._source_sampling_rate_hz)
        )
        if len(crop_indices) != expected_source_times:
            raise ValueError(
                "Published epoch does not cover the configured half-open window "
                f"[{window_start}, {window_stop}) at "
                f"{self._source_sampling_rate_hz:g} Hz: expected "
                f"{expected_source_times} samples, found {len(crop_indices)}"
            )
        if len(crop_indices) == 0 or not np.array_equal(
            crop_indices,
            np.arange(crop_indices[0], crop_indices[0] + len(crop_indices)),
        ):
            raise ValueError("Configured time window is empty or non-contiguous")
        self._crop_indices = crop_indices

        rate_ratio = Fraction(
            str(self._target_sampling_rate_hz)
        ) / Fraction(str(self._source_sampling_rate_hz))
        rate_ratio = rate_ratio.limit_denominator(10_000)
        self._resample_up = int(rate_ratio.numerator)
        self._resample_down = int(rate_ratio.denominator)
        self._output_times = int(
            round((window_stop - window_start) * self._target_sampling_rate_hz)
        )

        selection = [epochs.ch_names.index(name) for name in actual_order]
        self.info = mne.pick_info(epochs.info, selection, copy=True)
        rename = {
            actual: canonical
            for actual, canonical in zip(actual_order, canonical_order)
            if actual != canonical
        }
        if rename:
            mne.rename_channels(self.info, rename)
        with self.info._unlock(check_after=True):
            self.info["sfreq"] = self._target_sampling_rate_hz
            if (
                self.info["lowpass"] is not None
                and self.info["lowpass"] > self._target_sampling_rate_hz / 2
            ):
                self.info["lowpass"] = self._target_sampling_rate_hz / 2
        self.times = window_start + (
            np.arange(self._output_times, dtype=np.float64)
            / self._target_sampling_rate_hz
        )
        self.events = epochs.events

    @property
    def ch_names(self) -> list[str]:
        return self.info["ch_names"]

    def __len__(self) -> int:
        return len(self._epochs)

    def get_data(self, *, item=None, copy=True, verbose=None):
        data = self._epochs.get_data(
            item=item,
            picks=list(self._actual_order),
            copy=copy,
            verbose=verbose,
        )
        data = data[..., self._crop_indices]
        if (self._resample_up, self._resample_down) != (1, 1):
            from scipy.signal import resample_poly

            data = resample_poly(
                data,
                up=self._resample_up,
                down=self._resample_down,
                axis=-1,
                padtype="line",
            )
        if data.shape[-1] != self._output_times:
            raise RuntimeError(
                f"Directional-word adapter produced {data.shape[-1]} samples; "
                f"expected {self._output_times}"
            )
        return data


def _eeg_channel_names(epochs) -> tuple[str, ...]:
    import mne

    picks = mne.pick_types(
        epochs.info,
        meg=False,
        eeg=True,
        eog=False,
        ecg=False,
        emg=False,
        stim=False,
        exclude=[],
    )
    if len(picks) == 0:
        raise RuntimeError("No channels marked as EEG were found in the FIF file")
    return tuple(epochs.ch_names[index] for index in picks)


@lru_cache(maxsize=8)
def _common_language_channels(root_text: str, language: str) -> tuple[str, ...]:
    """Return the case-insensitive EEG intersection in first-subject order."""
    import mne

    root = Path(root_text)
    channel_lists: list[tuple[str, ...]] = []
    for subject in _LANGUAGE_SUBJECTS[language]:
        path = root / "preprocessed" / language / subject / f"{subject}_epochs.fif"
        if not path.is_file():
            raise FileNotFoundError(
                f"Cannot harmonize {language} channels; missing published FIF: {path}"
            )
        epochs = mne.read_epochs(path, preload=False, verbose="ERROR")
        channel_lists.append(_eeg_channel_names(epochs))
    common_folded = set.intersection(
        *({name.casefold() for name in names} for names in channel_lists)
    )
    common = tuple(
        name for name in channel_lists[0] if name.casefold() in common_folded
    )
    return common


def _harmonize_directional_epochs(
    epochs,
    common_channels: Sequence[str],
    *,
    target_sampling_rate_hz: float,
    time_window_seconds: tuple[float, float],
):
    actual = _eeg_channel_names(epochs)
    by_folded: dict[str, str] = {}
    for name in actual:
        key = name.casefold()
        if key in by_folded:
            raise RuntimeError(f"Case-insensitive duplicate EEG channel: {name}")
        by_folded[key] = name
    missing = [
        name for name in common_channels if name.casefold() not in by_folded
    ]
    if missing:
        raise RuntimeError("Missing common EEG channels: " + ", ".join(missing))
    actual_order = [by_folded[name.casefold()] for name in common_channels]
    return _LazyDirectionalView(
        epochs,
        actual_order,
        common_channels,
        target_sampling_rate_hz=target_sampling_rate_hz,
        time_window_seconds=time_window_seconds,
    )


def _parse_harmonization_options(
    options: Mapping | None,
    *,
    source_sampling_rate_hz: float,
    source_tmin: float,
    source_tmax: float,
) -> tuple[bool, float, tuple[float, float], int | None]:
    configured = options or {}
    harmonize = bool(configured.get("harmonize_within_language", False))
    target_rate = float(
        configured.get("target_sampling_rate_hz", source_sampling_rate_hz)
    )
    raw_window = configured.get("time_window_seconds")
    if raw_window is None:
        time_window = (
            float(source_tmin),
            float(source_tmax + 1.0 / source_sampling_rate_hz),
        )
    else:
        if (
            isinstance(raw_window, (str, bytes))
            or not isinstance(raw_window, Sequence)
            or len(raw_window) != 2
        ):
            raise ValueError("time_window_seconds must be [start, stop]")
        time_window = (float(raw_window[0]), float(raw_window[1]))
    expected_channels = configured.get("expected_common_channels")
    if expected_channels is not None:
        expected_channels = int(expected_channels)
        if expected_channels <= 0:
            raise ValueError("expected_common_channels must be positive")
    return harmonize, target_rate, time_window, expected_channels


def _normalize_subject(subject: str | int) -> tuple[str, str]:
    match = _SUBJECT_PATTERN.fullmatch(str(subject).strip())
    if match is None:
        raise ValueError(
            "Directional-word subjects must include language, for example "
            "'spanish/sub0' or 'russian/sub12'"
        )
    return match.group(1).casefold(), match.group(2).casefold()


def _normalize_task(task: str) -> str:
    aliases = {
        "covert_words": "covert_words",
        "covert": "covert_words",
        "overt_words": "overt_words",
        "overt": "overt_words",
        "speech_mode": "speech_mode",
        "mode": "speech_mode",
    }
    try:
        return aliases[task.strip().lower()]
    except KeyError:
        raise ValueError(
            "Directional-word task must be 'covert_words', 'overt_words', "
            "or 'speech_mode'"
        ) from None


def load_task_sessions(
    dataset_root: str | Path,
    task: str,
    subjects: Sequence[str | int],
    sessions: Sequence[str | int] | str = "auto",
    *,
    preload: bool = False,
    options: Mapping | None = None,
) -> tuple[DS003626TaskData, ...]:
    """Read the supplied ICA-cleaned FIF files without reprocessing them."""
    try:
        import mne
    except ImportError as exc:
        raise RuntimeError(
            "MNE is required to read directional-word epochs"
        ) from exc

    root = Path(dataset_root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    normalized_task = _normalize_task(task)
    include_next = bool((options or {}).get("include_next", False))
    word_classes = RUSSIAN_WORDS if include_next else COMMON_WORDS
    word_to_label = {
        name: index for index, name in enumerate(word_classes)
    }

    normalized_subjects = tuple(_normalize_subject(item) for item in subjects)
    if not normalized_subjects:
        raise ValueError("subjects must contain at least one subject")
    if len(set(normalized_subjects)) != len(normalized_subjects):
        raise ValueError("subjects contains duplicates after normalization")

    records: list[DS003626TaskData] = []
    for language, subject in normalized_subjects:
        epochs_path = (
            root
            / "preprocessed"
            / language
            / subject
            / f"{subject}_epochs.fif"
        )
        if not epochs_path.is_file():
            raise FileNotFoundError(
                f"Missing published preprocessed FIF: {epochs_path}"
            )
        epochs = mne.read_epochs(
            epochs_path,
            preload=preload,
            verbose="ERROR",
        )
        (
            harmonize,
            target_sampling_rate_hz,
            time_window_seconds,
            expected_common_channels,
        ) = _parse_harmonization_options(
            options,
            source_sampling_rate_hz=float(epochs.info["sfreq"]),
            source_tmin=float(epochs.times[0]),
            source_tmax=float(epochs.times[-1]),
        )
        common_channels = (
            _common_language_channels(str(root.resolve()), language)
            if harmonize
            else None
        )
        if (
            common_channels is not None
            and expected_common_channels is not None
            and len(common_channels) != expected_common_channels
        ):
            raise RuntimeError(
                f"{language} published FIF intersection has "
                f"{len(common_channels)} channels; configured and audited "
                f"count is {expected_common_channels}"
            )
        code_to_name = {
            int(code): name.upper() for name, code in epochs.event_id.items()
        }
        selected_indices: list[int] = []
        selected_words: list[str] = []
        selected_modes: list[int] = []
        for index, code in enumerate(epochs.events[:, 2]):
            match = _WORD_EVENT_PATTERN.fullmatch(
                code_to_name.get(int(code), "")
            )
            if match is None:
                continue
            word = match.group(1).casefold()
            mode = int(match.group(2)) - 1
            if word == "next" and not include_next:
                continue
            if normalized_task == "covert_words" and mode != 1:
                continue
            if normalized_task == "overt_words" and mode != 0:
                continue
            selected_indices.append(index)
            selected_words.append(word)
            selected_modes.append(mode)
        if not selected_indices:
            raise ValueError(
                f"No events for task {normalized_task!r} in {epochs_path}"
            )

        trial_indices = np.asarray(selected_indices, dtype=np.int64)
        selected_epochs = epochs[trial_indices]
        if common_channels is not None:
            selected_epochs = _harmonize_directional_epochs(
                selected_epochs,
                common_channels,
                target_sampling_rate_hz=target_sampling_rate_hz,
                time_window_seconds=time_window_seconds,
            )
        if normalized_task == "speech_mode":
            class_names = ("overt", "covert")
            labels = np.asarray(selected_modes, dtype=np.int64)
        else:
            class_names = word_classes
            labels = np.asarray(
                [word_to_label[word] for word in selected_words],
                dtype=np.int64,
            )
        records.append(
            DS003626TaskData(
                task=normalized_task,
                subject=f"{language}/{subject}",
                session="epochs",
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
