"""Reader for the published preprocessed directional-word FIF files."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np

from .ds003626 import DS003626TaskData


COMMON_WORDS = ("back", "down", "forward", "left", "right", "up")
RUSSIAN_WORDS = (*COMMON_WORDS, "next")
HARMONIZATION_PROFILES = {
    "russian": (250.0, 0.0, 0.996),
    "spanish": (500.0, -0.5, 0.998),
}
_LANGUAGE_SUBJECTS = {
    "russian": tuple(f"sub{index}" for index in range(1, 13)),
    "spanish": tuple(f"sub{index}" for index in range(10)),
}
_SUBJECT_PATTERN = re.compile(r"^(russian|spanish)[/\\](sub\d+)$", re.I)
_WORD_EVENT_PATTERN = re.compile(
    r"^(BACK|DOWN|FORWARD|LEFT|NEXT|RIGHT|UP)([12])$",
    re.I,
)


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
    names = tuple(epochs.ch_names[index] for index in picks)
    folded = tuple(name.casefold() for name in names)
    if len(set(folded)) != len(folded):
        raise RuntimeError("Case-insensitive duplicate EEG channel names were found")
    return names


@lru_cache(maxsize=8)
def common_language_channels(root_text: str, language: str) -> tuple[str, ...]:
    """Return the EEG-channel intersection for one language.

    Russian and Spanish are resolved independently. Ordering follows the first
    released subject for that language and is therefore identical in every fold.
    """
    import mne

    normalized_language = str(language).strip().casefold()
    if normalized_language not in _LANGUAGE_SUBJECTS:
        raise ValueError(f"Unsupported directional-word language: {language!r}")
    root = Path(root_text).expanduser()
    channel_lists: list[tuple[str, ...]] = []
    for subject in _LANGUAGE_SUBJECTS[normalized_language]:
        path = (
            root
            / "preprocessed"
            / normalized_language
            / subject
            / f"{subject}_epochs.fif"
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"Cannot resolve {normalized_language} channel intersection; "
                f"missing published FIF: {path}"
            )
        epochs = mne.read_epochs(path, preload=False, verbose="ERROR")
        channel_lists.append(_eeg_channel_names(epochs))

    common_folded = set.intersection(
        *({name.casefold() for name in names} for names in channel_lists)
    )
    common = tuple(
        name for name in channel_lists[0] if name.casefold() in common_folded
    )
    if not common:
        raise RuntimeError(
            f"No common EEG channels remain across {normalized_language} subjects"
        )
    return common


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
    harmonize_within_language = bool(
        (options or {}).get(
            "harmonize_within_language",
            (options or {}).get("harmonize_languages", True),
        )
    )
    word_classes = RUSSIAN_WORDS if include_next else COMMON_WORDS
    word_to_label = {
        name: index for index, name in enumerate(word_classes)
    }

    normalized_subjects = tuple(_normalize_subject(item) for item in subjects)
    if not normalized_subjects:
        raise ValueError("subjects must contain at least one subject")
    if len(set(normalized_subjects)) != len(normalized_subjects):
        raise ValueError("subjects contains duplicates after normalization")
    selected_languages = {language for language, _ in normalized_subjects}
    if harmonize_within_language and len(selected_languages) != 1:
        raise ValueError(
            "Russian and Spanish recordings are harmonized independently. "
            "Run one language per configuration."
        )

    language_channels = (
        {
            language: common_language_channels(str(root.resolve()), language)
            for language in selected_languages
        }
        if harmonize_within_language
        else {}
    )

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
        if harmonize_within_language:
            harmonized_channels = language_channels[language]
            (
                harmonized_sfreq,
                harmonized_tmin,
                harmonized_tmax,
            ) = HARMONIZATION_PROFILES[language]
            channel_lookup = {
                name.casefold(): index
                for index, name in enumerate(epochs.ch_names)
            }
            missing = [
                name for name in harmonized_channels
                if name.casefold() not in channel_lookup
            ]
            if missing:
                raise ValueError(
                    f"Cannot harmonize channels in {epochs_path}; missing {missing}"
                )
            channel_indices = tuple(
                channel_lookup[name.casefold()] for name in harmonized_channels
            )
            source_sfreq = float(epochs.info["sfreq"])
            ratio = source_sfreq / harmonized_sfreq
            step = int(round(ratio))
            if step < 1 or not np.isclose(ratio, step):
                raise ValueError(
                    f"Cannot harmonize sampling rate {source_sfreq} Hz "
                    f"in {epochs_path}"
                )
            start = int(
                np.searchsorted(epochs.times, harmonized_tmin, side="left")
            )
            stop = int(
                np.searchsorted(
                    epochs.times,
                    harmonized_tmax + np.finfo(float).eps * 16,
                    side="right",
                )
            )
            effective_times = epochs.times[start:stop:step]
            expected_count = int(
                round((harmonized_tmax - harmonized_tmin) * harmonized_sfreq)
            ) + 1
            expected_times = (
                harmonized_tmin
                + np.arange(expected_count) / harmonized_sfreq
            )
            if (
                len(effective_times) != len(expected_times)
                or not np.allclose(effective_times, expected_times)
            ):
                raise ValueError(
                    f"Unexpected directional time axis in {epochs_path}: "
                    f"{effective_times[0]}..{effective_times[-1]} "
                    f"({len(effective_times)} samples)"
                )
            channel_names = harmonized_channels
            time_slice = (start, stop, step)
            sampling_rate_hz = harmonized_sfreq
        else:
            channel_indices = None
            channel_names = None
            time_slice = None
            sampling_rate_hz = None
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
                channel_indices=channel_indices,
                channel_names=channel_names,
                time_slice=time_slice,
                sampling_rate_hz=sampling_rate_hz,
            )
        )
    return tuple(records)
