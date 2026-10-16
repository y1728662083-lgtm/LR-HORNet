"""Dataset registry for the EEG training pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


_ALIASES = {
    "ds003626": "ds003626",
    "openneuro_ds003626": "ds003626",
    "bimodal": "bimodal_inner_speech",
    "bimodal_inner_speech": "bimodal_inner_speech",
    "ugr": "ugr_mindvoice",
    "ugr_mindvoice": "ugr_mindvoice",
    "directional": "directional_words",
    "directional_words": "directional_words",
}


def normalize_dataset_name(name: str | None) -> str:
    """Return the canonical dataset key, preserving the old default."""
    normalized = "ds003626" if name is None else str(name).strip().casefold()
    try:
        return _ALIASES[normalized]
    except KeyError:
        raise ValueError(
            f"Unknown dataset {name!r}; available datasets are "
            "ds003626, bimodal_inner_speech, ugr_mindvoice, "
            "directional_words"
        ) from None


def load_task_sessions(
    dataset_name: str | None,
    dataset_root,
    task: str,
    subjects: Sequence,
    sessions: Sequence | str,
    *,
    preload: bool = False,
    options: Mapping | None = None,
):
    """Dispatch task loading to the selected dataset adapter."""
    canonical_name = normalize_dataset_name(dataset_name)
    if canonical_name == "ds003626":
        from .ds003626 import load_task_sessions as loader

        if sessions == "auto":
            raise ValueError("ds003626 requires an explicit sessions list")
        return loader(
            dataset_root,
            task,
            subjects=subjects,
            sessions=sessions,
            preload=preload,
        )
    if canonical_name == "bimodal_inner_speech":
        from .bimodal_inner_speech import load_task_sessions as loader
    elif canonical_name == "ugr_mindvoice":
        from .ugr_mindvoice import load_task_sessions as loader
    else:
        from .directional_words import load_task_sessions as loader
    return loader(
        dataset_root,
        task,
        subjects=subjects,
        sessions=sessions,
        preload=preload,
        options=options,
    )
