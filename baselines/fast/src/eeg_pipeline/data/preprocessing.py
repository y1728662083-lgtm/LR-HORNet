"""Dataset-aware preprocessing that invokes the authors' supplied code."""

from __future__ import annotations

import importlib.util
import logging
import re
import sys
import tempfile
import types
from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import yaml

from .registry import normalize_dataset_name


BIMODAL_PREPROCESS_MEMBER = (
    "Inner_Speech_EEG_fMRI-main/EEG_preprocessing/preprocess.py"
)
UGR_DATA_READER_MEMBER = "mind-voice-main/src/data/data_reader.py"
UGR_EPOCHS_MEMBER = "mind-voice-main/src/data/epochs.py"
UGR_CONFIG_MEMBER = "mind-voice-main/config.yaml"

BIMODAL_EVENT_ID = {
    "child": 111,
    "daughter": 112,
    "father": 113,
    "wife": 114,
    "four": 125,
    "three": 126,
    "ten": 127,
    "six": 128,
}
BIMODAL_SOCIAL_WORDS = {"child", "daughter", "father", "wife"}


class _CapturedOfficialEpochs(Exception):
    """Carry the official script's local Epochs object past its pdb hook."""

    def __init__(self, epochs) -> None:
        super().__init__("captured official epochs")
        self.epochs = epochs


def _import_pandas():
    """Import pandas without loading its optional Arrow integration.

    The project does not use Arrow. Some Windows scientific environments ship
    an incompatible optional pyarrow DLL; treating that optional package as
    unavailable keeps pandas' CSV path usable without changing EEG behavior.
    """
    marker = object()
    previous = sys.modules.get("pyarrow", marker)
    if previous is marker:
        sys.modules["pyarrow"] = None
    try:
        import pandas
    finally:
        if previous is marker:
            sys.modules.pop("pyarrow", None)
    return pandas


def _load_zip_module(
    archive_path: Path,
    member: str,
    module_name: str,
    extraction_root: Path,
):
    if not archive_path.is_file():
        raise FileNotFoundError(f"Official code archive does not exist: {archive_path}")
    with ZipFile(archive_path) as archive:
        try:
            source = archive.read(member)
        except KeyError as exc:
            raise FileNotFoundError(
                f"Official archive {archive_path} does not contain {member}"
            ) from exc
    module_path = extraction_root / Path(member).name
    module_path.write_bytes(source)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import official source: {member}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_path(value, *, base: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() and base is not None:
        path = base / path
    return path


def _normalize_numeric_subject(subject, *, maximum: int) -> int:
    value = str(subject).strip().lower().removeprefix("sub-")
    if not value.isdecimal() or not 1 <= int(value) <= maximum:
        raise ValueError(f"Invalid subject identifier: {subject!r}")
    return int(value)


def _write_bimodal_compatibility_timings(
    events_path: Path,
    output_path: Path,
) -> None:
    """Translate the released BIDS table to the official script's columns."""
    pd = _import_pandas()
    events = pd.read_csv(events_path, sep="\t")
    required = {"onset", "trial_type"}
    if not required.issubset(events.columns):
        raise ValueError(
            f"BIDS events table lacks columns {sorted(required)}: {events_path}"
        )
    words = events["trial_type"].astype(str).str.casefold()
    unknown = sorted(set(words) - set(BIMODAL_EVENT_ID))
    if unknown:
        raise ValueError(
            f"Unexpected bimodal trial_type values in {events_path}: {unknown}"
        )
    onset_ms = pd.to_numeric(events["onset"], errors="raise")
    compatibility = pd.DataFrame(
        {
            "fixation.OnsetTime": onset_ms,
            "stimulus.OnsetTime": onset_ms + 1000,
            "rest.OnsetTime": onset_ms + 3000,
            "imgTrigger": words.map(BIMODAL_EVENT_ID),
            "InnerWord": words,
            "Condition": words.map(
                lambda word: (
                    "social" if word in BIMODAL_SOCIAL_WORDS else "number"
                )
            ),
        }
    )
    with output_path.open("w", encoding="utf-16", newline="") as stream:
        stream.write("Generated from the released BIDS events table\n")
        compatibility.to_csv(stream, sep="\t", index=False)


def _bimodal_output_path(root: Path, output_root: Path, subject: int) -> Path:
    subject_name = f"sub-{subject:02d}"
    return output_root / subject_name / f"{subject_name}_task-inner_epo.fif"


def _preprocess_bimodal_subject(
    root: Path,
    output_root: Path,
    official_zip: Path,
    subject: int,
    *,
    overwrite: bool,
    verbose: bool,
) -> Path:
    subject_name = f"sub-{subject:02d}"
    eeg_dir = root / subject_name / "ses-EEG" / "eeg"
    bdf_path = eeg_dir / f"{subject_name}_ses-EEG_task-inner_eeg.bdf"
    events_path = eeg_dir / f"{subject_name}_ses-EEG_task-inner_events.tsv"
    for path in (bdf_path, events_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing bimodal source file: {path}")

    output_path = _bimodal_output_path(root, output_root, subject)
    if output_path.is_file() and not overwrite:
        return output_path

    with tempfile.TemporaryDirectory(prefix="paper4_bimodal_") as temp_name:
        temp_root = Path(temp_name)
        timings_path = temp_root / "official_timings_export.txt"
        _write_bimodal_compatibility_timings(events_path, timings_path)
        _import_pandas()
        module = _load_zip_module(
            official_zip,
            BIMODAL_PREPROCESS_MEMBER,
            f"_paper4_bimodal_official_{subject}",
            temp_root,
        )

        def capture_epochs() -> None:
            frame = sys._getframe(1)
            epochs = frame.f_locals.get("epochs")
            if epochs is None:
                raise RuntimeError(
                    "The official preprocessing script reached its debug hook "
                    "without an Epochs object"
                )
            raise _CapturedOfficialEpochs(epochs)

        module.db = capture_epochs
        module.paths = {
            "data": str(bdf_path),
            "timings": str(timings_path),
        }
        official_args = {
            "id": subject,
            "verbose": verbose,
            "plot": False,
        }
        try:
            module.Preprocess(official_args, module.paths).clean_data()
        except _CapturedOfficialEpochs as captured:
            epochs = captured.epochs
        else:
            raise RuntimeError(
                "The official bimodal script returned before producing epochs"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        epochs.save(output_path, overwrite=overwrite, verbose="ERROR")
    return output_path


def _deep_update(target: dict, update: dict) -> dict:
    for key, value in update.items():
        if (
            isinstance(value, dict)
            and isinstance(target.get(key), dict)
        ):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def _load_ugr_official_config(archive_path: Path) -> dict:
    with ZipFile(archive_path) as archive:
        try:
            text = archive.read(UGR_CONFIG_MEMBER).decode("utf-8")
        except KeyError as exc:
            raise FileNotFoundError(
                f"Official archive lacks {UGR_CONFIG_MEMBER}"
            ) from exc
    config = yaml.safe_load(text)
    if not isinstance(config, dict):
        raise ValueError("Official UGR config root is not a mapping")
    return config


def _load_ugr_epochs_module(archive_path: Path, temp_root: Path):
    previous_src = sys.modules.get("src")
    previous_utils = sys.modules.get("src.utils")
    src_module = types.ModuleType("src")
    src_module.__path__ = []
    utils_module = types.ModuleType("src.utils")

    def log_info(logger, text) -> None:
        logger.info("%s", text)

    utils_module.log_info = log_info
    sys.modules["src"] = src_module
    sys.modules["src.utils"] = utils_module
    try:
        return _load_zip_module(
            archive_path,
            UGR_EPOCHS_MEMBER,
            "_paper4_ugr_official_epochs",
            temp_root,
        )
    finally:
        if previous_src is None:
            sys.modules.pop("src", None)
        else:
            sys.modules["src"] = previous_src
        if previous_utils is None:
            sys.modules.pop("src.utils", None)
        else:
            sys.modules["src.utils"] = previous_utils


def _ugr_bids_root(root: Path) -> Path:
    data_subdir = root / "Data"
    if (data_subdir / "dataset_description.json").is_file():
        return data_subdir
    return root


def _discover_ugr_recordings(
    bids_root: Path,
    subjects,
    sessions,
) -> tuple[tuple[int, int], ...]:
    requested_subjects = {
        _normalize_numeric_subject(item, maximum=15) for item in subjects
    }
    explicit_sessions = None
    if sessions != "auto":
        explicit_sessions = {
            _normalize_numeric_subject(item, maximum=99) for item in sessions
        }
    pattern = re.compile(r"sub-(\d+)_ses-(\d+)_task-VCV_run-01_eeg\.edf$")
    recordings: list[tuple[int, int]] = []
    for path in sorted(bids_root.glob("sub-*/ses-*/eeg/*_eeg.edf")):
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        subject, session = map(int, match.groups())
        if subject not in requested_subjects:
            continue
        if explicit_sessions is not None and session not in explicit_sessions:
            continue
        recordings.append((subject, session))
    missing_subjects = requested_subjects - {subject for subject, _ in recordings}
    if missing_subjects:
        raise FileNotFoundError(
            f"No UGR BIDS recordings found for subjects {sorted(missing_subjects)}"
        )
    return tuple(recordings)


def _ugr_output_dir(output_root: Path, subject: int, session: int) -> Path:
    return output_root / f"sub-{subject:02d}" / f"ses-{session:02d}"


def _preprocess_ugr_recording(
    bids_root: Path,
    output_root: Path,
    official_zip: Path,
    subject: int,
    session: int,
    *,
    overwrite: bool,
    config_overrides: dict,
) -> tuple[Path, ...]:
    try:
        import mne
    except ImportError as exc:
        raise RuntimeError("MNE is required for UGR preprocessing") from exc

    output_dir = _ugr_output_dir(output_root, subject, session)
    subject_name = f"sub-{subject:02d}"
    session_name = f"ses-{session:02d}"
    raw_path = (
        output_dir
        / f"{subject_name}_{session_name}_task-VCV_desc-preproc_raw.fif"
    )
    epoch_paths = tuple(
        output_dir
        / f"{subject_name}_{session_name}_task-{condition}_epo.fif"
        for condition in ("overt", "covert", "rest")
    )
    if (
        raw_path.is_file()
        and all(path.is_file() for path in epoch_paths)
        and not overwrite
    ):
        return (raw_path, *epoch_paths)

    logger = logging.getLogger(
        f"paper4.ugr.{subject_name}.{session_name}"
    )
    with tempfile.TemporaryDirectory(prefix="paper4_ugr_") as temp_name:
        temp_root = Path(temp_name)
        _import_pandas()
        reader_module = _load_zip_module(
            official_zip,
            UGR_DATA_READER_MEMBER,
            f"_paper4_ugr_official_reader_{subject}_{session}",
            temp_root,
        )
        official_config = _load_ugr_official_config(official_zip)
        official_config["dataset"]["BIDS_DIR"] = str(bids_root)
        _deep_update(official_config, deepcopy(config_overrides))

        if raw_path.is_file() and not overwrite:
            raw = mne.io.read_raw_fif(
                raw_path,
                preload=True,
                verbose="ERROR",
            )
        else:
            reader = reader_module.DataReader(
                config=official_config,
                logger=logger,
                subject=f"{subject:02d}",
                session=f"{session:02d}",
            )
            reader.preprocess_data()
            raw = reader.preprocessed_data
            output_dir.mkdir(parents=True, exist_ok=True)
            raw.save(raw_path, overwrite=overwrite, verbose="ERROR")

        epochs_module = _load_ugr_epochs_module(official_zip, temp_root)
        definitions = (
            (
                "overt",
                ("RealWordsExperiment",),
                "StartStimulus",
            ),
            (
                "covert",
                ("SilentWordsExperiment",),
                "StartStimulus",
            ),
            (
                "rest",
                ("SilentWordsExperiment", "RealWordsExperiment"),
                "StartFixation",
            ),
        )
        tmin = float(official_config["decoding"]["tmin"])
        tmax = float(official_config["decoding"]["tmax"])
        for output_path, (condition, tasks, event) in zip(
            epoch_paths,
            definitions,
            strict=True,
        ):
            if output_path.is_file() and not overwrite:
                continue
            epochs = epochs_module.EpochsData(
                config=official_config,
                logger=logger,
                raw=raw,
                tasks=tasks,
                event=event,
                modalities=("Audio", "Text", "Pictures"),
                tmin=tmin,
                tmax=tmax,
                baseline=(tmin, 0),
            ).create_epochs()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            epochs.save(output_path, overwrite=overwrite, verbose="ERROR")
    return (raw_path, *epoch_paths)


def _directional_paths(root: Path, subjects) -> tuple[Path, ...]:
    pattern = re.compile(r"^(russian|spanish)[/\\](sub\d+)$", re.I)
    paths = []
    for subject_spec in subjects:
        match = pattern.fullmatch(str(subject_spec).strip())
        if match is None:
            raise ValueError(
                "Directional subject must look like 'spanish/sub0' or "
                f"'russian/sub1', got {subject_spec!r}"
            )
        language, subject = (part.casefold() for part in match.groups())
        path = (
            root
            / "preprocessed"
            / language
            / subject
            / f"{subject}_epochs.fif"
        )
        if not path.is_file():
            raise FileNotFoundError(f"Missing published preprocessed FIF: {path}")
        paths.append(path)
    return tuple(paths)


def preprocess_from_config(config: dict, *, dry_run: bool = False) -> tuple[Path, ...]:
    """Preprocess or validate the dataset selected by a training config."""
    data_config = config["data"]
    preprocessing_config = config.get("preprocessing", {})
    dataset_name = normalize_dataset_name(data_config.get("dataset"))
    root = Path(data_config["root"]).expanduser()
    subjects = data_config["subjects"]
    sessions = data_config.get("sessions", "auto")
    overwrite = bool(preprocessing_config.get("overwrite", False))

    if dataset_name == "ds003626":
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {root}")
        return ()

    if dataset_name == "directional_words":
        paths = _directional_paths(root, subjects)
        if not dry_run:
            import mne

            for path in paths:
                mne.read_epochs(path, preload=False, verbose="ERROR")
        return paths

    official_zip = _resolve_path(
        preprocessing_config["official_code_zip"]
    )
    if not official_zip.is_file():
        raise FileNotFoundError(
            f"Official preprocessing archive does not exist: {official_zip}"
        )

    configured_output = preprocessing_config.get("output_root")
    if configured_output is None:
        if dataset_name == "bimodal_inner_speech":
            output_root = root / "derivatives" / "paper4" / "bimodal"
        else:
            output_root = root / "derivatives" / "paper4" / "ugr_mindvoice"
    else:
        output_root = _resolve_path(configured_output, base=root)

    if dataset_name == "bimodal_inner_speech":
        normalized_subjects = tuple(
            _normalize_numeric_subject(item, maximum=5)
            for item in subjects
        )
        invalid = sorted(set(normalized_subjects) - {1, 2, 3, 5})
        if invalid:
            raise ValueError(f"Unavailable bimodal subjects: {invalid}")
        planned = tuple(
            _bimodal_output_path(root, output_root, subject)
            for subject in normalized_subjects
        )
        if dry_run:
            for subject in normalized_subjects:
                subject_name = f"sub-{subject:02d}"
                source = (
                    root
                    / subject_name
                    / "ses-EEG"
                    / "eeg"
                    / f"{subject_name}_ses-EEG_task-inner_eeg.bdf"
                )
                if not source.is_file():
                    raise FileNotFoundError(
                        f"Missing bimodal source file: {source}"
                    )
            return planned
        return tuple(
            _preprocess_bimodal_subject(
                root,
                output_root,
                official_zip,
                subject,
                overwrite=overwrite,
                verbose=bool(preprocessing_config.get("verbose", False)),
            )
            for subject in normalized_subjects
        )

    bids_root = _ugr_bids_root(root)
    recordings = _discover_ugr_recordings(
        bids_root,
        subjects,
        sessions,
    )
    excluded = {
        str(item).casefold()
        for item in preprocessing_config.get("exclude_recordings", ())
    }
    recordings = tuple(
        item
        for item in recordings
        if f"{item[0]:02d}/{item[1]:02d}" not in excluded
        and f"{item[0]}/{item[1]}" not in excluded
    )
    planned = tuple(
        path
        for subject, session in recordings
        for path in (
            _ugr_output_dir(output_root, subject, session)
            / (
                f"sub-{subject:02d}_ses-{session:02d}_task-VCV_"
                "desc-preproc_raw.fif"
            ),
            *(
                _ugr_output_dir(output_root, subject, session)
                / (
                    f"sub-{subject:02d}_ses-{session:02d}_"
                    f"task-{condition}_epo.fif"
                )
                for condition in ("overt", "covert", "rest")
            ),
        )
    )
    if dry_run:
        return planned
    outputs: list[Path] = []
    overrides = preprocessing_config.get("official_config_overrides", {})
    for subject, session in recordings:
        outputs.extend(
            _preprocess_ugr_recording(
                bids_root,
                output_root,
                official_zip,
                subject,
                session,
                overwrite=overwrite,
                config_overrides=overrides,
            )
        )
    return tuple(outputs)
