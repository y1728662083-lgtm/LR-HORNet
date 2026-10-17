"""Direct official Braindecode trial-classification baselines.

The catalog is synchronized with Braindecode's own
``models_mandatory_parameters`` registry.  Each builder returns the public
Braindecode class itself: no wrapper, pooling rule, output adapter, or
architecture hyperparameter is added.  Only metadata describing this dataset
is passed to the official constructor.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import traceback
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from portable_paths import resolve_config_data_root


MODEL_KEY_PREFIX: Final = "braindecode_"


@dataclass(frozen=True)
class BaselineSpec:
    """One official classification architecture in Braindecode 1.8.1."""

    architecture: str
    paper_year: int | None = None
    runnable: bool = True
    reason: str = ""

    @property
    def model_key(self) -> str:
        return f"{MODEL_KEY_PREFIX}{self.architecture.lower()}"


# This is the classification subset of Braindecode 1.8.1's authoritative
# ``models_mandatory_parameters`` list after removing its official
# ``non_classification_models`` entries.  Keeping the expected names here makes
# an upgrade mismatch fail loudly instead of silently omitting a new model.
_CLASSIFIER_NAMES: Final = (
    "ATCNet",
    "BDTCN",
    "Deep4Net",
    "DeepSleepNet",
    "EEGConformer",
    "EEGInceptionERP",
    "EEGInceptionMI",
    "EEGITNet",
    "EEGNet",
    "EEGPT",
    "InterpolatedEEGPT",
    "ShallowFBCSPNet",
    "SleepStagerBlanco2020",
    "SleepStagerChambon2018",
    "AttnSleep",
    "TIDNet",
    "USleep",
    "BIOT",
    "InterpolatedBIOT",
    "AttentionBaseNet",
    "Labram",
    "InterpolatedLaBraM",
    "EEGSimpleConv",
    "SPARCNet",
    "ContraWR",
    "EEGNeX",
    "EEGSym",
    "TSception",
    "EEGTCNet",
    "SyncNet",
    "MSVTNet",
    "EEGMiner",
    "CTNet",
    "TCFormer",
    "SincShallowNet",
    "SCCNet",
    "SignalJEPA_Contextual",
    "SignalJEPA_PostLocal",
    "SignalJEPA_PreLocal",
    "FBCNet",
    "FBMSNet",
    "FBLightConvNet",
    "IFNet",
    "PBT",
    "SSTDPN",
    "BrainModule",
    "BENDR",
    "InterpolatedBENDR",
    "LUNA",
    "MEDFormer",
    "STEEGFormer",
    "MVPFormer",
    "REVE",
    "CBraMod",
    "CodeBrain",
    "DGCNN",
    "EEGDINO",
    "ZUNA",
)

_PAPER_YEARS: Final = {
    "CBraMod": 2025,
    "CodeBrain": 2025,
    "EEGDINO": 2025,
    "LUNA": 2025,
    "PBT": 2025,
    "REVE": 2025,
    "SSTDPN": 2025,
    "TCFormer": 2025,
    "MVPFormer": 2026,
    "STEEGFormer": 2026,
    "ZUNA": 2026,
}

BASELINE_SPECS: Final = tuple(
    BaselineSpec(name, paper_year=_PAPER_YEARS.get(name))
    for name in _CLASSIFIER_NAMES
)
_SPEC_BY_ARCHITECTURE: Final = {
    spec.architecture.casefold(): spec for spec in BASELINE_SPECS
}
_SPEC_BY_MODEL_KEY: Final = {
    spec.model_key.casefold(): spec for spec in BASELINE_SPECS
}


def _official_classifier_registry() -> dict[str, tuple[str, ...]]:
    """Return Braindecode's official classification constructor registry."""
    from braindecode.models.util import (
        models_mandatory_parameters,
        non_classification_models,
    )

    excluded = set(non_classification_models)
    return {
        model_name: tuple(required_parameters)
        for model_name, required_parameters, _ in models_mandatory_parameters
        if model_name not in excluded
    }


def _validate_catalog_against_braindecode() -> dict[str, tuple[str, ...]]:
    registry = _official_classifier_registry()
    expected = set(_CLASSIFIER_NAMES)
    actual = set(registry)
    if expected != actual:
        missing = sorted(actual - expected)
        stale = sorted(expected - actual)
        raise RuntimeError(
            "The baseline catalog does not match the local Braindecode "
            f"classification registry; add={missing}, remove={stale}"
        )
    return registry


def get_baseline_spec(name: str) -> BaselineSpec:
    """Resolve either an architecture name or its stable experiment key."""
    normalized = name.casefold()
    spec = _SPEC_BY_ARCHITECTURE.get(normalized)
    if spec is None:
        spec = _SPEC_BY_MODEL_KEY.get(normalized)
    if spec is None:
        available = ", ".join(spec.architecture for spec in BASELINE_SPECS)
        raise ValueError(
            f"Unknown Braindecode baseline {name!r}; available: {available}"
        )
    return spec


def is_braindecode_model_key(name: str) -> bool:
    """Return whether *name* is a generated Braindecode model key."""
    return name.casefold() in _SPEC_BY_MODEL_KEY


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _build_reordered_biosemi128_chs_info(
    *,
    n_channels: int,
    sfreq: float,
    n_electrode_groups: int,
) -> list[dict]:
    """Build real channel metadata in the dataset's spatially reordered order."""
    if n_channels != 128:
        raise ValueError(
            "Coordinate-aware baselines currently require the configured "
            "BioSemi-128 dataset (n_channels=128)"
        )

    import mne

    from eeg_pipeline.data.ds003626 import (
        BIOSEMI128_MONTAGE,
        build_biosemi128_electrode_layout,
    )

    montage = mne.channels.make_standard_montage(BIOSEMI128_MONTAGE)
    layout = build_biosemi128_electrode_layout(
        montage.ch_names,
        n_electrode_groups=n_electrode_groups,
    )
    info = mne.create_info(
        ch_names=list(layout.reordered_channel_names),
        sfreq=sfreq,
        ch_types="eeg",
    )
    info.set_montage(montage, match_case=False, on_missing="raise")
    return list(info["chs"])


def _constructor_metadata(
    architecture: str,
    *,
    n_classes: int,
    n_channels: int,
    n_times: int,
    sfreq: float,
    n_electrode_groups: int,
    chs_info: list[dict] | tuple[dict, ...] | None = None,
) -> dict:
    registry = _validate_catalog_against_braindecode()
    required = registry[architecture]
    values: dict[str, object] = {
        "n_outputs": n_classes,
        "n_chans": n_channels,
        "n_times": n_times,
        "sfreq": sfreq,
        "input_window_seconds": n_times / sfreq,
    }
    if "chs_info" in required:
        if chs_info is None:
            chs_info = _build_reordered_biosemi128_chs_info(
                n_channels=n_channels,
                sfreq=sfreq,
                n_electrode_groups=n_electrode_groups,
            )
        if len(chs_info) != n_channels:
            raise ValueError(
                "chs_info length does not match n_channels: "
                f"{len(chs_info)} != {n_channels}"
            )
        values["chs_info"] = list(chs_info)
    return {name: values[name] for name in required}


def build_braindecode_baseline(
    n_classes: int,
    architecture: str,
    *,
    n_channels: int = 128,
    n_times: int = 1153,
    sfreq: float = 256.0,
    n_electrode_groups: int = 16,
    chs_info: list[dict] | tuple[dict, ...] | None = None,
) -> nn.Module:
    """Instantiate one official class with dataset metadata only."""
    n_classes = _positive_integer(n_classes, "n_classes")
    n_channels = _positive_integer(n_channels, "n_channels")
    n_times = _positive_integer(n_times, "n_times")
    n_electrode_groups = _positive_integer(
        n_electrode_groups, "n_electrode_groups"
    )
    sfreq = float(sfreq)
    if not math.isfinite(sfreq) or sfreq <= 0.0:
        raise ValueError("sfreq must be a finite positive number")

    spec = get_baseline_spec(architecture)
    if not spec.runnable:
        raise ValueError(f"{spec.architecture} is excluded: {spec.reason}")

    # Direct Labram interprets a 128-channel tensor without ``ch_names`` as its
    # own canonical 10-20 order.  This dataset is reordered BioSemi-128, whose
    # A1-D32 names are not LaBraM canonical names.  Silently running it would
    # attach wrong learned position embeddings; InterpolatedLaBraM is the
    # official montage-safe alternative and is independently included.
    if spec.architecture == "Labram":
        raise ValueError(
            "Labram cannot safely consume spatially reordered BioSemi-128 "
            "through the pipeline's model(eeg) call: its forward method "
            "requires canonical LaBraM channel names/order. "
            "InterpolatedLaBraM is included as the official montage adapter."
        )

    import braindecode.models as braindecode_models

    try:
        model_class = getattr(braindecode_models, spec.architecture)
    except AttributeError:
        version = getattr(__import__("braindecode"), "__version__", "unknown")
        raise RuntimeError(
            f"Braindecode {version} does not export {spec.architecture}"
        ) from None

    constructor_kwargs = _constructor_metadata(
        spec.architecture,
        n_classes=n_classes,
        n_channels=n_channels,
        n_times=n_times,
        sfreq=sfreq,
        n_electrode_groups=n_electrode_groups,
        chs_info=chs_info,
    )
    model = model_class(**constructor_kwargs)
    if type(model) is not model_class:
        raise RuntimeError(
            f"{spec.architecture} constructor returned {type(model).__name__}"
        )
    return model


def build_braindecode_from_model_key(
    name: str,
    n_classes: int,
    **model_kwargs,
) -> nn.Module:
    """Build a baseline from the stable key stored in a config."""
    spec = get_baseline_spec(name)
    return build_braindecode_baseline(
        n_classes=n_classes,
        architecture=spec.architecture,
        **model_kwargs,
    )


def _resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _resolve_output_root(config: dict) -> Path:
    root = Path(config["training"]["output_root"])
    root = root if root.is_absolute() else PROJECT_ROOT / root
    dataset_name = str(config["data"].get("dataset", "ds003626"))
    experiment_name = str(
        config["data"].get("experiment_name", dataset_name)
    )
    if dataset_name != "ds003626":
        root = root / experiment_name
    return root


def _baseline_runtime_context(config: dict) -> tuple[dict, int]:
    """Infer exact signal metadata from the selected dataset."""
    from eeg_pipeline.data.dataset import EEGTaskDataset
    from eeg_pipeline.data.registry import load_task_sessions

    data_config = config["data"]
    subjects = tuple(data_config["subjects"])
    if not subjects:
        raise ValueError("data.subjects must not be empty")
    task_sessions = load_task_sessions(
        data_config.get("dataset", "ds003626"),
        data_config["root"],
        data_config["task"],
        subjects=(subjects[0],),
        sessions=data_config.get("sessions", "auto"),
        preload=False,
        options=data_config.get("options"),
    )
    dataset = EEGTaskDataset(
        task_sessions,
        standardize=bool(data_config.get("standardize", False)),
        n_electrode_groups=int(data_config.get("n_electrode_groups", 1)),
    )
    reference = task_sessions[0]
    reference_info = reference.epochs.info
    source_indices = (
        tuple(range(len(reference.epochs.ch_names)))
        if reference.channel_indices is None
        else reference.channel_indices
    )
    chs_info = []
    for output_index, permutation_index in enumerate(
        dataset.channel_permutation
    ):
        channel = deepcopy(
            reference_info["chs"][source_indices[int(permutation_index)]]
        )
        channel["ch_name"] = dataset.channel_names[output_index]
        chs_info.append(channel)
    inferred = {
        "n_channels": dataset.num_channels,
        "n_times": dataset.num_times,
        "sfreq": dataset.sampling_rate_hz,
        "n_electrode_groups": dataset.n_electrode_groups,
    }
    configured = config.get("baseline", {})
    for key, value in inferred.items():
        if key not in configured:
            continue
        expected = float(configured[key]) if key == "sfreq" else int(configured[key])
        if expected != value:
            raise ValueError(
                f"baseline.{key}={configured[key]} does not match the "
                f"selected dataset ({value})"
            )
    return ({**inferred, "chs_info": chs_info}, len(dataset.class_names))


def _serializable_model_metadata(model_kwargs: dict, n_classes: int) -> dict:
    chs_info = model_kwargs.get("chs_info", ())
    return {
        key: value
        for key, value in model_kwargs.items()
        if key != "chs_info"
    } | {
        "n_classes": n_classes,
        "channel_names": [str(channel["ch_name"]) for channel in chs_info],
    }


def _catalog_payload(config: dict) -> dict:
    import braindecode

    registry = _validate_catalog_against_braindecode()
    model_kwargs, n_classes = _baseline_runtime_context(config)
    metadata = _serializable_model_metadata(model_kwargs, n_classes)
    fold_indices = [int(value) for value in config["split"]["fold_indices"]]
    return {
        "braindecode_version": braindecode.__version__,
        "catalog_rule": (
            "models_mandatory_parameters minus non_classification_models"
        ),
        "task": str(config["data"]["task"]),
        "dataset": str(
            config["data"].get(
                "experiment_name",
                config["data"].get("dataset", "ds003626"),
            )
        ),
        "dataset_adapter": str(
            config["data"].get("dataset", "ds003626")
        ),
        "fold_indices": fold_indices,
        "expected_folds": [index + 1 for index in fold_indices],
        "training_monitor": str(config["training"]["monitor"]),
        "input_shape_without_batch": [
            model_kwargs["n_channels"],
            model_kwargs["n_times"],
        ],
        "sampling_rate_hz": model_kwargs["sfreq"],
        "n_electrode_groups": model_kwargs["n_electrode_groups"],
        "n_classes": n_classes,
        "channel_names": metadata["channel_names"],
        "models": [
            {
                **asdict(spec),
                "model_key": spec.model_key,
                "required_signal_parameters": list(
                    registry[spec.architecture]
                ),
            }
            for spec in BASELINE_SPECS
        ],
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def _parameter_counts(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def _run_architecture(config_path: Path, architecture: str) -> None:
    from main import load_config, run_experiment

    config_path = _resolve_project_path(config_path)
    config = deepcopy(load_config(config_path))
    resolve_config_data_root(config, repository_root=PROJECT_ROOT)
    spec = get_baseline_spec(architecture)
    model_kwargs, n_classes = _baseline_runtime_context(config)
    config["model"] = {
        "name": spec.model_key,
        "kwargs": model_kwargs,
    }

    fold_indices = [int(value) for value in config["split"]["fold_indices"]]
    output_root = _resolve_output_root(config)
    status_path = output_root / "_status" / f"{spec.model_key}.json"
    status = {
        "architecture": spec.architecture,
        "model_key": spec.model_key,
        "paper_year": spec.paper_year,
        "status": "running",
        "task": str(config["data"]["task"]),
        "dataset": str(
            config["data"].get(
                "experiment_name",
                config["data"].get("dataset", "ds003626"),
            )
        ),
        "fold_indices": fold_indices,
        "expected_folds": [index + 1 for index in fold_indices],
        "config": str(config_path),
        "constructor_dataset_metadata": _serializable_model_metadata(
            model_kwargs,
            n_classes,
        ),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(status_path, status)
    print(
        f"Braindecode baseline={spec.architecture}, "
        f"model_key={spec.model_key}"
    )

    try:
        # Count the exact official model on CPU before training.  This both
        # records size when the later CUDA transfer/OOM fails and guarantees
        # the count corresponds to the same constructor used by ``main.py``.
        inspected_model = build_braindecode_baseline(
            n_classes=n_classes,
            architecture=spec.architecture,
            **model_kwargs,
        )
        total_parameters, trainable_parameters = _parameter_counts(
            inspected_model
        )
        status.update(
            total_parameters=total_parameters,
            trainable_parameters=trainable_parameters,
            official_model_class=(
                f"{type(inspected_model).__module__}."
                f"{type(inspected_model).__qualname__}"
            ),
        )
        _write_json(status_path, status)
        del inspected_model
        gc.collect()

        results = run_experiment(config)
    except Exception as error:
        status.update(
            status="failed",
            reason=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
            finished_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        _write_json(status_path, status)
        raise

    status.update(
        status="completed",
        completed_folds=[int(result["fold"]) for result in results],
        results=results,
        finished_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    _write_json(status_path, status)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "braindecode_baselines.yaml",
    )
    parser.add_argument("--architecture")
    parser.add_argument("--list-runnable", action="store_true")
    parser.add_argument("--write-catalog", type=Path)
    parser.add_argument("--print-output-root", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_runnable:
        _validate_catalog_against_braindecode()
        for spec in BASELINE_SPECS:
            if spec.runnable:
                print(spec.architecture)
        return

    if args.print_output_root:
        from main import load_config

        config = load_config(_resolve_project_path(args.config))
        print(_resolve_output_root(config))
        return

    if args.write_catalog is not None:
        from main import load_config

        config = load_config(_resolve_project_path(args.config))
        _write_json(
            _resolve_project_path(args.write_catalog),
            _catalog_payload(config),
        )
        if args.architecture is None:
            return

    if args.architecture is None:
        raise SystemExit(
            "One of --architecture, --list-runnable, --write-catalog, or "
            "--print-output-root is required"
        )
    _run_architecture(args.config, args.architecture)


if __name__ == "__main__":
    main()
