"""LaBraM-only adapter around the pinned official implementation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from common import import_module_from_file, resolve_from_addition, sha256_file


# Exact order in the official LaBraM utils.py. The model's pos_embed has one
# CLS entry followed by these 128 channel positions.
LABRAM_STANDARD_1020 = (
    "FP1", "FPZ", "FP2", "AF9", "AF7", "AF5", "AF3", "AF1", "AFZ",
    "AF2", "AF4", "AF6", "AF8", "AF10", "F9", "F7", "F5", "F3",
    "F1", "FZ", "F2", "F4", "F6", "F8", "F10", "FT9", "FT7",
    "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8",
    "FT10", "T9", "T7", "C5", "C3", "C1", "CZ", "C2", "C4",
    "C6", "T8", "T10", "TP9", "TP7", "CP5", "CP3", "CP1",
    "CPZ", "CP2", "CP4", "CP6", "TP8", "TP10", "P9", "P7",
    "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "P10",
    "PO9", "PO7", "PO5", "PO3", "PO1", "POZ", "PO2", "PO4",
    "PO6", "PO8", "PO10", "O1", "OZ", "O2", "O9", "CB1", "CB2",
    "IZ", "O10", "T3", "T5", "T4", "T6", "M1", "M2", "A1",
    "A2", "CFC1", "CFC2", "CFC3", "CFC4", "CFC5", "CFC6", "CFC7",
    "CFC8", "CCP1", "CCP2", "CCP3", "CCP4", "CCP5", "CCP6",
    "CCP7", "CCP8", "T1", "T2", "FTT9h", "TTP7h", "TPP9h",
    "FTT10h", "TPP8h", "TPP10h", "FP1-F7", "F7-T7", "T7-P7",
    "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FP1-F3",
    "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4",
    "P4-O2",
)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError(f"Invalid electrode coordinate: {vector}")
    return vector / norm


def _montage_positions(kind: str) -> dict[str, tuple[str, np.ndarray]]:
    import mne

    positions = mne.channels.make_standard_montage(kind).get_positions()["ch_pos"]
    return {
        name.casefold(): (name, _unit(np.asarray(position, dtype=float)))
        for name, position in positions.items()
    }


MONTAGE_CANDIDATES = (
    "biosemi128",
    "biosemi64",
    "standard_1020",
    "standard_1005",
)


def _dataset_positions(channel_names: Iterable[str]) -> dict[str, np.ndarray]:
    """Resolve a complete channel set against one authoritative MNE montage."""
    names = tuple(str(name) for name in channel_names)
    if not names:
        raise ValueError("channel_names must not be empty")
    attempts: list[str] = []
    for montage_name in MONTAGE_CANDIDATES:
        available = _montage_positions(montage_name)
        missing = [name for name in names if name.casefold() not in available]
        if not missing:
            return {name: available[name.casefold()][1] for name in names}
        attempts.append(
            f"{montage_name}: missing {', '.join(missing[:8])}"
            + (" ..." if len(missing) > 8 else "")
        )
    raise ValueError(
        "No single supported MNE montage covers every EEG channel. "
        + " | ".join(attempts)
    )


def _nearest_reference(
    position: np.ndarray,
    references: Mapping[str, np.ndarray],
) -> str:
    # Unit-vector cosine distance is insensitive to montage radius conventions.
    return min(references, key=lambda name: float(1.0 - position @ references[name]))


def build_labram_channel_mapping(
    channel_names: Iterable[str],
) -> tuple[list[int], dict[str, str]]:
    channel_names = tuple(str(name) for name in channel_names)
    source_positions = _dataset_positions(channel_names)
    standard = _montage_positions("standard_1020")
    supported: dict[str, np.ndarray] = {}
    supported_index: dict[str, int] = {}
    for index, configured_name in enumerate(LABRAM_STANDARD_1020, start=1):
        if index > 128:
            continue
        item = standard.get(configured_name.casefold())
        if item is not None:
            canonical_name, position = item
            supported[canonical_name] = position
            supported_index[canonical_name] = index
    if len(supported) < 70:
        raise RuntimeError("Too few LaBraM positional references found in MNE")
    input_chans = [0]
    nearest_reference: dict[str, str] = {}
    for source_name in channel_names:
        reference = _nearest_reference(source_positions[source_name], supported)
        nearest_reference[source_name] = reference
        input_chans.append(supported_index[reference])
    if len(input_chans) != len(channel_names) + 1:
        raise RuntimeError("LaBraM channel mapping has an invalid length")
    return input_chans, nearest_reference


class ResampleTo200(nn.Module):
    def __init__(self, source_rate: int = 256, target_rate: int = 200):
        super().__init__()
        self.source_rate = int(source_rate)
        self.target_rate = int(target_rate)

    def target_length(self, source_length: int) -> int:
        return int(round(source_length * self.target_rate / self.source_rate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, channel, time], got {tuple(x.shape)}")
        return F.interpolate(
            x,
            size=self.target_length(x.shape[-1]),
            mode="linear",
            align_corners=False,
        )


def patchify_200(x: torch.Tensor, patch_size: int = 200) -> tuple[torch.Tensor, int]:
    remainder = x.shape[-1] % patch_size
    padding = (patch_size - remainder) % patch_size
    if padding:
        x = F.pad(x, (0, padding), mode="constant", value=0.0)
    return x.unfold(-1, patch_size, patch_size).contiguous(), padding


def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("model", "state_dict", "model_state_dict", "module"):
            nested = payload.get(key)
            if isinstance(nested, dict) and nested:
                payload = nested
                break
    if not isinstance(payload, dict) or not payload:
        raise TypeError("Checkpoint does not contain a non-empty state dictionary")
    return {str(key): value for key, value in payload.items() if torch.is_tensor(value)}


def _load_shape_checked(
    module: nn.Module,
    state: Mapping[str, torch.Tensor],
    *,
    removable_prefixes: tuple[str, ...] = (),
    excluded_prefixes: tuple[str, ...] = (),
    minimum_element_coverage: float = 0.90,
) -> dict[str, Any]:
    target = module.state_dict()
    matched: dict[str, torch.Tensor] = {}
    unexpected: list[str] = []
    shape_mismatch: list[dict[str, Any]] = []
    for original_key, value in state.items():
        key = original_key
        changed = True
        while changed:
            changed = False
            for prefix in removable_prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
        if any(key.startswith(prefix) for prefix in excluded_prefixes):
            continue
        if key not in target:
            unexpected.append(original_key)
        elif tuple(value.shape) != tuple(target[key].shape):
            shape_mismatch.append(
                {"key": key, "checkpoint": list(value.shape), "model": list(target[key].shape)}
            )
        else:
            matched[key] = value

    required = {
        key: value
        for key, value in target.items()
        if not any(key.startswith(prefix) for prefix in excluded_prefixes)
    }
    total_elements = sum(value.numel() for value in required.values())
    loaded_elements = sum(required[key].numel() for key in matched if key in required)
    coverage = loaded_elements / max(total_elements, 1)
    if coverage < minimum_element_coverage:
        raise RuntimeError(
            f"Checkpoint load coverage {coverage:.2%} is below the required "
            f"{minimum_element_coverage:.2%}; matched={len(matched)}, "
            f"target={len(required)}, shape_mismatch={shape_mismatch[:5]}"
        )
    incompatible = module.load_state_dict(matched, strict=False)
    return {
        "matched_tensors": len(matched),
        "target_tensors": len(required),
        "element_coverage": coverage,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_source_keys": unexpected,
        "shape_mismatch": shape_mismatch,
    }


class LaBraMAdapter(nn.Module):
    def __init__(
        self,
        *,
        source: str | Path,
        checkpoint: str | Path | None,
        checkpoint_required: bool,
        checkpoint_kind: str,
        channel_names: Iterable[str],
        n_times: int,
        n_classes: int,
        source_rate: int = 256,
        target_rate: int = 200,
        patch_size: int = 200,
        init_values: float = 0.1,
        drop_path_rate: float = 0.1,
        **_: Any,
    ) -> None:
        super().__init__()
        if patch_size != 200:
            raise ValueError("The released LaBraM Base model requires 200-point patches")
        source_path = resolve_from_addition(source)
        if source_path is None or not source_path.is_file():
            raise FileNotFoundError(f"LaBraM official source not found: {source_path}")
        official = import_module_from_file("paper4_official_labram", source_path)
        self.model = official.labram_base_patch200_200(
            pretrained=False,
            num_classes=int(n_classes),
            use_mean_pooling=True,
            use_abs_pos_emb=True,
            use_rel_pos_bias=False,
            use_shared_rel_pos_bias=False,
            qkv_bias=False,
            init_values=float(init_values),
            drop_path_rate=float(drop_path_rate),
        )
        input_chans, nearest = build_labram_channel_mapping(channel_names)
        self.register_buffer(
            "input_chans",
            torch.tensor(input_chans, dtype=torch.long),
            persistent=True,
        )
        self.resample = ResampleTo200(source_rate, target_rate)
        self.patch_size = patch_size

        checkpoint_path = resolve_from_addition(checkpoint)
        if checkpoint_path is None or not checkpoint_path.is_file():
            if checkpoint_required:
                raise FileNotFoundError(f"Required LaBraM checkpoint not found: {checkpoint_path}")
            self.checkpoint_report = {"pretrained": False, "path": str(checkpoint_path)}
        else:
            if checkpoint_path.suffix == ".safetensors":
                from safetensors.torch import load_file

                state = load_file(str(checkpoint_path), device="cpu")
            else:
                state = _extract_state_dict(torch.load(checkpoint_path, map_location="cpu"))
            if checkpoint_kind == "student_mirror":
                state = {
                    key[len("backbone.student."):]: value
                    for key, value in state.items()
                    if key.startswith("backbone.student.")
                }
            report = _load_shape_checked(
                self.model,
                state,
                removable_prefixes=("module.", "model.", "backbone."),
                excluded_prefixes=("head.", "lm_head.", "mask_token"),
                minimum_element_coverage=0.95,
            )
            self.checkpoint_report = {
                "pretrained": True,
                "path": str(checkpoint_path),
                "sha256": sha256_file(checkpoint_path),
                "kind": checkpoint_kind,
                **report,
            }
        self.adapter_metadata = {
            "official_class": "modeling_finetune.labram_base_patch200_200",
            "source": str(source_path),
            "source_sha256": sha256_file(source_path),
            "source_rate_hz": source_rate,
            "target_rate_hz": target_rate,
            "source_time_points": n_times,
            "resampled_time_points": self.resample.target_length(n_times),
            "patch_size_samples": patch_size,
            "patch_count_after_zero_pad": math.ceil(self.resample.target_length(n_times) / patch_size),
            "input_chans": input_chans,
            "nearest_reference_electrode": nearest,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resample(x)
        x, _ = patchify_200(x, self.patch_size)
        return self.model(x, input_chans=self.input_chans)


def build_official_model(
    name: str,
    config: Mapping[str, Any],
    *,
    channel_names: Iterable[str],
    n_times: int,
    n_classes: int,
    input_config: Mapping[str, Any],
) -> nn.Module:
    if name != "labram":
        raise KeyError(f"This bundle only provides LaBraM, got {name!r}")
    model_config = {
        key: value
        for key, value in config.items()
        if key not in {"enabled", "training"}
    }
    return LaBraMAdapter(
        channel_names=tuple(channel_names),
        n_times=int(n_times),
        n_classes=int(n_classes),
        source_rate=int(input_config["source_sampling_rate_hz"]),
        target_rate=int(input_config["target_sampling_rate_hz"]),
        patch_size=int(input_config["patch_size_samples"]),
        **model_config,
    )
