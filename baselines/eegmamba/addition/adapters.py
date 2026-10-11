"""EEGMamba-only adapter around the pinned official implementation."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from common import resolve_from_addition, sha256_file


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


class EEGMambaAdapter(nn.Module):
    def __init__(
        self,
        *,
        source_root: str | Path,
        checkpoint: str | Path | None,
        checkpoint_required: bool,
        n_times: int,
        n_classes: int,
        source_rate: int = 256,
        target_rate: int = 200,
        patch_size: int = 200,
        classifier: str = "avgpooling_patch_reps",
        dropout: float = 0.1,
        **_: Any,
    ) -> None:
        super().__init__()
        if patch_size != 200:
            raise ValueError("The released EEGMamba spectral projection requires 200-point patches")
        source_root_path = resolve_from_addition(source_root)
        if source_root_path is None or not source_root_path.is_dir():
            raise FileNotFoundError(f"EEGMamba source root not found: {source_root_path}")
        sys.path.insert(0, str(source_root_path))
        try:
            wrapper_module = importlib.import_module("models.model_for_speech")
        finally:
            if sys.path[0] == str(source_root_path):
                sys.path.pop(0)
        module_path = Path(wrapper_module.__file__).resolve()
        if source_root_path.resolve() not in module_path.parents:
            raise ImportError(f"Imported an unexpected models package: {module_path}")
        params = SimpleNamespace(
            use_pretrained_weights=False,
            classifier=classifier,
            num_of_classes=int(n_classes),
            dropout=float(dropout),
            cuda=0,
        )
        self.model = wrapper_module.Model(params)
        self.resample = ResampleTo200(source_rate, target_rate)
        self.patch_size = patch_size

        checkpoint_path = resolve_from_addition(checkpoint)
        if checkpoint_path is None or not checkpoint_path.is_file():
            if checkpoint_required:
                raise FileNotFoundError(f"Required EEGMamba checkpoint not found: {checkpoint_path}")
            self.checkpoint_report = {"pretrained": False, "path": str(checkpoint_path)}
        else:
            payload = torch.load(checkpoint_path, map_location="cpu")
            state = _extract_state_dict(payload)
            report = _load_shape_checked(
                self.model.backbone,
                state,
                removable_prefixes=("module.", "backbone."),
                excluded_prefixes=("proj_out.",),
                minimum_element_coverage=0.95,
            )
            self.checkpoint_report = {
                "pretrained": True,
                "path": str(checkpoint_path),
                "sha256": sha256_file(checkpoint_path),
                **report,
            }
        self.adapter_metadata = {
            "official_class": "models.model_for_speech.Model",
            "source_root": str(source_root_path),
            "source_rate_hz": source_rate,
            "target_rate_hz": target_rate,
            "source_time_points": n_times,
            "resampled_time_points": self.resample.target_length(n_times),
            "patch_size_samples": patch_size,
            "classifier": classifier,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resample(x)
        x, _ = patchify_200(x, self.patch_size)
        return self.model(x)


def build_official_model(
    name: str,
    config: Mapping[str, Any],
    *,
    channel_names: Iterable[str],
    n_times: int,
    n_classes: int,
    input_config: Mapping[str, Any],
) -> nn.Module:
    if name != "eegmamba":
        raise KeyError(f"This bundle only provides EEGMamba, got {name!r}")
    model_config = {
        key: value
        for key, value in config.items()
        if key not in {"enabled", "training"}
    }
    return EEGMambaAdapter(
        channel_names=tuple(channel_names),
        n_times=int(n_times),
        n_classes=int(n_classes),
        source_rate=int(input_config["source_sampling_rate_hz"]),
        target_rate=int(input_config["target_sampling_rate_hz"]),
        patch_size=int(input_config["patch_size_samples"]),
        **model_config,
    )
