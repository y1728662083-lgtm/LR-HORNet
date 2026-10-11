"""FAST adapter that keeps each dataset's native sampling rate."""

from __future__ import annotations

import math
import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import nn
from torch.nn import functional as F

import adapters as _base
import adapters_multidataset as _multidataset  # noqa: F401; installs montage lookup


def _self_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def nearest_window_grid_length(
    n_times: int,
    window_samples: int,
    stride_samples: int,
) -> tuple[int, int, int]:
    """Return closest valid window grid length and right pad/crop counts."""
    n_times = int(n_times)
    window_samples = int(window_samples)
    stride_samples = int(stride_samples)
    if min(n_times, window_samples, stride_samples) < 1:
        raise ValueError("time and window dimensions must be positive")
    if stride_samples > window_samples:
        raise ValueError("FAST stride_samples must not exceed window_samples")
    if n_times <= window_samples:
        return window_samples, window_samples - n_times, 0

    offset = n_times - window_samples
    floor_count = offset // stride_samples + 1
    floor_length = window_samples + (floor_count - 1) * stride_samples
    ceil_count = math.ceil(offset / stride_samples) + 1
    ceil_length = window_samples + (ceil_count - 1) * stride_samples
    crop = n_times - floor_length
    pad = ceil_length - n_times
    if crop <= pad:
        return floor_length, 0, crop
    return ceil_length, pad, 0


class NativeRateFASTAdapter(nn.Module):
    """Use official FAST unchanged after only minimal tail pad/crop."""

    def __init__(
        self,
        *,
        source: str,
        channel_names: Iterable[str],
        n_times: int,
        n_classes: int,
        source_rate: int,
        target_rate: int,
        window_samples: int,
        stride_samples: int,
        window_remainder: str = "nearest",
        **model_config: Any,
    ) -> None:
        super().__init__()
        if int(source_rate) != int(target_rate):
            raise ValueError(
                "Native FAST requires target_sampling_rate_hz to equal "
                "source_sampling_rate_hz"
            )
        if window_remainder != "nearest":
            raise ValueError("Native FAST supports window_remainder='nearest' only")
        adjusted, right_pad, right_crop = nearest_window_grid_length(
            n_times, window_samples, stride_samples
        )
        self.original_n_times = int(n_times)
        self.adjusted_n_times = adjusted
        self.right_padding = right_pad
        self.right_crop = right_crop
        self.inner = _base.FASTAdapter(
            source=source,
            channel_names=tuple(channel_names),
            n_times=adjusted,
            n_classes=n_classes,
            source_rate=source_rate,
            target_rate=target_rate,
            window_samples=window_samples,
            stride_samples=stride_samples,
            **model_config,
        )
        self.checkpoint_report = self.inner.checkpoint_report
        self.adapter_metadata = {
            **self.inner.adapter_metadata,
            "native_sampling_rate": True,
            "original_time_points": self.original_n_times,
            "model_time_points": self.adjusted_n_times,
            "right_zero_padding": self.right_padding,
            "right_crop": self.right_crop,
            "window_remainder_policy": window_remainder,
            "native_adapter_sha256": _self_sha256(),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.original_n_times:
            raise ValueError(
                f"Expected [batch, channel, {self.original_n_times}], got {tuple(x.shape)}"
            )
        if self.right_padding:
            x = F.pad(x, (0, self.right_padding), mode="constant", value=0.0)
        elif self.right_crop:
            x = x[..., : self.adjusted_n_times]
        # Deliberately bypass FASTAdapter.resample: inputs remain at native rate.
        return self.inner.model(x)


def build_fast_native_model(
    name: str,
    config: Mapping[str, Any],
    *,
    channel_names: Iterable[str],
    n_times: int,
    n_classes: int,
    input_config: Mapping[str, Any],
) -> nn.Module:
    if name != "fast":
        raise ValueError(f"This runner only builds FAST, got {name!r}")
    model_config = {
        key: value
        for key, value in config.items()
        if key not in {"enabled", "training", "checkpoint"}
    }
    return NativeRateFASTAdapter(
        channel_names=tuple(channel_names),
        n_times=int(n_times),
        n_classes=int(n_classes),
        source_rate=int(input_config["source_sampling_rate_hz"]),
        target_rate=int(input_config["target_sampling_rate_hz"]),
        **model_config,
    )
