"""Boundary utilities shared by Patch-based temporal encoders."""

from __future__ import annotations

import math

from torch import Tensor
from torch.nn import functional as F


def patch_layout(
    n_times: int,
    patch_size: int,
    patch_stride: int,
    *,
    pad_incomplete_patch: bool,
) -> tuple[int, int]:
    """Return Patch count and covered length for one temporal sequence."""
    if n_times < patch_size:
        raise ValueError("n_times must be at least patch_size")
    if patch_size < 1 or patch_stride < 1 or patch_stride > patch_size:
        raise ValueError("require 1 <= patch_stride <= patch_size")
    if not isinstance(pad_incomplete_patch, bool):
        raise TypeError("pad_incomplete_patch must be a bool")

    relative_length = n_times - patch_size
    if pad_incomplete_patch:
        n_patches = math.ceil(relative_length / patch_stride) + 1
    else:
        n_patches = relative_length // patch_stride + 1
    effective_n_times = patch_size + (n_patches - 1) * patch_stride
    return n_patches, effective_n_times


def fit_right_boundary(input: Tensor, effective_n_times: int) -> Tensor:
    """Crop an uncovered suffix or zero-pad an incomplete final Patch."""
    n_times = input.shape[-1]
    if effective_n_times > n_times:
        return F.pad(
            input,
            (0, effective_n_times - n_times),
            mode="constant",
            value=0.0,
        )
    return input[..., :effective_n_times]
