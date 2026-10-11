"""Deformation-specific ablations for Group-4 Patch-LSTM EEGNet.

The Patch-LSTM temporal encoder and all downstream EEGNet layers are held
fixed. Only the mechanism that produces temporal sampling offsets changes.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from eeg_pipeline.models.causal1_group8PatchLSTM_EEGNet import (
    _positive_integer,
    causal1_group8PatchLSTM_EEGNet,
)


__all__ = [
    "group4_patchLstm_noDeformation",
    "group4_patchLstm_pointwiseDeformation",
    "group4_patchLstm_staticDeformation",
    "group4_patchLstm_groupedConvDeformation",
    "build_group4_patchlstm_nodeformation",
    "build_group4_patchlstm_pointwisedeformation",
    "build_group4_patchlstm_staticdeformation",
    "build_group4_patchlstm_groupedconvdeformation",
]


def _require_four_groups(value: int) -> int:
    value = _positive_integer("n_electrode_groups", value)
    if value != 4:
        raise ValueError(
            "Group-4 ablations require n_electrode_groups=4, "
            f"received {value}."
        )
    return value


def _validate_warp_input(
    input: Tensor,
    n_feature_maps: int,
) -> tuple[int, int]:
    if input.ndim != 4:
        raise ValueError(
            "temporal deformation expects (B,M,1,T), "
            f"got shape {tuple(input.shape)}."
        )
    batch_size, feature_maps, spatial_size, n_times = input.shape
    if feature_maps != n_feature_maps or spatial_size != 1:
        raise ValueError(
            f"expected (B,{n_feature_maps},1,T), got {tuple(input.shape)}."
        )
    return batch_size, n_times


def _sample_with_offset(input: Tensor, offset: Tensor) -> Tensor:
    """Linearly sample input at time t + offset with border padding."""
    batch_size, n_feature_maps, _, n_times = input.shape
    expected_shape = (batch_size, n_feature_maps, n_times)
    if tuple(offset.shape) != expected_shape:
        raise ValueError(
            f"expected offset shape {expected_shape}, got {tuple(offset.shape)}."
        )

    flattened_offset = offset.reshape(batch_size * n_feature_maps, n_times)
    signal = input.reshape(batch_size * n_feature_maps, 1, 1, n_times)
    base_time = torch.arange(
        n_times,
        device=input.device,
        dtype=input.dtype,
    ).unsqueeze(0)
    sample_time = base_time + flattened_offset
    if n_times == 1:
        normalized_time = torch.zeros_like(sample_time)
    else:
        normalized_time = 2.0 * sample_time / (n_times - 1) - 1.0
    grid = torch.stack(
        (
            normalized_time.unsqueeze(1),
            torch.zeros_like(normalized_time).unsqueeze(1),
        ),
        dim=-1,
    )
    warped = F.grid_sample(
        signal,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return warped.reshape(batch_size, n_feature_maps, 1, n_times)


class _SharedPointwiseOffsetWarp(nn.Module):
    """Predict offsets from the current feature value without history."""

    def __init__(
        self,
        *,
        n_feature_maps: int,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.n_feature_maps = _positive_integer(
            "n_feature_maps", n_feature_maps
        )
        factory_kwargs = {"device": device, "dtype": dtype}
        self.offset_head = nn.Linear(1, 1, **factory_kwargs)
        self.gate_head = nn.Linear(1, 1, **factory_kwargs)
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)

    def forward(self, input: Tensor) -> Tensor:
        batch_size, n_times = _validate_warp_input(
            input, self.n_feature_maps
        )
        values = input[:, :, 0, :].reshape(
            batch_size * self.n_feature_maps * n_times,
            1,
        )
        raw_offset = self.offset_head(values)
        gate = torch.sigmoid(self.gate_head(values))
        offset = (gate * raw_offset).reshape(
            batch_size,
            self.n_feature_maps,
            n_times,
        )
        return _sample_with_offset(input, offset)


class _StaticFeatureMapOffsetWarp(nn.Module):
    """Learn one input-independent offset for each spatial feature map."""

    def __init__(
        self,
        *,
        n_feature_maps: int,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.n_feature_maps = _positive_integer(
            "n_feature_maps", n_feature_maps
        )
        factory_kwargs = {"device": device, "dtype": dtype}
        self.raw_offset = nn.Parameter(
            torch.zeros(1, self.n_feature_maps, 1, **factory_kwargs)
        )
        self.gate_logit = nn.Parameter(
            torch.zeros(1, self.n_feature_maps, 1, **factory_kwargs)
        )

    def forward(self, input: Tensor) -> Tensor:
        batch_size, n_times = _validate_warp_input(
            input, self.n_feature_maps
        )
        offset = (
            torch.sigmoid(self.gate_logit) * self.raw_offset
        ).expand(batch_size, self.n_feature_maps, n_times)
        return _sample_with_offset(input, offset)


class _GroupedConvOffsetWarp(nn.Module):
    """Predict local temporal offsets with four grouped convolutions.

    Feature maps are divided into the same four groups as the Patch-LSTM
    backbone. Each group predicts its own time-varying offsets from a local
    temporal window. There is no recurrent state, so this control isolates
    convolutional offset prediction from the Full model's LSTM history.
    """

    def __init__(
        self,
        *,
        n_feature_maps: int,
        n_groups: int,
        kernel_size: int,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.n_feature_maps = _positive_integer(
            "n_feature_maps", n_feature_maps
        )
        self.n_groups = _positive_integer("offset_conv_groups", n_groups)
        self.kernel_size = _positive_integer(
            "offset_conv_kernel_size", kernel_size
        )
        if self.n_feature_maps % self.n_groups != 0:
            raise ValueError(
                "n_feature_maps must be divisible by offset_conv_groups; "
                f"received {self.n_feature_maps} and {self.n_groups}."
            )
        if self.kernel_size % 2 == 0:
            raise ValueError(
                "offset_conv_kernel_size must be odd so the offset sequence "
                f"keeps its length; received {self.kernel_size}."
            )

        factory_kwargs = {"device": device, "dtype": dtype}
        conv_kwargs = {
            "in_channels": self.n_feature_maps,
            "out_channels": self.n_feature_maps,
            "kernel_size": self.kernel_size,
            "padding": self.kernel_size // 2,
            "groups": self.n_groups,
            "bias": True,
            **factory_kwargs,
        }
        self.offset_head = nn.Conv1d(**conv_kwargs)
        self.gate_head = nn.Conv1d(**conv_kwargs)

        # Match the other deformation variants: start from identity sampling.
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)

    def forward(self, input: Tensor) -> Tensor:
        _validate_warp_input(input, self.n_feature_maps)
        values = input[:, :, 0, :]
        raw_offset = self.offset_head(values)
        gate = torch.sigmoid(self.gate_head(values))
        return _sample_with_offset(input, gate * raw_offset)


class _Group4PatchLstmBase(causal1_group8PatchLSTM_EEGNet):
    def __init__(
        self,
        *args,
        n_electrode_groups: int = 4,
        **kwargs,
    ) -> None:
        n_electrode_groups = _require_four_groups(n_electrode_groups)
        super().__init__(
            *args,
            n_electrode_groups=n_electrode_groups,
            allow_uneven_groups=True,
            **kwargs,
        )

    def _install_deformation(self, deformation: nn.Module) -> None:
        original_pool = self.pool_1
        self.pool_1 = nn.Sequential(
            OrderedDict(
                (
                    ("deformation", deformation),
                    ("downsample", original_pool),
                )
            )
        )
        self.deformation_axis = "time"
        self.deformation_placement = "after_spatial_before_pool1"


class group4_patchLstm_noDeformation(_Group4PatchLstmBase):
    """Patch-LSTM backbone with temporal resampling removed."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.deformation_axis = None
        self.deformation_placement = None


class group4_patchLstm_pointwiseDeformation(_Group4PatchLstmBase):
    """Replace the deformation LSTM with a shared pointwise predictor."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        reference_parameter = next(self.conv_temporal.parameters())
        deformation = _SharedPointwiseOffsetWarp(
            n_feature_maps=self.F1 * self.D,
            device=reference_parameter.device,
            dtype=reference_parameter.dtype,
        )
        self._install_deformation(deformation)
        self.offset_predictor = "shared_pointwise_linear"
        self.offset_gate = "sigmoid"


class group4_patchLstm_staticDeformation(_Group4PatchLstmBase):
    """Replace adaptive offsets with one static offset per feature map."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        reference_parameter = next(self.conv_temporal.parameters())
        deformation = _StaticFeatureMapOffsetWarp(
            n_feature_maps=self.F1 * self.D,
            device=reference_parameter.device,
            dtype=reference_parameter.dtype,
        )
        self._install_deformation(deformation)
        self.offset_predictor = "static_per_feature_map"
        self.offset_gate = "sigmoid"


class group4_patchLstm_groupedConvDeformation(_Group4PatchLstmBase):
    """Replace the deformation LSTM with grouped temporal convolution."""

    def __init__(
        self,
        *args,
        offset_conv_kernel_size: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        reference_parameter = next(self.conv_temporal.parameters())
        deformation = _GroupedConvOffsetWarp(
            n_feature_maps=self.F1 * self.D,
            n_groups=4,
            kernel_size=offset_conv_kernel_size,
            device=reference_parameter.device,
            dtype=reference_parameter.dtype,
        )
        self._install_deformation(deformation)
        self.offset_predictor = "grouped_temporal_conv1d"
        self.offset_conv_groups = deformation.n_groups
        self.offset_conv_kernel_size = deformation.kernel_size
        self.offset_gate = "sigmoid"


def build_group4_patchlstm_nodeformation(
    n_classes: int,
    *,
    n_channels: int = 128,
    n_times: int = 1153,
    **model_kwargs,
) -> group4_patchLstm_noDeformation:
    return group4_patchLstm_noDeformation(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        **model_kwargs,
    )


def build_group4_patchlstm_pointwisedeformation(
    n_classes: int,
    *,
    n_channels: int = 128,
    n_times: int = 1153,
    **model_kwargs,
) -> group4_patchLstm_pointwiseDeformation:
    return group4_patchLstm_pointwiseDeformation(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        **model_kwargs,
    )


def build_group4_patchlstm_staticdeformation(
    n_classes: int,
    *,
    n_channels: int = 128,
    n_times: int = 1153,
    **model_kwargs,
) -> group4_patchLstm_staticDeformation:
    return group4_patchLstm_staticDeformation(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        **model_kwargs,
    )

def build_group4_patchlstm_groupedconvdeformation(
    n_classes: int,
    *,
    n_channels: int = 128,
    n_times: int = 1153,
    **model_kwargs,
) -> group4_patchLstm_groupedConvDeformation:
    return group4_patchLstm_groupedConvDeformation(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        **model_kwargs,
    )
