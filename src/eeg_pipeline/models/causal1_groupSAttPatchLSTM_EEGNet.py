"""Patch-LSTM EEGNet with feature-wise regional gated spatial attention.

The parent Patch-LSTM temporal encoder is unchanged.  Attention is inserted
immediately before EEGNet's original depthwise spatial convolution.  Q, K and
V are produced by three independent feature-grouped pointwise convolutions;
attention is computed independently for every temporal feature, time sample
and contiguous electrode group.  The attended representation is projected
back to one scalar per token and fused through a learnable residual gate.

The original full-electrode spatial convolution remains responsible for
collapsing the electrode dimension, so all downstream EEGNet blocks retain
their original shapes and implementations.
"""

from __future__ import annotations

from collections import OrderedDict
import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from eeg_pipeline.models.causal1_group8PatchLSTM_EEGNet import (
    _positive_integer,
    causal1_group8PatchLSTM_EEGNet,
)


__all__ = [
    "causal1_groupSAttPatchLSTM_EEGNet",
    "build_causal1_groupsattpatchlstm_eegnet",
]


def _probability(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be in [0, 1), got {value!r}.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} must be in [0, 1), got {value!r}."
        ) from error
    if not math.isfinite(parsed) or not 0.0 <= parsed < 1.0:
        raise ValueError(f"{name} must be in [0, 1), got {parsed}.")
    return parsed


class _FeatureGroupedRegionalGatedSpatialAttention(nn.Module):
    """Attend over electrodes without mixing temporal feature dimensions.

    Input and output shapes are both ``(B, F, C, T)``.  The electrode axis is
    split into ``n_electrode_groups`` contiguous regions.  RoPE positions are
    local to each region and therefore rely on the dataset's deterministic
    within-region electrode ordering.
    """

    def __init__(
        self,
        *,
        n_features: int,
        n_chans: int,
        n_electrode_groups: int,
        attention_dim: int = 8,
        attention_dropout: float = 0.0,
        use_rope: bool = True,
        time_chunk_size: Optional[int] = 32,
        checkpoint_attention: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.n_features = _positive_integer("n_features", n_features)
        self.n_chans = _positive_integer("n_chans", n_chans)
        self.n_electrode_groups = _positive_integer(
            "n_electrode_groups", n_electrode_groups
        )
        if self.n_chans % self.n_electrode_groups != 0:
            raise ValueError(
                "n_chans must be divisible by n_electrode_groups; "
                f"got n_chans={self.n_chans} and "
                f"n_electrode_groups={self.n_electrode_groups}."
            )
        self.chans_per_group = self.n_chans // self.n_electrode_groups

        self.attention_dim = _positive_integer(
            "attention_dim", attention_dim
        )
        if self.attention_dim % 2 != 0:
            raise ValueError(
                "attention_dim must be even for RoPE; "
                f"got {self.attention_dim}."
            )
        self.attention_dropout = _probability(
            "attention_dropout", attention_dropout
        )
        if not isinstance(use_rope, bool):
            raise TypeError("use_rope must be a boolean")
        if not isinstance(checkpoint_attention, bool):
            raise TypeError("checkpoint_attention must be a boolean")
        self.use_rope = use_rope
        self.checkpoint_attention = checkpoint_attention
        if time_chunk_size is None:
            self.time_chunk_size = None
        else:
            self.time_chunk_size = _positive_integer(
                "time_chunk_size", time_chunk_size
            )

        factory_kwargs = {"device": device, "dtype": dtype}
        projected_features = self.n_features * self.attention_dim
        self.query_projection = nn.Conv2d(
            self.n_features,
            projected_features,
            kernel_size=1,
            groups=self.n_features,
            bias=False,
            **factory_kwargs,
        )
        self.key_projection = nn.Conv2d(
            self.n_features,
            projected_features,
            kernel_size=1,
            groups=self.n_features,
            bias=False,
            **factory_kwargs,
        )
        self.value_projection = nn.Conv2d(
            self.n_features,
            projected_features,
            kernel_size=1,
            groups=self.n_features,
            bias=False,
            **factory_kwargs,
        )
        self.output_projection = nn.Conv2d(
            projected_features,
            self.n_features,
            kernel_size=1,
            groups=self.n_features,
            bias=False,
            **factory_kwargs,
        )
        self.gate_projection = nn.Conv2d(
            self.n_features,
            self.n_features,
            kernel_size=1,
            groups=self.n_features,
            bias=True,
            **factory_kwargs,
        )
        nn.init.zeros_(self.gate_projection.weight)
        nn.init.constant_(self.gate_projection.bias, -2.0)

        rope_frequency = 1.0 / (
            10000.0
            ** (
                torch.arange(0, self.attention_dim, 2, dtype=torch.float32)
                / self.attention_dim
            )
        )
        self.register_buffer(
            "rope_frequency", rope_frequency, persistent=False
        )

    def _project_tokens(self, projection: nn.Conv2d, input: Tensor) -> Tensor:
        batch_size, _, n_chans, n_times = input.shape
        projected = projection(input)
        return projected.reshape(
            batch_size,
            self.n_features,
            self.attention_dim,
            self.n_electrode_groups,
            self.chans_per_group,
            n_times,
        ).permute(0, 1, 5, 3, 4, 2)

    def _apply_rope(self, input: Tensor) -> Tensor:
        positions = torch.arange(
            self.chans_per_group,
            device=input.device,
            dtype=torch.float32,
        )
        angles = torch.outer(positions, self.rope_frequency)
        cosine = angles.cos().to(dtype=input.dtype)
        sine = angles.sin().to(dtype=input.dtype)
        even = input[..., 0::2]
        odd = input[..., 1::2]
        rotated_even = even * cosine - odd * sine
        rotated_odd = even * sine + odd * cosine
        return torch.stack(
            (rotated_even, rotated_odd), dim=-1
        ).flatten(-2)

    def _forward_chunk(self, input: Tensor) -> Tensor:
        batch_size, _, n_chans, n_times = input.shape
        query = self._project_tokens(self.query_projection, input)
        key = self._project_tokens(self.key_projection, input)
        value = self._project_tokens(self.value_projection, input)
        if self.use_rope:
            query = self._apply_rope(query)
            key = self._apply_rope(key)

        query = query.reshape(
            -1, 1, self.chans_per_group, self.attention_dim
        )
        key = key.reshape(
            -1, 1, self.chans_per_group, self.attention_dim
        )
        value = value.reshape(
            -1, 1, self.chans_per_group, self.attention_dim
        )
        attended = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        attended = attended.reshape(
            batch_size,
            self.n_features,
            n_times,
            self.n_electrode_groups,
            self.chans_per_group,
            self.attention_dim,
        )
        attended = attended.permute(0, 1, 5, 3, 4, 2).reshape(
            batch_size,
            self.n_features * self.attention_dim,
            n_chans,
            n_times,
        )
        attended = self.output_projection(attended)
        gate = torch.sigmoid(self.gate_projection(input))
        return input + gate * attended

    def forward(self, input: Tensor) -> Tensor:
        if input.ndim != 4:
            raise ValueError(
                "spatial attention expects (B,F,C,T), "
                f"got shape {tuple(input.shape)}."
            )
        _, n_features, n_chans, n_times = input.shape
        if n_features != self.n_features or n_chans != self.n_chans:
            raise ValueError(
                "spatial attention expected "
                f"F={self.n_features}, C={self.n_chans}; got "
                f"F={n_features}, C={n_chans}."
            )

        chunk_size = self.time_chunk_size or n_times
        outputs = []
        for start in range(0, n_times, chunk_size):
            input_chunk = input[..., start : start + chunk_size]
            if (
                self.checkpoint_attention
                and self.training
                and torch.is_grad_enabled()
            ):
                output_chunk = checkpoint(
                    self._forward_chunk,
                    input_chunk,
                    use_reentrant=False,
                )
            else:
                output_chunk = self._forward_chunk(input_chunk)
            outputs.append(output_chunk)
        return torch.cat(outputs, dim=-1)


class causal1_groupSAttPatchLSTM_EEGNet(
    causal1_group8PatchLSTM_EEGNet
):
    """Add regional gated spatial attention before the original spatial conv."""

    def __init__(
        self,
        *args,
        spatial_attention_dim: int = 8,
        spatial_attention_dropout: float = 0.0,
        spatial_attention_use_rope: bool = True,
        spatial_attention_time_chunk_size: Optional[int] = 32,
        spatial_attention_checkpoint: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        reference_weight = self.conv_spatial.weight
        spatial_attention = _FeatureGroupedRegionalGatedSpatialAttention(
            n_features=self.F1,
            n_chans=self.n_chans,
            n_electrode_groups=self.n_electrode_groups,
            attention_dim=spatial_attention_dim,
            attention_dropout=spatial_attention_dropout,
            use_rope=spatial_attention_use_rope,
            time_chunk_size=spatial_attention_time_chunk_size,
            checkpoint_attention=spatial_attention_checkpoint,
            device=reference_weight.device,
            dtype=reference_weight.dtype,
        )
        ordered_modules = OrderedDict()
        for name, module in self._modules.items():
            if name == "conv_spatial":
                ordered_modules["spatial_attention"] = spatial_attention
            ordered_modules[name] = module
        self._modules = ordered_modules

        self.spatial_attention_dim = spatial_attention.attention_dim
        self.spatial_attention_dropout = spatial_attention.attention_dropout
        self.spatial_attention_use_rope = spatial_attention.use_rope
        self.spatial_attention_time_chunk_size = (
            spatial_attention.time_chunk_size
        )
        self.spatial_attention_checkpoint = (
            spatial_attention.checkpoint_attention
        )


def build_causal1_groupsattpatchlstm_eegnet(
    n_classes: int,
    *,
    n_channels: int = 128,
    n_times: int = 1153,
    **model_kwargs,
) -> causal1_groupSAttPatchLSTM_EEGNet:
    """Build the regional gated-spatial-attention Patch-LSTM EEGNet."""
    return causal1_groupSAttPatchLSTM_EEGNet(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        **model_kwargs,
    )
