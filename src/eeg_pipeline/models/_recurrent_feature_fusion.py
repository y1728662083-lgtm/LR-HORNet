"""Recurrent feature-collapse modules used by FilterNFC ablations."""

from __future__ import annotations

import math
import operator
from typing import Any

import torch
from braindecode.functional import glorot_weight_zero_bias
from braindecode.modules import LinearWithConstraint
from torch import Tensor, nn


__all__ = [
    "PointLSTMFeatureCollapse",
    "DualStateFeatureFusionLSTM",
    "replace_backbone_output_width",
]


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        parsed = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


class PointLSTMFeatureCollapse(nn.Module):
    """Replace a point convolution with a 16-to-1 recurrent collapse."""

    def __init__(
        self,
        n_input_features: int,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.n_input_features = _positive_integer(
            "n_input_features", n_input_features
        )
        self.n_output_features = 1
        self.lstm = nn.LSTM(
            input_size=self.n_input_features,
            hidden_size=self.n_output_features,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)

    def forward(self, input: Tensor) -> Tensor:
        if input.ndim != 4:
            raise ValueError(
                "point-LSTM collapse expects (B,F,1,T), got "
                f"{tuple(input.shape)}"
            )
        batch_size, n_features, spatial_size, n_times = input.shape
        if n_features != self.n_input_features or spatial_size != 1:
            raise ValueError(
                f"expected (B,{self.n_input_features},1,T), got "
                f"{tuple(input.shape)}"
            )
        sequence = input[:, :, 0, :].transpose(1, 2).contiguous()
        output, _ = self.lstm(sequence)
        return output.transpose(1, 2).reshape(
            batch_size, self.n_output_features, 1, n_times
        )


class DualStateFeatureFusionLSTM(nn.Module):
    """Feature-independent score/value recurrences with Softmax fusion.

    Each original feature is folded into the batch axis, so recurrent states
    never cross feature streams. The score and value branches share their
    parameters across features but do not share parameters with each other.
    """

    def __init__(
        self,
        n_input_features: int,
        *,
        hidden_size: int = 8,
        temperature: float = 1.0,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.n_input_features = _positive_integer(
            "n_input_features", n_input_features
        )
        self.hidden_size = _positive_integer("hidden_size", hidden_size)
        if isinstance(temperature, bool):
            raise ValueError("temperature must be finite and positive")
        self.temperature = float(temperature)
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")

        self.score_lstm = nn.LSTM(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.value_lstm = nn.LSTM(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        # A shared scalar bias would cancel exactly in the feature Softmax.
        self.score_head = nn.Linear(self.hidden_size, 1, bias=False)
        self.value_head = nn.Linear(self.hidden_size, 1, bias=False)
        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)

    def forward_components(
        self, input: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return fused output, scores, fusion weights, and candidate values."""
        if input.ndim != 4:
            raise ValueError(
                "dual-state fusion expects (B,F,1,T), got "
                f"{tuple(input.shape)}"
            )
        batch_size, n_features, spatial_size, n_times = input.shape
        if n_features != self.n_input_features or spatial_size != 1:
            raise ValueError(
                f"expected (B,{self.n_input_features},1,T), got "
                f"{tuple(input.shape)}"
            )

        sequences = input[:, :, 0, :].reshape(
            batch_size * n_features, n_times, 1
        )
        score_states, _ = self.score_lstm(sequences)
        value_states, _ = self.value_lstm(sequences)
        scores = self.score_head(score_states).reshape(
            batch_size, n_features, n_times
        )
        values = self.value_head(value_states).reshape(
            batch_size, n_features, n_times
        )
        weights = torch.softmax(scores / self.temperature, dim=1)
        fused = (weights * values).sum(dim=1, keepdim=True).unsqueeze(2)
        return fused, scores, weights, values

    def forward(self, input: Tensor) -> Tensor:
        fused, _, _, _ = self.forward_components(input)
        return fused


def replace_backbone_output_width(
    model: Any,
    *,
    n_output_features: int,
    final_conv_length: int,
    reference_parameter: Tensor,
) -> None:
    """Resize BN2 and the classifier after a feature-width-changing module."""
    n_output_features = _positive_integer(
        "n_output_features", n_output_features
    )
    final_conv_length = _positive_integer(
        "final_conv_length", final_conv_length
    )

    old_norm = model.bnorm_2
    if isinstance(old_norm, nn.BatchNorm2d):
        new_norm: nn.Module = nn.BatchNorm2d(
            n_output_features,
            eps=old_norm.eps,
            momentum=old_norm.momentum,
            affine=old_norm.affine,
            track_running_stats=old_norm.track_running_stats,
            device=reference_parameter.device,
            dtype=reference_parameter.dtype,
        )
    elif isinstance(old_norm, nn.GroupNorm):
        new_norm = nn.GroupNorm(
            num_groups=1,
            num_channels=n_output_features,
            eps=old_norm.eps,
            affine=old_norm.affine,
            device=reference_parameter.device,
            dtype=reference_parameter.dtype,
        )
    else:
        raise TypeError(
            "bnorm_2 must be BatchNorm2d or GroupNorm, got "
            f"{type(old_norm).__name__}"
        )
    model.bnorm_2 = new_norm
    model.F2 = n_output_features
    model.final_conv_length = final_conv_length

    if hasattr(model.final_layer, "linearconstraint"):
        classifier = LinearWithConstraint(
            in_features=n_output_features * final_conv_length,
            out_features=model.n_outputs,
            max_norm=model.norm_rate,
        ).to(
            device=reference_parameter.device,
            dtype=reference_parameter.dtype,
        )
        model.final_layer.linearconstraint = classifier
    else:
        classifier = nn.Conv2d(
            n_output_features,
            model.n_outputs,
            kernel_size=(1, final_conv_length),
            bias=True,
        ).to(
            device=reference_parameter.device,
            dtype=reference_parameter.dtype,
        )
        model.final_layer.conv_classifier = classifier
    glorot_weight_zero_bias(model.final_layer)
