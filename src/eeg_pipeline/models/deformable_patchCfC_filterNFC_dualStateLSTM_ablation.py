"""Factorial ablations of dual-state recurrent feature fusion.

The classification and training-only FilterNFC paths are inherited unchanged.
Only the replacement for EEGNet's separable depthwise/pointwise block is
varied through two orthogonal factors:

``score_mode``
    ``"recurrent"`` learns time-dependent feature weights with an LSTM;
    ``"uniform"`` fixes every feature weight to ``1 / F``.

``value_mode``
    ``"recurrent"`` learns an independent recurrent candidate value for each
    feature; ``"identity"`` uses the incoming scalar feature unchanged.

The ``recurrent/recurrent`` cell instantiates the project's existing
``DualStateFeatureFusionLSTM`` directly.  It therefore preserves the complete
dual-state variant's module names, state-dict layout, initialization order and
forward semantics.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn

from eeg_pipeline.models._recurrent_feature_fusion import (
    DualStateFeatureFusionLSTM,
    replace_backbone_output_width,
)
from eeg_pipeline.models.deformable_patchCfC_filterNFC import (
    deformable_patchCfC_filterNFC,
)


ScoreMode = Literal["recurrent", "uniform"]
ValueMode = Literal["recurrent", "identity"]

__all__ = [
    "DualStateComponentAblationFusion",
    "build_dual_state_component_fusion",
    "deformable_patchCfC_filterNFC_dualStateLSTM_ablation",
    "build_deformable_patchcfc_filternfc_dualstatelstm_ablation",
]


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_temperature(value: float) -> float:
    if isinstance(value, bool):
        raise ValueError("temperature must be finite and positive")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError("temperature must be finite and positive")
    return parsed


def _choice(name: str, value: str, choices: tuple[str, ...]) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be one of {choices}")
    normalized = value.strip().lower()
    if normalized not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}")
    return normalized


class DualStateComponentAblationFusion(nn.Module):
    """Feature-wise score/value ablation with a common output contract.

    Input and output follow the complete dual-state module:
    ``(B,F,1,T) -> (B,1,1,T)``.  When present, recurrent branches fold the
    feature dimension into the batch dimension, so their hidden states remain
    independent across features while parameters are shared across features.
    """

    def __init__(
        self,
        n_input_features: int,
        *,
        hidden_size: int = 8,
        temperature: float = 1.0,
        score_mode: ScoreMode = "recurrent",
        value_mode: ValueMode = "recurrent",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.n_input_features = _positive_integer(
            "n_input_features", n_input_features
        )
        self.hidden_size = _positive_integer("hidden_size", hidden_size)
        self.temperature = _positive_temperature(temperature)
        self.score_mode = _choice(
            "score_mode", score_mode, ("recurrent", "uniform")
        )
        self.value_mode = _choice(
            "value_mode", value_mode, ("recurrent", "identity")
        )

        self.score_lstm: nn.LSTM | None = None
        self.score_head: nn.Linear | None = None
        if self.score_mode == "recurrent":
            self.score_lstm = nn.LSTM(
                input_size=1,
                hidden_size=self.hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=False,
            )
            # A shared scalar bias cancels exactly in the feature Softmax.
            self.score_head = nn.Linear(self.hidden_size, 1, bias=False)

        self.value_lstm: nn.LSTM | None = None
        self.value_head: nn.Linear | None = None
        if self.value_mode == "recurrent":
            self.value_lstm = nn.LSTM(
                input_size=1,
                hidden_size=self.hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=False,
            )
            self.value_head = nn.Linear(self.hidden_size, 1, bias=False)

        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)

    def forward_components(
        self, input: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return fused output, scores, weights and candidate values."""
        if input.ndim != 4:
            raise ValueError(
                "dual-state component ablation expects (B,F,1,T), got "
                f"{tuple(input.shape)}"
            )
        batch_size, n_features, spatial_size, n_times = input.shape
        if n_features != self.n_input_features or spatial_size != 1:
            raise ValueError(
                f"expected (B,{self.n_input_features},1,T), got "
                f"{tuple(input.shape)}"
            )

        raw_values = input[:, :, 0, :]
        sequences = raw_values.reshape(batch_size * n_features, n_times, 1)

        if self.score_mode == "recurrent":
            if self.score_lstm is None or self.score_head is None:
                raise RuntimeError("recurrent score branch was not constructed")
            score_states, _ = self.score_lstm(sequences)
            scores = self.score_head(score_states).reshape(
                batch_size, n_features, n_times
            )
            weights = torch.softmax(scores / self.temperature, dim=1)
        else:
            scores = input.new_zeros((batch_size, n_features, n_times))
            weights = input.new_full(
                (batch_size, n_features, n_times), 1.0 / n_features
            )

        if self.value_mode == "recurrent":
            if self.value_lstm is None or self.value_head is None:
                raise RuntimeError("recurrent value branch was not constructed")
            value_states, _ = self.value_lstm(sequences)
            values = self.value_head(value_states).reshape(
                batch_size, n_features, n_times
            )
        else:
            values = raw_values

        fused = (weights * values).sum(dim=1, keepdim=True).unsqueeze(2)
        return fused, scores, weights, values

    def forward(self, input: Tensor) -> Tensor:
        fused, _, _, _ = self.forward_components(input)
        return fused


def build_dual_state_component_fusion(
    n_input_features: int,
    *,
    hidden_size: int,
    temperature: float,
    score_mode: ScoreMode,
    value_mode: ValueMode,
    device=None,
    dtype=None,
) -> nn.Module:
    """Build one cell of the score-by-value factorial ablation.

    The complete cell deliberately delegates to the already-tested project
    implementation rather than duplicating it.
    """
    normalized_score = _choice(
        "score_mode", score_mode, ("recurrent", "uniform")
    )
    normalized_value = _choice(
        "value_mode", value_mode, ("recurrent", "identity")
    )
    if normalized_score == "recurrent" and normalized_value == "recurrent":
        return DualStateFeatureFusionLSTM(
            n_input_features,
            hidden_size=hidden_size,
            temperature=temperature,
            device=device,
            dtype=dtype,
        )
    return DualStateComponentAblationFusion(
        n_input_features,
        hidden_size=hidden_size,
        temperature=temperature,
        score_mode=normalized_score,
        value_mode=normalized_value,
        device=device,
        dtype=dtype,
    )


class deformable_patchCfC_filterNFC_dualStateLSTM_ablation(
    deformable_patchCfC_filterNFC
):
    """FilterNFC model with configurable score/value fusion components."""

    def __init__(
        self,
        *args,
        dual_state_hidden_size: int = 8,
        fusion_temperature: float = 1.0,
        score_mode: ScoreMode = "recurrent",
        value_mode: ValueMode = "recurrent",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        score_mode = _choice(
            "score_mode", score_mode, ("recurrent", "uniform")
        )
        value_mode = _choice(
            "value_mode", value_mode, ("recurrent", "identity")
        )
        old_depth = self.conv_separable_depth
        old_point = self.conv_separable_point
        n_input_features = old_depth.in_channels
        if (
            old_depth.out_channels != n_input_features
            or old_point.in_channels != n_input_features
            or old_point.out_channels != n_input_features
        ):
            raise RuntimeError(
                "dual-state ablation expects a width-preserving separable block"
            )

        reference_parameter = next(old_depth.parameters())
        self.conv_separable_depth = nn.Identity()
        self.conv_separable_point = build_dual_state_component_fusion(
            n_input_features,
            hidden_size=dual_state_hidden_size,
            temperature=fusion_temperature,
            score_mode=score_mode,
            value_mode=value_mode,
            device=reference_parameter.device,
            dtype=reference_parameter.dtype,
        )

        final_conv_length = self.temporal_output_length // self.pool2_kernel_size
        replace_backbone_output_width(
            self,
            n_output_features=1,
            final_conv_length=final_conv_length,
            reference_parameter=reference_parameter,
        )

        self.recurrent_fusion = (
            f"dual_state_ablation_score_{score_mode}_value_{value_mode}_16_to_1"
        )
        self.dual_state_input_features = n_input_features
        self.dual_state_hidden_size = dual_state_hidden_size
        self.fusion_temperature = float(fusion_temperature)
        self.score_mode = score_mode
        self.value_mode = value_mode


def build_deformable_patchcfc_filternfc_dualstatelstm_ablation(
    n_classes: int,
    *,
    n_channels: int = 128,
    n_times: int = 1153,
    **model_kwargs,
) -> deformable_patchCfC_filterNFC_dualStateLSTM_ablation:
    """Build a configurable dual-state component-ablation model."""
    return deformable_patchCfC_filterNFC_dualStateLSTM_ablation(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        **model_kwargs,
    )
