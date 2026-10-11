"""Four-group Patch-LSTM EEGNet with LSTM-driven temporal deformation.

The Patch-LSTM temporal encoder and the remaining EEGNet backbone are kept
unchanged. Immediately after the spatial block, a globally shared LSTM
predicts one gated temporal displacement for every feature map and time point.
The warped sequence is passed to the original first pooling layer, so this is
a controlled deformation add-on rather than a classifier-size change.
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
from eeg_pipeline.models.deformable_patchCfC import _dropout_probability


__all__ = [
    "group4_patchLstm_deformableLstm",
    "build_group4_patchlstm_deformablelstm",
]


class _FeatureMapTemporalLSTMOffsetWarp(nn.Module):
    """Predict feature-map-specific temporal offsets with a shared LSTM."""

    def __init__(
        self,
        *,
        n_feature_maps: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.n_feature_maps = _positive_integer(
            "n_feature_maps", n_feature_maps
        )
        self.hidden_size = _positive_integer(
            "offset_lstm_hidden_size", hidden_size
        )
        self.num_layers = _positive_integer(
            "offset_lstm_layers", num_layers
        )
        self.configured_dropout = _dropout_probability(
            "offset_lstm_dropout", dropout
        )
        recurrent_dropout = (
            self.configured_dropout if self.num_layers > 1 else 0.0
        )
        factory_kwargs = {"device": device, "dtype": dtype}
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
            bidirectional=False,
            **factory_kwargs,
        )
        self.offset_head = nn.Linear(
            self.hidden_size, 1, **factory_kwargs
        )
        self.gate_head = nn.Linear(
            self.hidden_size, 1, **factory_kwargs
        )

        # Start from the exact identity grid, matching deformable_patchCfC.
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)
        self.padding_mode = "border"
        self.align_corners = True

    def forward(self, input: Tensor) -> Tensor:
        if input.ndim != 4:
            raise ValueError(
                "temporal LSTM deformation expects (B,M,1,T), "
                f"got shape {tuple(input.shape)}."
            )
        batch_size, n_feature_maps, spatial_size, n_times = input.shape
        if n_feature_maps != self.n_feature_maps or spatial_size != 1:
            raise ValueError(
                f"expected (B,{self.n_feature_maps},1,T), got "
                f"{tuple(input.shape)}."
            )

        sequences = input[:, :, 0, :].reshape(
            batch_size * n_feature_maps,
            n_times,
            1,
        )
        hidden_sequence, _ = self.lstm(sequences)
        raw_offset = self.offset_head(hidden_sequence).squeeze(-1)
        gate = torch.sigmoid(self.gate_head(hidden_sequence)).squeeze(-1)
        offset = gate * raw_offset

        signal = input.reshape(batch_size * n_feature_maps, 1, 1, n_times)
        base_time = torch.arange(
            n_times,
            device=input.device,
            dtype=input.dtype,
        ).unsqueeze(0)
        sample_time = base_time + offset
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
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )
        return warped.reshape(batch_size, n_feature_maps, 1, n_times)


class group4_patchLstm_deformableLstm(
    causal1_group8PatchLSTM_EEGNet
):
    """Add LSTM-driven temporal deformation to a four-group Patch-LSTM.

    Channels are divided into four balanced contiguous groups. When the
    channel count is not divisible by four, the earlier groups receive one
    extra channel (for example, 62 channels become 16/16/15/15).
    """

    def __init__(
        self,
        *args,
        n_electrode_groups: int = 4,
        offset_lstm_hidden_size: int = 8,
        offset_lstm_layers: int = 1,
        offset_lstm_dropout: float = 0.0,
        **kwargs,
    ) -> None:
        n_electrode_groups = _positive_integer(
            "n_electrode_groups", n_electrode_groups
        )
        if n_electrode_groups != 4:
            raise ValueError(
                "group4_patchLstm_deformableLstm requires "
                "n_electrode_groups=4."
            )
        super().__init__(
            *args,
            n_electrode_groups=n_electrode_groups,
            allow_uneven_groups=True,
            **kwargs,
        )

        reference_parameter = next(self.conv_temporal.parameters())
        deformation = _FeatureMapTemporalLSTMOffsetWarp(
            n_feature_maps=self.F1 * self.D,
            hidden_size=offset_lstm_hidden_size,
            num_layers=offset_lstm_layers,
            dropout=offset_lstm_dropout,
            device=reference_parameter.device,
            dtype=reference_parameter.dtype,
        )
        original_pool = self.pool_1
        self.pool_1 = nn.Sequential(
            OrderedDict(
                (
                    ("deformation", deformation),
                    ("downsample", original_pool),
                )
            )
        )

        self.offset_lstm_hidden_size = deformation.hidden_size
        self.offset_lstm_layers = deformation.num_layers
        self.offset_lstm_dropout = deformation.configured_dropout
        self.offset_padding_mode = deformation.padding_mode
        self.offset_gate = "sigmoid"
        self.deformation_axis = "time"
        self.deformation_placement = "after_spatial_before_pool1"


def build_group4_patchlstm_deformablelstm(
    n_classes: int,
    *,
    n_channels: int = 128,
    n_times: int = 1153,
    **model_kwargs,
) -> group4_patchLstm_deformableLstm:
    """Build the four-group Patch-LSTM with LSTM temporal offsets."""
    return group4_patchLstm_deformableLstm(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        **model_kwargs,
    )


if __name__ == "__main__":
    model = build_group4_patchlstm_deformablelstm(
        n_classes=4,
        n_channels=8,
        n_times=129,
        patch_size=64,
        patch_stride=32,
        F1=4,
        D=1,
        F2=4,
    )
    eeg = torch.randn(2, 8, 129, requires_grad=True)
    deformation_input = torch.randn(2, 4, 1, 128)
    with torch.no_grad():
        identity_output = model.pool_1.deformation(deformation_input)
    max_identity_error = (
        identity_output - deformation_input
    ).abs().max().item()
    if max_identity_error > 1e-4:
        raise RuntimeError(
            "zero-initialized LSTM deformation exceeded the numerical "
            f"identity tolerance: max_abs_error={max_identity_error:.3e}"
        )
    logits = model(eeg)
    if logits.shape != (2, 4):
        raise RuntimeError(f"unexpected output shape: {tuple(logits.shape)}")
    logits.sum().backward()
    print(
        "group4_patchLstm_deformableLstm self-test passed: "
        f"max_identity_error={max_identity_error:.3e}"
    )
