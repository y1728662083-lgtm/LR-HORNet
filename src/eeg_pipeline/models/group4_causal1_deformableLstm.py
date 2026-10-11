"""Four-group causal EEGNet with LSTM-driven temporal deformation.

This control keeps the electrode grouping and group-shared causal temporal
convolution from causal1_EEGNet. It adds only the same Deformable-LSTM module
used by the Full model after the spatial convolution and before pool_1.
"""

from __future__ import annotations

from collections import OrderedDict

from torch import nn

from eeg_pipeline.models.causal1_EEGNet import causal1_EEGNet
from eeg_pipeline.models.causal1_group8PatchLSTM_EEGNet import (
    _positive_integer,
)
from eeg_pipeline.models.group4_patchLstm_deformableLstm import (
    _FeatureMapTemporalLSTMOffsetWarp,
)


__all__ = [
    "group4_causal1_deformableLstm",
    "build_group4_causal1_deformablelstm",
]


class group4_causal1_deformableLstm(causal1_EEGNet):
    """Add LSTM offsets to four-group causal EEGNet without Patch-LSTM."""

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
                "group4_causal1_deformableLstm requires "
                "n_electrode_groups=4."
            )
        super().__init__(
            *args,
            n_electrode_groups=n_electrode_groups,
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
        self.offset_predictor = "shared_feature_map_lstm"
        self.deformation_axis = "time"
        self.deformation_placement = "after_spatial_before_pool1"


def build_group4_causal1_deformablelstm(
    n_classes: int,
    *,
    n_channels: int = 128,
    n_times: int = 1153,
    **model_kwargs,
) -> group4_causal1_deformableLstm:
    return group4_causal1_deformableLstm(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        **model_kwargs,
    )