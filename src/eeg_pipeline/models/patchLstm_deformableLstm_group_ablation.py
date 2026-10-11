"""Controlled electrode-group ablation for PatchLSTM/DeformableLSTM."""

from __future__ import annotations

from collections import OrderedDict

from torch import nn

from eeg_pipeline.models.causal1_group8PatchLSTM_EEGNet import (
    _positive_integer,
    causal1_group8PatchLSTM_EEGNet,
)
from eeg_pipeline.models.group4_patchLstm_deformableLstm import (
    _FeatureMapTemporalLSTMOffsetWarp,
)


__all__ = [
    "patchLstm_deformableLstm_group_ablation",
    "build_group1_patchlstm_deformablelstm",
    "build_group2_patchlstm_deformablelstm",
    "build_group8_patchlstm_deformablelstm",
    "build_group16_patchlstm_deformablelstm",
]


ALLOWED_ELECTRODE_GROUPS = frozenset({1, 2, 4, 8, 16})


class patchLstm_deformableLstm_group_ablation(
    causal1_group8PatchLSTM_EEGNet
):
    """Keep one topology while varying only PatchLSTM parameter sharing."""

    def __init__(
        self,
        *args,
        n_electrode_groups: int,
        offset_lstm_hidden_size: int = 8,
        offset_lstm_layers: int = 1,
        offset_lstm_dropout: float = 0.0,
        **kwargs,
    ) -> None:
        n_electrode_groups = _positive_integer(
            "n_electrode_groups", n_electrode_groups
        )
        if n_electrode_groups not in ALLOWED_ELECTRODE_GROUPS:
            raise ValueError(
                "n_electrode_groups must be one of "
                f"{sorted(ALLOWED_ELECTRODE_GROUPS)}, got "
                f"{n_electrode_groups}."
            )
        super().__init__(
            *args,
            n_electrode_groups=n_electrode_groups,
            allow_uneven_groups=False,
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


def _build_expected_group(
    expected_groups: int,
    n_classes: int,
    *,
    n_channels: int,
    n_times: int,
    **model_kwargs,
) -> patchLstm_deformableLstm_group_ablation:
    configured_groups = model_kwargs.get("n_electrode_groups")
    if configured_groups != expected_groups:
        raise ValueError(
            f"this builder requires n_electrode_groups={expected_groups}, "
            f"got {configured_groups!r}."
        )
    return patchLstm_deformableLstm_group_ablation(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        **model_kwargs,
    )


def build_group1_patchlstm_deformablelstm(
    n_classes: int,
    *,
    n_channels: int = 128,
    n_times: int = 1153,
    **model_kwargs,
) -> patchLstm_deformableLstm_group_ablation:
    return _build_expected_group(
        1, n_classes, n_channels=n_channels, n_times=n_times, **model_kwargs
    )


def build_group2_patchlstm_deformablelstm(
    n_classes: int,
    *,
    n_channels: int = 128,
    n_times: int = 1153,
    **model_kwargs,
) -> patchLstm_deformableLstm_group_ablation:
    return _build_expected_group(
        2, n_classes, n_channels=n_channels, n_times=n_times, **model_kwargs
    )


def build_group8_patchlstm_deformablelstm(
    n_classes: int,
    *,
    n_channels: int = 128,
    n_times: int = 1153,
    **model_kwargs,
) -> patchLstm_deformableLstm_group_ablation:
    return _build_expected_group(
        8, n_classes, n_channels=n_channels, n_times=n_times, **model_kwargs
    )


def build_group16_patchlstm_deformablelstm(
    n_classes: int,
    *,
    n_channels: int = 128,
    n_times: int = 1153,
    **model_kwargs,
) -> patchLstm_deformableLstm_group_ablation:
    return _build_expected_group(
        16, n_classes, n_channels=n_channels, n_times=n_times, **model_kwargs
    )

