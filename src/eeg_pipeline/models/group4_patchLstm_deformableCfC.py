"""Four-group Patch-LSTM EEGNet with CfC-driven temporal deformation.

This controlled counterpart of deformable_patchCfC reuses the same shared
CfC offset generator, gated displacement, and bilinear temporal resampling.
The original first pooling layer remains after the warp, so the Patch-LSTM
feature length and classifier are unchanged.
"""

from __future__ import annotations

from collections import OrderedDict

import ncps
import torch
from torch import nn

from eeg_pipeline.models.causal1_group8PatchLSTM_EEGNet import (
    _positive_integer,
    causal1_group8PatchLSTM_EEGNet,
)
from eeg_pipeline.models.deformable_patchCfC import (
    _GlobalSpatialCfCOffsetWarp,
)


__all__ = [
    "group4_patchLstm_deformableCfC",
    "build_group4_patchlstm_deformablecfc",
]


class group4_patchLstm_deformableCfC(
    causal1_group8PatchLSTM_EEGNet
):
    """Add CfC-driven temporal deformation to the four-group Patch-LSTM."""

    def __init__(
        self,
        *args,
        n_electrode_groups: int = 4,
        offset_cfc_hidden_size: int = 8,
        offset_cfc_backbone_units: int = 8,
        offset_cfc_backbone_layers: int = 1,
        offset_cfc_backbone_dropout: float = 0.0,
        **kwargs,
    ) -> None:
        n_electrode_groups = _positive_integer(
            "n_electrode_groups", n_electrode_groups
        )
        if n_electrode_groups != 4:
            raise ValueError(
                "group4_patchLstm_deformableCfC requires "
                "n_electrode_groups=4."
            )
        super().__init__(
            *args,
            n_electrode_groups=n_electrode_groups,
            **kwargs,
        )

        reference_parameter = next(self.conv_temporal.parameters())
        deformation = _GlobalSpatialCfCOffsetWarp(
            n_feature_maps=self.F1 * self.D,
            hidden_size=offset_cfc_hidden_size,
            backbone_units=offset_cfc_backbone_units,
            backbone_layers=offset_cfc_backbone_layers,
            backbone_dropout=offset_cfc_backbone_dropout,
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

        self.offset_cfc_hidden_size = deformation.hidden_size
        self.offset_cfc_backbone_units = deformation.backbone_units
        self.offset_cfc_backbone_layers = deformation.backbone_layers
        self.offset_cfc_backbone_dropout = deformation.backbone_dropout
        self.offset_padding_mode = deformation.padding_mode
        self.offset_gate = "sigmoid"
        self.deformation_axis = "time"
        self.deformation_placement = "after_spatial_before_pool1"
        self.ncps_version = ncps.__version__


def build_group4_patchlstm_deformablecfc(
    n_classes: int,
    *,
    n_channels: int = 128,
    n_times: int = 1153,
    **model_kwargs,
) -> group4_patchLstm_deformableCfC:
    """Build the four-group Patch-LSTM with CfC temporal offsets."""
    return group4_patchLstm_deformableCfC(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        **model_kwargs,
    )


if __name__ == "__main__":
    model = build_group4_patchlstm_deformablecfc(
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
            "zero-initialized CfC deformation exceeded the numerical "
            f"identity tolerance: max_abs_error={max_identity_error:.3e}"
        )
    logits = model(eeg)
    if logits.shape != (2, 4):
        raise RuntimeError(f"unexpected output shape: {tuple(logits.shape)}")
    logits.sum().backward()
    print(
        "group4_patchLstm_deformableCfC self-test passed: "
        f"max_identity_error={max_identity_error:.3e}"
    )
