"""Four-group full-sequence Patch-CfC with CfC temporal deformation.

This is the cell-type counterpart of ``group4_patchLstm_deformableCfC``.
Only the patch encoder changes from LSTM to the official ncps CfC.  Patch
geometry, overlap-save reconstruction, EEGNet, and the post-spatial CfC warp
remain unchanged.
"""

from __future__ import annotations

from typing import Optional

import ncps

from eeg_pipeline.models.causal1_group8PatchLSTM_EEGNet import (
    _positive_integer,
)
from eeg_pipeline.models.egroupPatchCfC_EEGNet import (
    _ElectrodeGroupPatchNcpsCfCTemporal,
)
from eeg_pipeline.models.group4_patchLstm_deformableCfC import (
    group4_patchLstm_deformableCfC,
)


__all__ = [
    "group4_patchCfC_deformableCfC",
    "build_group4_patchcfc_deformablecfc",
]


class group4_patchCfC_deformableCfC(
    group4_patchLstm_deformableCfC
):
    """Replace the four-group PatchLSTM with full-sequence PatchCfC."""

    def __init__(
        self,
        *args,
        cfc_hidden_size: Optional[int] = None,
        cfc_backbone_units: int = 8,
        cfc_backbone_layers: int = 1,
        cfc_backbone_dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        if self.n_electrode_groups != 4:
            raise RuntimeError(
                "group4_patchCfC_deformableCfC requires exactly four groups"
            )
        if cfc_hidden_size is None:
            cfc_hidden_size = self.F1
        cfc_hidden_size = _positive_integer(
            "cfc_hidden_size", cfc_hidden_size
        )
        if cfc_hidden_size != self.F1:
            raise ValueError(
                "cfc_hidden_size must equal F1 so the unchanged EEGNet "
                f"receives {self.F1} features; got {cfc_hidden_size}."
            )

        reference_parameter = next(self.conv_temporal.parameters())
        self.conv_temporal = _ElectrodeGroupPatchNcpsCfCTemporal(
            n_chans=self.n_chans,
            n_features=cfc_hidden_size,
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
            n_electrode_groups=self.n_electrode_groups,
            backbone_units=cfc_backbone_units,
            backbone_layers=cfc_backbone_layers,
            backbone_dropout=cfc_backbone_dropout,
            device=reference_parameter.device,
            dtype=reference_parameter.dtype,
        )

        self.cfc_hidden_size = cfc_hidden_size
        self.cfc_backbone_units = self.conv_temporal.backbone_units
        self.cfc_backbone_layers = self.conv_temporal.backbone_layers
        self.cfc_backbone_dropout = self.conv_temporal.backbone_dropout
        self.cfc_mode = self.conv_temporal.cfc_mode
        self.cfc_mixed_memory = self.conv_temporal.cfc_mixed_memory
        self.cfc_timespan = self.conv_temporal.cfc_timespan
        self.patch_recurrent_cell = "ncps.CfC"
        self.patch_output_policy = "full_sequence_overlap_save"
        self.ncps_version = ncps.__version__


def build_group4_patchcfc_deformablecfc(
    n_classes: int,
    *,
    n_channels: int = 128,
    n_times: int = 1153,
    **model_kwargs,
) -> group4_patchCfC_deformableCfC:
    return group4_patchCfC_deformableCfC(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        **model_kwargs,
    )

