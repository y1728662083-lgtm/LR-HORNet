"""Group4 PatchLSTM/DeformableLSTM with electrode-only sparse attention."""

from __future__ import annotations

from eeg_pipeline.models._group4_patchlstm_deformablelstm_msdeform import (
    _Group4PatchLSTMDeformableLSTMMSDeform,
)


__all__ = [
    "group4_patchLstm_deformableLstm_satt",
    "build_group4_patchlstm_deformablelstm_satt",
]


class group4_patchLstm_deformableLstm_satt(
    _Group4PatchLSTMDeformableLSTMMSDeform
):
    """Apply SAtt independently at each time point before spatial filtering."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, attention_mode="space_1d", **kwargs)


def build_group4_patchlstm_deformablelstm_satt(
    n_classes: int,
    *,
    n_channels: int = 128,
    n_times: int = 1153,
    **model_kwargs,
) -> group4_patchLstm_deformableLstm_satt:
    return group4_patchLstm_deformableLstm_satt(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        **model_kwargs,
    )

