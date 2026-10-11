"""Group4 PatchLSTM/DeformableLSTM with time-electrode sparse attention."""

from __future__ import annotations

from eeg_pipeline.models._group4_patchlstm_deformablelstm_msdeform import (
    _Group4PatchLSTMDeformableLSTMMSDeform,
)


__all__ = [
    "group4_patchLstm_deformableLstm_statt",
    "build_group4_patchlstm_deformablelstm_statt",
]


class group4_patchLstm_deformableLstm_statt(
    _Group4PatchLSTMDeformableLSTMMSDeform
):
    """Apply StAtt over the time-electrode plane before spatial filtering."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, attention_mode="spacetime_2d", **kwargs)


def build_group4_patchlstm_deformablelstm_statt(
    n_classes: int,
    *,
    n_channels: int = 128,
    n_times: int = 1153,
    **model_kwargs,
) -> group4_patchLstm_deformableLstm_statt:
    return group4_patchLstm_deformableLstm_statt(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        **model_kwargs,
    )

