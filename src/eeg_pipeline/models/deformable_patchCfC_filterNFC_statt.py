"""FilterNFC backbone with feature-wise time-space MSDeformAttn."""

from __future__ import annotations

from eeg_pipeline.models._deformable_patchCfC_filterNFC_msdeform import (
    _DeformablePatchCfCFilterNFCMSDeform,
)


__all__ = [
    "deformable_patchCfC_filterNFC_statt",
    "build_deformable_patchcfc_filternfc_statt",
]


class deformable_patchCfC_filterNFC_statt(
    _DeformablePatchCfCFilterNFCMSDeform
):
    """Attend sparsely over the full Patch-time by electrode plane."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, attention_mode="spacetime_2d", **kwargs)


def build_deformable_patchcfc_filternfc_statt(
    n_classes: int,
    *,
    n_channels: int = 128,
    n_times: int = 1153,
    **model_kwargs,
) -> deformable_patchCfC_filterNFC_statt:
    return deformable_patchCfC_filterNFC_statt(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        **model_kwargs,
    )
