"""FilterNFC backbone with feature-wise one-dimensional spatial MSDeformAttn."""

from __future__ import annotations

from eeg_pipeline.models._deformable_patchCfC_filterNFC_msdeform import (
    _DeformablePatchCfCFilterNFCMSDeform,
)


__all__ = [
    "deformable_patchCfC_filterNFC_satt",
    "build_deformable_patchcfc_filternfc_satt",
]


class deformable_patchCfC_filterNFC_satt(
    _DeformablePatchCfCFilterNFCMSDeform
):
    """Attend sparsely over 128 electrodes at each time independently."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, attention_mode="space_1d", **kwargs)


def build_deformable_patchcfc_filternfc_satt(
    n_classes: int,
    *,
    n_channels: int = 128,
    n_times: int = 1153,
    **model_kwargs,
) -> deformable_patchCfC_filterNFC_satt:
    return deformable_patchCfC_filterNFC_satt(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        **model_kwargs,
    )
