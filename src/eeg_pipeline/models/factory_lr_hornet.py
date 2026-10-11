"""Minimal factory for the archived LR-HORNet experiment only."""

from __future__ import annotations

from torch import nn

from eeg_pipeline.models.group4_patchLstm_deformableLstm import (
    build_group4_patchlstm_deformablelstm,
)


MODEL_NAME = "group4_patchLstm_deformableLstm"


def build_model(name: str, n_classes: int, **model_kwargs) -> nn.Module:
    if name != MODEL_NAME:
        raise ValueError(
            f"This archive contains only {MODEL_NAME!r}; got {name!r}."
        )
    return build_group4_patchlstm_deformablelstm(
        n_classes=n_classes,
        **model_kwargs,
    )
