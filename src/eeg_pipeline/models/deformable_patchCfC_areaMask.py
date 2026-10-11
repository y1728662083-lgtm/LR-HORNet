"""Deformable Patch-CfC with shared CfC1 and regional importance masks."""

from __future__ import annotations

import torch
from ncps.torch import CfC
from torch import Tensor, nn

from eeg_pipeline.models.causal1_group8PatchLSTM_EEGNet import (
    _positive_integer,
)
from eeg_pipeline.models.deformable_patchCfC import (
    _dropout_probability,
    deformable_patchCfC,
)


__all__ = [
    "deformable_patchCfC_areaMask",
    "build_deformable_patchcfc_areamask",
]


class _AreaMaskedSharedPatchLastCfCTemporal(nn.Module):
    """Gate anatomical groups, then apply one shared CfC to all patches."""

    def __init__(
        self,
        *,
        n_chans: int,
        n_features: int,
        patch_size: int,
        patch_stride: int,
        n_electrode_groups: int,
        backbone_units: int,
        backbone_layers: int,
        backbone_dropout: float,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.n_chans = _positive_integer("n_chans", n_chans)
        self.n_features = _positive_integer("n_features", n_features)
        self.patch_size = _positive_integer("patch_size", patch_size)
        self.patch_stride = _positive_integer("patch_stride", patch_stride)
        if self.patch_stride > self.patch_size:
            raise ValueError("patch_stride must not exceed patch_size.")

        self.n_electrode_groups = _positive_integer(
            "n_electrode_groups", n_electrode_groups
        )
        if self.n_chans % self.n_electrode_groups != 0:
            raise ValueError(
                "n_chans must be divisible by n_electrode_groups; "
                f"got n_chans={self.n_chans} and "
                f"n_electrode_groups={self.n_electrode_groups}."
            )
        self.chans_per_group = self.n_chans // self.n_electrode_groups

        backbone_units = _positive_integer(
            "cfc_backbone_units", backbone_units
        )
        backbone_layers = _positive_integer(
            "cfc_backbone_layers", backbone_layers
        )
        backbone_dropout = _dropout_probability(
            "cfc_backbone_dropout", backbone_dropout
        )

        self.cfc = CfC(
            input_size=1,
            units=self.n_features,
            proj_size=None,
            return_sequences=False,
            batch_first=True,
            mixed_memory=False,
            mode="default",
            activation="lecun_tanh",
            backbone_units=backbone_units,
            backbone_layers=backbone_layers,
            backbone_dropout=backbone_dropout,
        )
        if device is not None or dtype is not None:
            self.cfc = self.cfc.to(device=device, dtype=dtype)

        self.area_mask_logits = nn.Parameter(
            torch.zeros(
                self.n_electrode_groups,
                device=device,
                dtype=dtype,
            )
        )
        electrode_area_index = torch.arange(
            self.n_chans,
            device=device,
            dtype=torch.long,
        ) // self.chans_per_group
        self.register_buffer(
            "electrode_area_index",
            electrode_area_index,
            persistent=True,
        )

        self.backbone_units = backbone_units
        self.backbone_layers = backbone_layers
        self.backbone_dropout = backbone_dropout

    def area_importance_weights(self) -> Tensor:
        """Return bounded positive group weights; zero logits give identity."""
        return 1.0 + torch.tanh(self.area_mask_logits)

    def electrode_importance_weights(self) -> Tensor:
        """Expand each group weight over its contiguous reordered electrodes."""
        return self.area_importance_weights().index_select(
            0, self.electrode_area_index
        )

    def forward(self, input: Tensor) -> Tensor:
        if input.ndim != 4:
            raise ValueError(
                "area-masked Patch-CfC expects (B,1,C,T), "
                f"got shape {tuple(input.shape)}."
            )
        batch_size, input_features, n_chans, n_times = input.shape
        if input_features != 1:
            raise ValueError(
                "area-masked Patch-CfC expects one scalar feature per "
                f"electrode, got {input_features}."
            )
        if n_chans != self.n_chans:
            raise ValueError(
                f"expected {self.n_chans} EEG channels, got {n_chans}."
            )
        if n_times < self.patch_size:
            raise ValueError(
                f"input time length {n_times} is shorter than "
                f"patch_size={self.patch_size}."
            )

        n_patches = (
            (n_times - self.patch_size) // self.patch_stride + 1
        )
        effective_n_times = (
            self.patch_size + (n_patches - 1) * self.patch_stride
        )
        signal = input[:, 0, :, :effective_n_times]
        electrode_weights = self.electrode_importance_weights().view(
            1, self.n_chans, 1
        )
        gated_signal = signal * electrode_weights

        patches = gated_signal.unfold(
            dimension=-1,
            size=self.patch_size,
            step=self.patch_stride,
        )
        sequences = patches.reshape(
            batch_size * self.n_chans * n_patches,
            self.patch_size,
            1,
        )

        # Every electrode-patch sequence starts from its own zero state. Only
        # the CfC parameters are shared across regions and electrodes.
        final_outputs, _ = self.cfc(
            sequences,
            hx=None,
            timespans=None,
        )
        output = final_outputs.reshape(
            batch_size,
            self.n_chans,
            n_patches,
            self.n_features,
        )
        return output.permute(0, 3, 1, 2).contiguous()


class deformable_patchCfC_areaMask(deformable_patchCfC):
    """Use one shared CfC1 after bounded, region-level input gating."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        previous_temporal = self.conv_temporal
        reference_parameter = next(previous_temporal.parameters())
        self.conv_temporal = _AreaMaskedSharedPatchLastCfCTemporal(
            n_chans=self.n_chans,
            n_features=self.F1,
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
            n_electrode_groups=self.n_electrode_groups,
            backbone_units=previous_temporal.backbone_units,
            backbone_layers=previous_temporal.backbone_layers,
            backbone_dropout=previous_temporal.backbone_dropout,
            device=reference_parameter.device,
            dtype=reference_parameter.dtype,
        )
        self.area_mask_transform = "1+tanh"


def build_deformable_patchcfc_areamask(
    n_classes: int,
    *,
    n_channels: int = 128,
    n_times: int = 1153,
    **model_kwargs,
) -> deformable_patchCfC_areaMask:
    """Build the shared-CfC1 model with regional importance masks."""
    return deformable_patchCfC_areaMask(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        **model_kwargs,
    )
