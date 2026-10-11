"""EEGNet with electrode-group-specific overlap-save ncps CfC modules.

The recurrent computation is provided directly by the vendored
``ncps.torch.CfC`` implementation. This file only adapts EEG tensors to the
public ncps sequence interface and restores the overlap-save time axis.
"""

from __future__ import annotations

import math
from typing import Optional

import ncps
import torch
from ncps.torch import CfC
from torch import Tensor, nn

from eeg_pipeline.models.causal1_group8PatchLSTM_EEGNet import (
    _positive_integer,
    causal1_group8PatchLSTM_EEGNet,
)
from eeg_pipeline.models.causal1_groupnorm_EEGNet import (
    _group_norm_from_batch_norm,
)


__all__ = ["egroupPatchCfC_EEGNet", "build_egrouppatchcfc_eegnet"]


class _ElectrodeGroupPatchNcpsCfCTemporal(nn.Module):
    """Adapt grouped EEG patches to independent ``ncps.torch.CfC`` modules."""

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
        if isinstance(backbone_dropout, bool):
            raise ValueError("cfc_backbone_dropout must be in [0, 1).")
        backbone_dropout = float(backbone_dropout)
        if not math.isfinite(backbone_dropout) or not 0.0 <= backbone_dropout < 1.0:
            raise ValueError("cfc_backbone_dropout must be finite and in [0, 1).")

        group_cfcs = []
        for _ in range(self.n_electrode_groups):
            cfc = CfC(
                input_size=1,
                units=self.n_features,
                proj_size=None,
                return_sequences=True,
                batch_first=True,
                mixed_memory=False,
                mode="default",
                activation="lecun_tanh",
                backbone_units=backbone_units,
                backbone_layers=backbone_layers,
                backbone_dropout=backbone_dropout,
            )
            if device is not None or dtype is not None:
                cfc = cfc.to(device=device, dtype=dtype)
            group_cfcs.append(cfc)
        self.group_cfcs = nn.ModuleList(group_cfcs)

        self.cfc_mode = "default"
        self.cfc_mixed_memory = False
        self.cfc_timespan = 1.0
        self.backbone_units = backbone_units
        self.backbone_layers = backbone_layers
        self.backbone_dropout = backbone_dropout

    def forward(self, input: Tensor) -> Tensor:
        if input.ndim != 4:
            raise ValueError(
                "group Patch-CfC expects a 4D tensor (B,1,C,T), "
                f"got shape {tuple(input.shape)}."
            )
        batch_size, input_features, n_chans, n_times = input.shape
        if input_features != 1:
            raise ValueError(
                "group Patch-CfC expects one input feature before temporal "
                f"encoding, got {input_features}."
            )
        if n_chans != self.n_chans:
            raise ValueError(
                f"expected {self.n_chans} EEG channels, got {n_chans}."
            )
        if n_times < self.patch_size:
            raise ValueError(
                f"input time length {n_times} is shorter than patch_size="
                f"{self.patch_size}."
            )

        n_patches = (
            (n_times - self.patch_size) // self.patch_stride + 1
        )
        effective_n_times = (
            self.patch_size + (n_patches - 1) * self.patch_stride
        )
        trimmed = input[:, 0, :, :effective_n_times]

        group_outputs = []
        for group_index, cfc in enumerate(self.group_cfcs):
            chan_start = group_index * self.chans_per_group
            chan_stop = chan_start + self.chans_per_group
            group = trimmed[:, chan_start:chan_stop, :]

            if self.patch_stride == self.patch_size:
                patches = group.reshape(
                    batch_size,
                    self.chans_per_group,
                    n_patches,
                    self.patch_size,
                )
            else:
                patches = group.unfold(
                    dimension=-1,
                    size=self.patch_size,
                    step=self.patch_stride,
                )
            sequences = patches.reshape(
                batch_size * self.chans_per_group * n_patches,
                self.patch_size,
                1,
            )

            # hx=None gives every flattened electrode-patch sequence an
            # independent zero state. timespans=None invokes ncps' documented
            # unit interval for regularly sampled sequences.
            recurrent_sequence, _ = cfc(sequences)
            recurrent_sequence = recurrent_sequence.reshape(
                batch_size,
                self.chans_per_group,
                n_patches,
                self.patch_size,
                self.n_features,
            )

            if n_patches == 1:
                group_output = recurrent_sequence[:, :, 0, :, :]
            elif self.patch_stride == self.patch_size:
                group_output = recurrent_sequence.reshape(
                    batch_size,
                    self.chans_per_group,
                    effective_n_times,
                    self.n_features,
                )
            else:
                first_patch = recurrent_sequence[:, :, 0, :, :]
                later_novel = recurrent_sequence[
                    :, :, 1:, self.patch_size - self.patch_stride :, :
                ].reshape(
                    batch_size,
                    self.chans_per_group,
                    (n_patches - 1) * self.patch_stride,
                    self.n_features,
                )
                group_output = torch.cat(
                    (first_patch, later_novel), dim=2
                )
            group_outputs.append(group_output)

        output = torch.cat(group_outputs, dim=1)
        return output.permute(0, 3, 1, 2).contiguous()


class egroupPatchCfC_EEGNet(causal1_group8PatchLSTM_EEGNet):
    """Use official ncps full gated CfCs with selectable normalization."""

    def __init__(
        self,
        *args,
        cfc_hidden_size: Optional[int] = None,
        cfc_backbone_units: int = 8,
        cfc_backbone_layers: int = 1,
        cfc_backbone_dropout: float = 0.0,
        normalization: str = "batchnorm",
        n_norm_groups: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        if cfc_hidden_size is None:
            cfc_hidden_size = self.F1
        cfc_hidden_size = _positive_integer(
            "cfc_hidden_size", cfc_hidden_size
        )
        if cfc_hidden_size != self.F1:
            raise ValueError(
                "cfc_hidden_size must equal F1 so all later EEGNet layers "
                f"remain unchanged; got {cfc_hidden_size} and F1={self.F1}."
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
        self.cfc_backbone_units = cfc_backbone_units
        self.cfc_backbone_layers = cfc_backbone_layers
        self.cfc_backbone_dropout = cfc_backbone_dropout
        self.cfc_mode = "default"
        self.cfc_mixed_memory = False
        self.cfc_timespan = 1.0
        self.ncps_version = ncps.__version__

        if not isinstance(normalization, str):
            raise TypeError("normalization must be 'batchnorm' or 'groupnorm'.")
        normalization = normalization.lower()
        if normalization not in {"batchnorm", "groupnorm"}:
            raise ValueError(
                "normalization must be 'batchnorm' or 'groupnorm', "
                f"got {normalization!r}."
            )

        if normalization == "groupnorm":
            n_norm_groups = _positive_integer(
                "n_norm_groups", n_norm_groups
            )
            batch_norms = (
                ("bnorm_temporal", self.bnorm_temporal),
                ("bnorm_1", self.bnorm_1),
                ("bnorm_2", self.bnorm_2),
            )
            invalid = [
                f"{name}={module.num_features}"
                for name, module in batch_norms
                if module.num_features % n_norm_groups != 0
            ]
            if invalid:
                raise ValueError(
                    "n_norm_groups must divide every normalized feature "
                    f"count; got n_norm_groups={n_norm_groups} and "
                    f"{', '.join(invalid)}."
                )
            self.bnorm_temporal = _group_norm_from_batch_norm(
                "bnorm_temporal", self.bnorm_temporal, n_norm_groups
            )
            self.bnorm_1 = _group_norm_from_batch_norm(
                "bnorm_1", self.bnorm_1, n_norm_groups
            )
            self.bnorm_2 = _group_norm_from_batch_norm(
                "bnorm_2", self.bnorm_2, n_norm_groups
            )
            self.n_norm_groups: Optional[int] = n_norm_groups
        else:
            self.n_norm_groups = None
        self.normalization = normalization


def build_egrouppatchcfc_eegnet(
    n_classes: int,
    *,
    n_channels: int = 128,
    n_times: int = 1153,
    **model_kwargs,
) -> egroupPatchCfC_EEGNet:
    """Build the electrode-group overlap-save ncps CfC EEGNet."""
    return egroupPatchCfC_EEGNet(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        **model_kwargs,
    )


if __name__ == "__main__":
    model = build_egrouppatchcfc_eegnet(
        n_classes=4,
        n_channels=8,
        n_times=129,
        patch_size=32,
        patch_stride=16,
        n_electrode_groups=2,
        normalization="groupnorm",
        n_norm_groups=4,
    )
    eeg = torch.randn(2, 8, 129, requires_grad=True)
    captured = []
    handle = model.conv_temporal.register_forward_hook(
        lambda _module, _inputs, output: captured.append(output)
    )
    logits = model(eeg)
    handle.remove()
    if captured[0].shape != (2, 8, 8, 128):
        raise RuntimeError(f"Unexpected CfC output shape: {captured[0].shape}")
    if logits.shape != (2, 4):
        raise RuntimeError(f"Unexpected logits shape: {logits.shape}")
    if any(isinstance(module, nn.LSTM) for module in model.modules()):
        raise RuntimeError("The active ncps CfC model contains an LSTM")
    if any(isinstance(module, nn.BatchNorm2d) for module in model.modules()):
        raise RuntimeError("GroupNorm CfC model still contains BatchNorm2d")
    if sum(isinstance(module, nn.GroupNorm) for module in model.modules()) != 3:
        raise RuntimeError("GroupNorm CfC model must contain three GroupNorms")
    if sum(isinstance(module, CfC) for module in model.modules()) != 2:
        raise RuntimeError("Expected one ncps CfC module per electrode group")
    logits.sum().backward()
    print("egroupPatchCfC_EEGNet ncps self-test passed")
