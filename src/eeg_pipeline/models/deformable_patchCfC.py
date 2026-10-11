"""EEGNet with group Patch-CfC encoding and gated temporal deformation.

The first temporal convolution is replaced by electrode-group-specific
``ncps.torch.CfC`` modules. Each temporal patch contributes only its final
CfC output. After EEGNet's spatial convolution, one globally shared CfC
predicts a gated, feature-map-specific temporal displacement. The displaced
spatial features are then processed by the unchanged separable temporal block.
"""

from __future__ import annotations

import math
from typing import Optional

import ncps
import torch
from braindecode.functional import glorot_weight_zero_bias
from braindecode.modules import LinearWithConstraint
from ncps.torch import CfC
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from eeg_pipeline.models.causal1_group8PatchLSTM_EEGNet import (
    _positive_integer,
)
from eeg_pipeline.models.egroupPatchCfC_EEGNet import egroupPatchCfC_EEGNet


__all__ = ["deformable_patchCfC", "build_deformable_patchcfc"]


def _dropout_probability(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and in [0, 1).")
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed < 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1).")
    return parsed


class _ElectrodeGroupPatchLastCfCTemporal(nn.Module):
    """Return only the final CfC state from every grouped temporal patch."""

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
        activation_checkpoint: bool = False,
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
        if not isinstance(activation_checkpoint, bool):
            raise ValueError("activation_checkpoint must be a boolean")

        group_cfcs = []
        for _ in range(self.n_electrode_groups):
            cfc = CfC(
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
                cfc = cfc.to(device=device, dtype=dtype)
            group_cfcs.append(cfc)
        self.group_cfcs = nn.ModuleList(group_cfcs)

        self.backbone_units = backbone_units
        self.backbone_layers = backbone_layers
        self.backbone_dropout = backbone_dropout
        self.activation_checkpoint = activation_checkpoint

    def _prepare_input(self, input: Tensor) -> tuple[int, int, Tensor]:
        """Validate an EEG tensor and return its trimmed patching view."""
        if input.ndim != 4:
            raise ValueError(
                "group Patch-CfC expects (B,1,C,T), "
                f"got shape {tuple(input.shape)}."
            )
        batch_size, input_features, n_chans, n_times = input.shape
        if input_features != 1:
            raise ValueError(
                "group Patch-CfC expects one scalar feature per electrode, "
                f"got {input_features}."
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
        return batch_size, n_patches, input[:, 0, :, :effective_n_times]

    def _run_cfc(self, cfc: CfC, sequences: Tensor) -> Tensor:
        def final_state(current_sequences: Tensor) -> Tensor:
            outputs, _ = cfc(
                current_sequences,
                hx=None,
                timespans=None,
            )
            return outputs

        if (
            self.activation_checkpoint
            and self.training
            and torch.is_grad_enabled()
        ):
            return checkpoint(
                final_state,
                sequences,
                use_reentrant=False,
            )
        return final_state(sequences)

    def forward(self, input: Tensor) -> Tensor:
        batch_size, n_patches, trimmed = self._prepare_input(input)

        group_outputs = []
        for group_index, cfc in enumerate(self.group_cfcs):
            chan_start = group_index * self.chans_per_group
            chan_stop = chan_start + self.chans_per_group
            group = trimmed[:, chan_start:chan_stop, :]
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

            # hx=None gives every electrode-patch sequence an independent
            # zero state. return_sequences=False returns only its final output.
            final_outputs = self._run_cfc(cfc, sequences)
            group_output = final_outputs.reshape(
                batch_size,
                self.chans_per_group,
                n_patches,
                self.n_features,
            )
            group_outputs.append(group_output)

        output = torch.cat(group_outputs, dim=1)
        return output.permute(0, 3, 1, 2).contiguous()

    def forward_selected(
        self,
        input: Tensor,
        position_indices: Tensor,
    ) -> Tensor:
        """Encode only selected flattened (electrode, patch) positions.

        position_indices uses the same channel-major flattening as the
        structured TF-C loss: position = channel * n_patches + patch.
        The returned shape is (B, F, S) and preserves the supplied order,
        including when positions are interleaved across electrode groups.
        """
        batch_size, n_patches, trimmed = self._prepare_input(input)
        if position_indices.ndim != 1:
            raise ValueError("position_indices must be one-dimensional")
        if position_indices.numel() < 1:
            raise ValueError("position_indices must not be empty")
        if position_indices.dtype not in (torch.int32, torch.int64):
            raise TypeError("position_indices must contain integer indices")
        position_indices = position_indices.to(
            device=input.device,
            dtype=torch.long,
        )
        n_positions = self.n_chans * n_patches
        if torch.any(
            (position_indices < 0) | (position_indices >= n_positions)
        ).item():
            raise IndexError(
                f"position_indices must be in [0, {n_positions})"
            )

        selected_chans = torch.div(
            position_indices,
            n_patches,
            rounding_mode="floor",
        )
        selected_patches = torch.remainder(position_indices, n_patches)
        selected_groups = torch.div(
            selected_chans,
            self.chans_per_group,
            rounding_mode="floor",
        )

        group_outputs = []
        group_position_slots = []
        for group_index, cfc in enumerate(self.group_cfcs):
            slots = torch.nonzero(
                selected_groups == group_index,
                as_tuple=False,
            ).flatten()
            if slots.numel() == 0:
                continue

            chan_start = group_index * self.chans_per_group
            chan_stop = chan_start + self.chans_per_group
            group = trimmed[:, chan_start:chan_stop, :]
            patches = group.unfold(
                dimension=-1,
                size=self.patch_size,
                step=self.patch_stride,
            )
            local_chans = selected_chans.index_select(0, slots) - chan_start
            patch_indices = selected_patches.index_select(0, slots)
            selected_sequences = patches[
                :, local_chans, patch_indices, :
            ].reshape(
                batch_size * slots.numel(),
                self.patch_size,
                1,
            )
            final_outputs = self._run_cfc(cfc, selected_sequences)
            group_outputs.append(
                final_outputs.reshape(
                    batch_size,
                    slots.numel(),
                    self.n_features,
                )
            )
            group_position_slots.append(slots)

        grouped_outputs = torch.cat(group_outputs, dim=1)
        grouped_slots = torch.cat(group_position_slots)
        restore_order = torch.argsort(grouped_slots)
        ordered_outputs = grouped_outputs.index_select(1, restore_order)
        return ordered_outputs.permute(0, 2, 1).contiguous()


class _GlobalSpatialCfCOffsetWarp(nn.Module):
    """Predict and apply gated temporal offsets to EEGNet spatial features."""

    def __init__(
        self,
        *,
        n_feature_maps: int,
        hidden_size: int,
        backbone_units: int,
        backbone_layers: int,
        backbone_dropout: float,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.n_feature_maps = _positive_integer(
            "n_feature_maps", n_feature_maps
        )
        self.hidden_size = _positive_integer(
            "offset_cfc_hidden_size", hidden_size
        )
        backbone_units = _positive_integer(
            "offset_cfc_backbone_units", backbone_units
        )
        backbone_layers = _positive_integer(
            "offset_cfc_backbone_layers", backbone_layers
        )
        backbone_dropout = _dropout_probability(
            "offset_cfc_backbone_dropout", backbone_dropout
        )

        self.cfc = CfC(
            input_size=1,
            units=self.hidden_size,
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
        self.offset_head = nn.Linear(self.hidden_size, 1)
        self.gate_head = nn.Linear(self.hidden_size, 1)
        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)

        # The deformation starts as the identity map. The offset head receives
        # gradients immediately through the neutral sigmoid gate (0.5).
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)

        self.backbone_units = backbone_units
        self.backbone_layers = backbone_layers
        self.backbone_dropout = backbone_dropout
        self.padding_mode = "border"
        self.align_corners = True

    def forward(self, input: Tensor) -> Tensor:
        if input.ndim != 4:
            raise ValueError(
                "spatial CfC deformation expects (B,M,1,T), "
                f"got shape {tuple(input.shape)}."
            )
        batch_size, n_feature_maps, spatial_size, n_times = input.shape
        if n_feature_maps != self.n_feature_maps or spatial_size != 1:
            raise ValueError(
                f"expected (B,{self.n_feature_maps},1,T), got "
                f"{tuple(input.shape)}."
            )

        sequences = input[:, :, 0, :].reshape(
            batch_size * n_feature_maps,
            n_times,
            1,
        )
        hidden_sequence, _ = self.cfc(
            sequences, hx=None, timespans=None
        )
        raw_offset = self.offset_head(hidden_sequence).squeeze(-1)
        gate = torch.sigmoid(self.gate_head(hidden_sequence)).squeeze(-1)
        offset = gate * raw_offset

        signal = input.reshape(batch_size * n_feature_maps, 1, 1, n_times)
        base_time = torch.arange(
            n_times,
            device=input.device,
            dtype=input.dtype,
        ).unsqueeze(0)
        sample_time = base_time + offset
        if n_times == 1:
            normalized_time = torch.zeros_like(sample_time)
        else:
            normalized_time = 2.0 * sample_time / (n_times - 1) - 1.0
        grid = torch.stack(
            (
                normalized_time.unsqueeze(1),
                torch.zeros_like(normalized_time).unsqueeze(1),
            ),
            dim=-1,
        )
        warped = F.grid_sample(
            signal,
            grid,
            mode="bilinear",
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )
        return warped.reshape(batch_size, n_feature_maps, 1, n_times)


class deformable_patchCfC(egroupPatchCfC_EEGNet):
    """Group Patch-CfC EEGNet with spatial-feature temporal deformation."""

    def __init__(
        self,
        *args,
        patch_size: int = 32,
        patch_stride: int = 4,
        n_electrode_groups: int = 16,
        cfc_hidden_size: Optional[int] = None,
        cfc_backbone_units: int = 8,
        cfc_backbone_layers: int = 1,
        cfc_backbone_dropout: float = 0.0,
        cfc_activation_checkpoint: bool = False,
        offset_cfc_hidden_size: int = 8,
        offset_cfc_backbone_units: Optional[int] = None,
        offset_cfc_backbone_layers: Optional[int] = None,
        offset_cfc_backbone_dropout: Optional[float] = None,
        normalization: str = "batchnorm",
        n_norm_groups: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            patch_size=patch_size,
            patch_stride=patch_stride,
            n_electrode_groups=n_electrode_groups,
            cfc_hidden_size=cfc_hidden_size,
            cfc_backbone_units=cfc_backbone_units,
            cfc_backbone_layers=cfc_backbone_layers,
            cfc_backbone_dropout=cfc_backbone_dropout,
            normalization=normalization,
            n_norm_groups=n_norm_groups,
            **kwargs,
        )

        if offset_cfc_backbone_units is None:
            offset_cfc_backbone_units = cfc_backbone_units
        if offset_cfc_backbone_layers is None:
            offset_cfc_backbone_layers = cfc_backbone_layers
        if offset_cfc_backbone_dropout is None:
            offset_cfc_backbone_dropout = cfc_backbone_dropout

        reference_parameter = next(self.conv_temporal.parameters())
        self.conv_temporal = _ElectrodeGroupPatchLastCfCTemporal(
            n_chans=self.n_chans,
            n_features=self.F1,
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
            n_electrode_groups=self.n_electrode_groups,
            backbone_units=cfc_backbone_units,
            backbone_layers=cfc_backbone_layers,
            backbone_dropout=cfc_backbone_dropout,
            activation_checkpoint=cfc_activation_checkpoint,
            device=reference_parameter.device,
            dtype=reference_parameter.dtype,
        )

        self.temporal_output_length = self.patch_count
        self.pool_1 = _GlobalSpatialCfCOffsetWarp(
            n_feature_maps=self.F1 * self.D,
            hidden_size=offset_cfc_hidden_size,
            backbone_units=offset_cfc_backbone_units,
            backbone_layers=offset_cfc_backbone_layers,
            backbone_dropout=offset_cfc_backbone_dropout,
            device=reference_parameter.device,
            dtype=reference_parameter.dtype,
        )

        self.offset_cfc_hidden_size = offset_cfc_hidden_size
        self.offset_cfc_backbone_units = offset_cfc_backbone_units
        self.offset_cfc_backbone_layers = offset_cfc_backbone_layers
        self.offset_cfc_backbone_dropout = offset_cfc_backbone_dropout
        self.cfc_activation_checkpoint = cfc_activation_checkpoint
        self.offset_padding_mode = self.pool_1.padding_mode
        self.offset_gate = "sigmoid"
        self.ncps_version = ncps.__version__

        feature_time = self._deformable_feature_time_length()
        if feature_time < 1:
            raise ValueError(
                "configured patching leaves too few samples before the "
                "classifier."
            )
        previous_final_length = self.final_conv_length
        self.final_conv_length = feature_time
        if previous_final_length != feature_time:
            if hasattr(self.final_layer, "linearconstraint"):
                previous_linear = self.final_layer.linearconstraint
                classifier = LinearWithConstraint(
                    in_features=self.F2 * feature_time,
                    out_features=self.n_outputs,
                    max_norm=self.norm_rate,
                ).to(
                    device=previous_linear.weight.device,
                    dtype=previous_linear.weight.dtype,
                )
                self.final_layer.linearconstraint = classifier
            else:
                previous_classifier = self.final_layer.conv_classifier
                classifier = nn.Conv2d(
                    self.F2,
                    self.n_outputs,
                    kernel_size=(1, feature_time),
                    bias=True,
                ).to(
                    device=previous_classifier.weight.device,
                    dtype=previous_classifier.weight.dtype,
                )
                self.final_layer.conv_classifier = classifier
            glorot_weight_zero_bias(self.final_layer)

    def _deformable_feature_time_length(self) -> int:
        depthwise_delta = (
            2 * (self.depthwise_kernel_length // 2)
            - self.depthwise_kernel_length
            + 1
        )
        after_depthwise = self.temporal_output_length + depthwise_delta
        if after_depthwise < 1:
            return 0
        return after_depthwise // self.pool2_kernel_size


def build_deformable_patchcfc(
    n_classes: int,
    *,
    n_channels: int = 128,
    n_times: int = 1153,
    **model_kwargs,
) -> deformable_patchCfC:
    """Build the deformable group Patch-CfC EEGNet."""
    return deformable_patchCfC(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        **model_kwargs,
    )


if __name__ == "__main__":
    model = build_deformable_patchcfc(
        n_classes=4,
        n_channels=16,
        n_times=129,
        patch_size=16,
        patch_stride=4,
        n_electrode_groups=4,
    )
    eeg = torch.randn(2, 16, 129, requires_grad=True)
    temporal_outputs = []
    offset_io = []
    temporal_handle = model.conv_temporal.register_forward_hook(
        lambda _module, _inputs, output: temporal_outputs.append(output)
    )
    offset_handle = model.pool_1.register_forward_hook(
        lambda _module, inputs, output: offset_io.append((inputs[0], output))
    )
    logits = model(eeg)
    temporal_handle.remove()
    offset_handle.remove()

    if temporal_outputs[0].shape != (2, 8, 16, 29):
        raise RuntimeError(
            f"unexpected Patch-CfC output: {temporal_outputs[0].shape}"
        )
    if offset_io[0][0].shape != (2, 16, 1, 29):
        raise RuntimeError(f"unexpected offset input: {offset_io[0][0].shape}")
    if not torch.allclose(offset_io[0][0], offset_io[0][1], atol=1e-6):
        raise RuntimeError("zero-initialized deformation is not the identity")
    if logits.shape != (2, 4):
        raise RuntimeError(f"unexpected logits shape: {logits.shape}")
    if any(isinstance(module, nn.LSTM) for module in model.modules()):
        raise RuntimeError("deformable_patchCfC contains an LSTM")
    if (
        sum(isinstance(module, nn.BatchNorm2d) for module in model.modules())
        != 3
    ):
        raise RuntimeError(
            "BatchNorm model must contain exactly three BatchNorm2d"
        )
    if any(isinstance(module, nn.GroupNorm) for module in model.modules()):
        raise RuntimeError("BatchNorm model still contains GroupNorm")
    if sum(isinstance(module, CfC) for module in model.modules()) != 5:
        raise RuntimeError("expected four group CfCs and one global CfC")

    logits.sum().backward()
    if model.pool_1.offset_head.weight.grad is None:
        raise RuntimeError("offset head did not receive gradients")
    print("deformable_patchCfC self-test passed")
