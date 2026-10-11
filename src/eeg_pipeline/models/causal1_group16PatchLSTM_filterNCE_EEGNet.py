"""Group-16 overlap-save Patch-LSTM EEGNet with training-only TF-C loss.

The classification backbone is the existing
``causal1_group8PatchLSTM_EEGNet`` configured with 16 electrode groups.  Its
complete overlap-save temporal output is retained for classification.  For
contrastive learning only, the final time point of every raw Patch-LSTM window
is aligned with one cached filter-bank position.  The augmented time branch
encodes only the sampled electrode/patch positions, while evaluation and
inference execute only the unchanged classifier.
"""

from __future__ import annotations

import operator
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from eeg_pipeline.models.causal1_group8PatchLSTM_EEGNet import (
    _ElectrodeGroupPatchLSTMTemporal,
    causal1_group8PatchLSTM_EEGNet,
)


__all__ = [
    "PatchLSTMFilterNCETrainingOutput",
    "causal1_group16PatchLSTM_filterNCE_EEGNet",
    "build_causal1_group16patchlstm_filternce_eegnet",
]


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        parsed = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


class PatchLSTMFilterNCETrainingOutput(NamedTuple):
    """Representations consumed by the existing structured TF-C objective."""

    logits: Tensor
    h_time: Tensor
    h_time_augmented: Tensor
    h_frequency: Tensor
    h_frequency_augmented: Tensor
    z_time: Tensor
    z_frequency: Tensor


class _PointwiseProjector(nn.Sequential):
    """The verified TF-C two-layer projector for structured EEG maps."""

    def __init__(
        self,
        input_features: int,
        hidden_features: int,
        output_features: int,
        *,
        batch_norm_momentum: float,
        batch_norm_eps: float,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                input_features,
                hidden_features,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                hidden_features,
                momentum=batch_norm_momentum,
                eps=batch_norm_eps,
            ),
            nn.ReLU(),
            nn.Conv2d(
                hidden_features,
                output_features,
                kernel_size=1,
                bias=True,
            ),
        )


class _CheckpointedGroupPatchLSTMTemporal(nn.Module):
    """Adapt an initialized group Patch-LSTM without changing its parameters.

    The original ``group_lstms`` ModuleList is transferred directly, so model
    initialization and state-dict keys remain ``conv_temporal.group_lstms.*``.
    """

    def __init__(
        self,
        source: _ElectrodeGroupPatchLSTMTemporal,
        *,
        activation_checkpoint: bool,
    ) -> None:
        super().__init__()
        if not isinstance(activation_checkpoint, bool):
            raise ValueError("lstm_activation_checkpoint must be a boolean")
        self.n_chans = source.n_chans
        self.n_features = source.n_features
        self.patch_size = source.patch_size
        self.patch_stride = source.patch_stride
        self.n_electrode_groups = source.n_electrode_groups
        self.chans_per_group = source.chans_per_group
        self.group_lstms = source.group_lstms
        self.activation_checkpoint = activation_checkpoint

    def _prepare_input(self, input: Tensor) -> tuple[int, int, Tensor]:
        if input.ndim != 4:
            raise ValueError(
                "group Patch-LSTM expects (B,1,C,T), "
                f"got shape {tuple(input.shape)}."
            )
        batch_size, input_features, n_chans, n_times = input.shape
        if input_features != 1:
            raise ValueError(
                "group Patch-LSTM expects one input feature before temporal "
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
        return batch_size, n_patches, input[:, 0, :, :effective_n_times]

    def _run_lstm(
        self,
        lstm: nn.LSTM,
        sequences: Tensor,
        *,
        final_only: bool,
    ) -> Tensor:
        def recurrent_outputs(current_sequences: Tensor) -> Tensor:
            outputs, _ = lstm(current_sequences)
            if final_only:
                return outputs[:, -1, :]
            return outputs

        if (
            self.activation_checkpoint
            and self.training
            and torch.is_grad_enabled()
        ):
            return checkpoint(
                recurrent_outputs,
                sequences,
                use_reentrant=False,
            )
        return recurrent_outputs(sequences)

    def forward(self, input: Tensor) -> Tensor:
        """Preserve the original complete overlap-save reconstruction."""
        batch_size, n_patches, trimmed = self._prepare_input(input)
        effective_n_times = trimmed.shape[-1]

        group_outputs = []
        for group_index, lstm in enumerate(self.group_lstms):
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
            recurrent_sequence = self._run_lstm(
                lstm,
                sequences,
                final_only=False,
            ).reshape(
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
                    (first_patch, later_novel),
                    dim=2,
                )
            group_outputs.append(group_output)

        output = torch.cat(group_outputs, dim=1)
        return output.permute(0, 3, 1, 2).contiguous()

    def forward_selected(
        self,
        input: Tensor,
        position_indices: Tensor,
    ) -> Tensor:
        """Encode only sampled channel-major electrode/Patch positions."""
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
        for group_index, lstm in enumerate(self.group_lstms):
            slots = torch.nonzero(
                selected_groups == group_index,
                as_tuple=False,
            ).flatten()
            if slots.numel() == 0:
                continue
            chan_start = group_index * self.chans_per_group
            chan_stop = chan_start + self.chans_per_group
            patches = trimmed[
                :, chan_start:chan_stop, :
            ].unfold(
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
            final_outputs = self._run_lstm(
                lstm,
                selected_sequences,
                final_only=True,
            )
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


class causal1_group16PatchLSTM_filterNCE_EEGNet(
    causal1_group8PatchLSTM_EEGNet
):
    """Fixed 16-group Patch-LSTM with a training-only filter-NCE branch."""

    def __init__(
        self,
        *args,
        patch_size: int = 128,
        patch_stride: int = 64,
        n_electrode_groups: int = 16,
        lstm_activation_checkpoint: bool = False,
        n_frequency_bands: int = 8,
        projection_hidden_size: int = 16,
        projection_size: int = 8,
        normalization: str = "batchnorm",
        **kwargs,
    ) -> None:
        if n_electrode_groups != 16:
            raise ValueError(
                "causal1_group16PatchLSTM_filterNCE_EEGNet requires "
                "n_electrode_groups=16"
            )
        if not isinstance(normalization, str):
            raise TypeError("normalization must be 'batchnorm'")
        if normalization.lower() != "batchnorm":
            raise ValueError(
                "causal1_group16PatchLSTM_filterNCE_EEGNet preserves the "
                "validated BatchNorm backbone; normalization must be "
                "'batchnorm'"
            )
        super().__init__(
            *args,
            patch_size=patch_size,
            patch_stride=patch_stride,
            n_electrode_groups=n_electrode_groups,
            normalization="batchnorm",
            **kwargs,
        )
        if not isinstance(lstm_activation_checkpoint, bool):
            raise ValueError("lstm_activation_checkpoint must be a boolean")
        self.conv_temporal = _CheckpointedGroupPatchLSTMTemporal(
            self.conv_temporal,
            activation_checkpoint=lstm_activation_checkpoint,
        )
        self.lstm_activation_checkpoint = lstm_activation_checkpoint
        self.n_frequency_bands = _positive_integer(
            "n_frequency_bands",
            n_frequency_bands,
        )
        projection_hidden_size = _positive_integer(
            "projection_hidden_size",
            projection_hidden_size,
        )
        projection_size = _positive_integer(
            "projection_size",
            projection_size,
        )

        # Freeze the inherited Sequential classifier path before registering
        # training-only modules.
        self._backbone_module_names = tuple(self._modules.keys())
        patch_end_indices = (
            torch.arange(self.patch_count, dtype=torch.long)
            * self.patch_stride
            + self.patch_size
            - 1
        )
        self.register_buffer(
            "_contrastive_patch_end_indices",
            patch_end_indices,
            persistent=False,
        )
        self.frequency_encoder = nn.Sequential(
            nn.Conv2d(
                self.n_frequency_bands,
                self.F1,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                self.F1,
                momentum=self.batch_norm_momentum,
                eps=self.batch_norm_eps,
            ),
            nn.ELU(),
        )
        self.time_projector = _PointwiseProjector(
            self.F1,
            projection_hidden_size,
            projection_size,
            batch_norm_momentum=self.batch_norm_momentum,
            batch_norm_eps=self.batch_norm_eps,
        )
        self.frequency_projector = _PointwiseProjector(
            self.F1,
            projection_hidden_size,
            projection_size,
            batch_norm_momentum=self.batch_norm_momentum,
            batch_norm_eps=self.batch_norm_eps,
        )
        self.projection_hidden_size = projection_hidden_size
        self.projection_size = projection_size

    def _select_patch_endpoints(self, full_time: Tensor) -> Tensor:
        if full_time.ndim != 4:
            raise ValueError(
                "full Patch-LSTM output must have shape (B,F,C,T), got "
                f"{tuple(full_time.shape)}"
            )
        indices = self._contrastive_patch_end_indices
        if indices[-1].item() >= full_time.shape[-1]:
            raise ValueError(
                "Patch-LSTM output is shorter than the configured patch "
                "endpoint indices"
            )
        return full_time.index_select(-1, indices)

    def _forward_backbone_with_time_features(
        self,
        eeg: Tensor,
    ) -> tuple[Tensor, Tensor]:
        features = eeg
        h_time = None
        for module_name in self._backbone_module_names:
            features = self._modules[module_name](features)
            if module_name == "conv_temporal":
                h_time = self._select_patch_endpoints(features)
        if h_time is None:
            raise RuntimeError("conv_temporal was not found in the backbone")
        return features, h_time

    def _encode_augmented_time(
        self,
        eeg: Tensor,
        position_indices: Tensor | None,
    ) -> Tensor:
        features = self.ensuredims(eeg)
        features = self.dimshuffle(features)
        if position_indices is not None:
            return self.conv_temporal.forward_selected(
                features,
                position_indices,
            )
        return self._select_patch_endpoints(self.conv_temporal(features))

    def _encode_frequency(self, frequency: Tensor) -> Tensor:
        if frequency.ndim != 4:
            raise ValueError(
                "frequency input must have shape (B,C,K,P), got "
                f"{tuple(frequency.shape)}"
            )
        if frequency.shape[1] != self.n_chans:
            raise ValueError(
                f"expected {self.n_chans} frequency channels, got "
                f"{frequency.shape[1]}"
            )
        if frequency.shape[2] != self.n_frequency_bands:
            raise ValueError(
                f"expected {self.n_frequency_bands} bands, got "
                f"{frequency.shape[2]}"
            )
        if frequency.shape[3] != self.patch_count:
            raise ValueError(
                f"expected {self.patch_count} frequency patches, got "
                f"{frequency.shape[3]}"
            )
        return self.frequency_encoder(
            frequency.permute(0, 2, 1, 3).contiguous()
        )

    def forward(self, eeg: Tensor) -> Tensor:
        """Run only the original overlap-save EEGNet classifier."""
        if not hasattr(self, "_backbone_module_names"):
            return nn.Sequential.forward(self, eeg)
        features = eeg
        for module_name in self._backbone_module_names:
            features = self._modules[module_name](features)
        return features

    def forward_contrastive(
        self,
        eeg: Tensor,
        augmented_eeg: Tensor,
        frequency: Tensor,
        augmented_frequency: Tensor,
        *,
        time_position_indices: Tensor | None = None,
    ) -> PatchLSTMFilterNCETrainingOutput:
        """Return aligned time/frequency views for training only."""
        logits, h_time = self._forward_backbone_with_time_features(eeg)
        h_time_augmented = self._encode_augmented_time(
            augmented_eeg,
            time_position_indices,
        )
        h_frequency = self._encode_frequency(frequency)
        h_frequency_augmented = self._encode_frequency(
            augmented_frequency
        )
        z_time = self.time_projector(h_time)
        z_frequency = self.frequency_projector(h_frequency)
        return PatchLSTMFilterNCETrainingOutput(
            logits=logits,
            h_time=h_time,
            h_time_augmented=h_time_augmented,
            h_frequency=h_frequency,
            h_frequency_augmented=h_frequency_augmented,
            z_time=z_time,
            z_frequency=z_frequency,
        )


def build_causal1_group16patchlstm_filternce_eegnet(
    n_classes: int,
    *,
    n_channels: int = 128,
    n_times: int = 1153,
    **model_kwargs,
) -> causal1_group16PatchLSTM_filterNCE_EEGNet:
    """Build the fixed 16-group Patch-LSTM filter-NCE model."""
    return causal1_group16PatchLSTM_filterNCE_EEGNet(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        **model_kwargs,
    )
