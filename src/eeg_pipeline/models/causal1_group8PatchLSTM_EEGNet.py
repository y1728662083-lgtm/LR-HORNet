"""EEGNet with electrode-group-specific overlap-save Patch-LSTMs.

The first temporal convolution of :class:`causal1_EEGNet` is replaced by
independent LSTMs, one per contiguous electrode group (eight by default).
Within a group, the same LSTM parameters are shared by every electrode and
every temporal patch, while each electrode-patch pair starts from a fresh
recurrent state.

Every patch is evaluated for all recurrent time steps. With overlapping
patches, the first patch keeps all K outputs and later patches keep only their
final S novel outputs. The discarded overlap outputs still provide recurrent
context. Retained outputs are restored to the original temporal order before
the unchanged EEGNet spatial block.
"""

from __future__ import annotations

import operator
from typing import Dict, Optional

import torch
from torch import Tensor, nn

from braindecode.functional import glorot_weight_zero_bias
from braindecode.modules import LinearWithConstraint

from eeg_pipeline.models.causal1_EEGNet import causal1_EEGNet
from eeg_pipeline.models.causal1_groupnorm_EEGNet import (
    _group_norm_from_batch_norm,
)


__all__ = [
    "causal1_group8PatchLSTM_EEGNet",
    "build_causal1_group8patchlstm_eegnet",
]


def _positive_integer(name: str, value: int) -> int:
    """Return ``value`` as an int, rejecting booleans and non-positive values."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    try:
        parsed = operator.index(value)
    except TypeError as error:
        raise ValueError(
            f"{name} must be a positive integer, got {value!r}."
        ) from error
    if parsed < 1:
        raise ValueError(f"{name} must be positive, got {parsed}.")
    return parsed


class _ElectrodeGroupPatchLSTMTemporal(nn.Module):
    """Apply one independent overlap-save LSTM per electrode group.

    Input shape is ``(B, 1, C, T)`` and output shape is
    ``(B, F1, C, T_effective)``. Patches have length ``patch_size`` and start
    ``patch_stride`` samples apart. The first patch emits all of its outputs;
    every later patch emits only its final ``patch_stride`` novel outputs.
    """

    def __init__(
        self,
        *,
        n_chans: int,
        n_features: int,
        patch_size: int,
        patch_stride: Optional[int] = None,
        n_electrode_groups: int = 8,
        allow_uneven_groups: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.n_chans = _positive_integer("n_chans", n_chans)
        self.n_features = _positive_integer("n_features", n_features)
        self.patch_size = _positive_integer("patch_size", patch_size)
        if patch_stride is None:
            patch_stride = self.patch_size
        self.patch_stride = _positive_integer(
            "patch_stride", patch_stride
        )
        if self.patch_stride > self.patch_size:
            raise ValueError(
                "patch_stride must not exceed patch_size; otherwise the "
                "restored time axis would contain gaps. Got "
                f"patch_stride={self.patch_stride} and "
                f"patch_size={self.patch_size}."
            )

        self.n_electrode_groups = _positive_integer(
            "n_electrode_groups", n_electrode_groups
        )
        if not isinstance(allow_uneven_groups, bool):
            raise TypeError("allow_uneven_groups must be a bool.")
        self.allow_uneven_groups = allow_uneven_groups
        base_size, remainder = divmod(
            self.n_chans, self.n_electrode_groups
        )
        if remainder and not self.allow_uneven_groups:
            raise ValueError(
                "n_chans must be divisible by n_electrode_groups; "
                f"got n_chans={self.n_chans} and "
                f"n_electrode_groups={self.n_electrode_groups}."
            )

        self.electrode_group_sizes = tuple(
            base_size + int(group_index < remainder)
            for group_index in range(self.n_electrode_groups)
        )
        boundaries = [0]
        for group_size in self.electrode_group_sizes:
            boundaries.append(boundaries[-1] + group_size)
        self._electrode_group_boundaries = tuple(boundaries)
        self.chans_per_group = (
            base_size if remainder == 0 else None
        )
        factory_kwargs = {"device": device, "dtype": dtype}
        self.group_lstms = nn.ModuleList(
            [
                nn.LSTM(
                    input_size=1,
                    hidden_size=self.n_features,
                    num_layers=1,
                    batch_first=True,
                    dropout=0.0,
                    bidirectional=False,
                    **factory_kwargs,
                )
                for _ in range(self.n_electrode_groups)
            ]
        )

    def forward(self, input: Tensor) -> Tensor:
        if input.ndim != 4:
            raise ValueError(
                "group Patch-LSTM expects a 4D tensor (B,1,C,T), "
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
        trimmed = input[:, 0, :, :effective_n_times]

        group_outputs = []
        for group_index, lstm in enumerate(self.group_lstms):
            chan_start = self._electrode_group_boundaries[group_index]
            chan_stop = self._electrode_group_boundaries[group_index + 1]
            group_size = chan_stop - chan_start
            group = trimmed[:, chan_start:chan_stop, :]

            # Each (batch, electrode, patch) tuple is an independent sequence.
            # Preserve the old trim+reshape path when there is no overlap.
            if self.patch_stride == self.patch_size:
                patches = group.reshape(
                    batch_size,
                    group_size,
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
                batch_size * group_size * n_patches,
                self.patch_size,
                1,
            )
            recurrent_sequence, _ = lstm(sequences)
            recurrent_sequence = recurrent_sequence.reshape(
                batch_size,
                group_size,
                n_patches,
                self.patch_size,
                self.n_features,
            )

            if n_patches == 1:
                group_output = recurrent_sequence[:, :, 0, :, :]
            elif self.patch_stride == self.patch_size:
                group_output = recurrent_sequence.reshape(
                    batch_size,
                    group_size,
                    effective_n_times,
                    self.n_features,
                )
            else:
                # Every electrode keeps its complete first patch. The leading
                # K-S outputs of later patches overlap already-emitted times;
                # they remain recurrent warm-up context but are not emitted.
                first_patch = recurrent_sequence[:, :, 0, :, :]
                later_novel = recurrent_sequence[
                    :, :, 1:, self.patch_size - self.patch_stride :, :
                ].reshape(
                    batch_size,
                    group_size,
                    (n_patches - 1) * self.patch_stride,
                    self.n_features,
                )
                group_output = torch.cat(
                    (first_patch, later_novel), dim=2
                )
            group_outputs.append(group_output)

        # Groups occupy disjoint slices of the original electrode axis.
        output = torch.cat(group_outputs, dim=1)
        return output.permute(0, 3, 1, 2).contiguous()


class causal1_group8PatchLSTM_EEGNet(causal1_EEGNet):
    """Replace causal1_EEGNet's first temporal convolution with group LSTMs.

    ``patch_size`` controls the recurrent context length. ``patch_stride``
    defaults to ``patch_size`` for non-overlapping patches and may be reduced
    to add overlap. ``n_electrode_groups`` controls how many independently
    parameterized LSTMs split the electrode axis and defaults to eight.
    For overlapping patches, only each later patch's final ``patch_stride``
    outputs are emitted; its leading overlap is recurrent warm-up context.
    Incomplete samples at the right edge are removed.

    Apart from ``conv_temporal`` and an automatically resized classifier when
    the effective feature length changes, inherited EEGNet layers are kept.
    The later separable temporal convolution remains symmetrically padded, so
    this model is not end-to-end causal.
    """

    def __init__(
        self,
        # Signal parameters
        n_chans: Optional[int] = None,
        n_outputs: Optional[int] = None,
        n_times: Optional[int] = None,
        # Patch-LSTM parameters
        patch_size: int = 64,
        patch_stride: Optional[int] = None,
        n_electrode_groups: int = 8,
        allow_uneven_groups: bool = False,
        # EEGNet parameters retained from causal1_EEGNet
        final_conv_length: str | int = "auto",
        pool_mode: str = "mean",
        F1: int = 8,
        D: int = 2,
        F2: Optional[int] = None,
        kernel_length: int = 64,
        *,
        depthwise_kernel_length: int = 16,
        pool1_kernel_size: int = 4,
        pool2_kernel_size: int = 8,
        conv_spatial_max_norm: int = 1,
        activation: type[nn.Module] = nn.ELU,
        batch_norm_momentum: float = 0.01,
        batch_norm_affine: bool = True,
        batch_norm_eps: float = 1e-3,
        normalization: str = "batchnorm",
        n_norm_groups: int = 4,
        drop_prob: float = 0.25,
        final_layer_with_constraint: bool = False,
        norm_rate: float = 0.25,
        # Alternative signal metadata
        chs_info: Optional[list[Dict]] = None,
        input_window_seconds=None,
        sfreq=None,
        **kwargs,
    ) -> None:
        patch_size = _positive_integer("patch_size", patch_size)
        if patch_stride is None:
            patch_stride = patch_size
        patch_stride = _positive_integer("patch_stride", patch_stride)
        if patch_stride > patch_size:
            raise ValueError(
                "patch_stride must be less than or equal to patch_size; "
                f"got patch_stride={patch_stride} and "
                f"patch_size={patch_size}."
            )
        n_electrode_groups = _positive_integer(
            "n_electrode_groups", n_electrode_groups
        )
        if final_conv_length != "auto":
            raise ValueError(
                "causal1_group8PatchLSTM_EEGNet requires "
                "final_conv_length='auto'; use patch_size and patch_stride "
                "to configure its Patch-LSTM temporal windows."
            )
        if kernel_length != 64:
            raise ValueError(
                "kernel_length belongs to the discarded causal convolution "
                "and must remain 64; use patch_size for this model."
            )

        # Keep this parent construction path independent of patch settings.
        # This preserves inherited-layer initialization across K and S.
        super().__init__(
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_times=n_times,
            final_conv_length="auto",
            pool_mode=pool_mode,
            F1=F1,
            D=D,
            F2=F2,
            kernel_length=64,
            depthwise_kernel_length=depthwise_kernel_length,
            pool1_kernel_size=pool1_kernel_size,
            pool2_kernel_size=pool2_kernel_size,
            conv_spatial_max_norm=conv_spatial_max_norm,
            activation=activation,
            batch_norm_momentum=batch_norm_momentum,
            batch_norm_affine=batch_norm_affine,
            batch_norm_eps=batch_norm_eps,
            drop_prob=drop_prob,
            final_layer_with_constraint=final_layer_with_constraint,
            norm_rate=norm_rate,
            chs_info=chs_info,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
            **kwargs,
        )

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

        if self.n_chans is None or self.n_times is None:
            raise ValueError("n_chans and n_times must be known for this model.")
        if self.n_times < patch_size:
            raise ValueError(
                f"n_times={self.n_times} is shorter than patch_size="
                f"{patch_size}."
            )
        if not isinstance(allow_uneven_groups, bool):
            raise TypeError("allow_uneven_groups must be a bool.")
        if (
            self.n_chans % n_electrode_groups != 0
            and not allow_uneven_groups
        ):
            raise ValueError(
                "n_chans must be divisible by n_electrode_groups; "
                f"got n_chans={self.n_chans} and "
                f"n_electrode_groups={n_electrode_groups}."
            )

        reference_weight = self.conv_temporal.weight
        self.conv_temporal = _ElectrodeGroupPatchLSTMTemporal(
            n_chans=self.n_chans,
            n_features=self.F1,
            patch_size=patch_size,
            patch_stride=patch_stride,
            n_electrode_groups=n_electrode_groups,
            allow_uneven_groups=allow_uneven_groups,
            device=reference_weight.device,
            dtype=reference_weight.dtype,
        )

        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.patch_overlap = self.patch_size - self.patch_stride
        self.n_electrode_groups = n_electrode_groups
        self.allow_uneven_groups = allow_uneven_groups
        self.patch_count = (
            (self.n_times - self.patch_size) // self.patch_stride + 1
        )
        self.effective_n_times = (
            self.patch_size
            + (self.patch_count - 1) * self.patch_stride
        )
        self.discarded_tail = self.n_times - self.effective_n_times

        patch_feature_time = self._patch_feature_time_length()
        if patch_feature_time < 1:
            raise ValueError(
                f"n_times={self.n_times}, patch_size={self.patch_size}, and "
                f"patch_stride={self.patch_stride} leave too few samples "
                "after the two pooling stages."
            )

        previous_final_length = self.final_conv_length
        self.final_conv_length = patch_feature_time
        if previous_final_length != patch_feature_time:
            if final_layer_with_constraint:
                previous_linear = self.final_layer.linearconstraint
                linear = LinearWithConstraint(
                    in_features=self.F2 * self.final_conv_length,
                    out_features=self.n_outputs,
                    max_norm=self.norm_rate,
                ).to(
                    device=previous_linear.weight.device,
                    dtype=previous_linear.weight.dtype,
                )
                self.final_layer.linearconstraint = linear
            else:
                previous_classifier = self.final_layer.conv_classifier
                classifier = nn.Conv2d(
                    self.F2,
                    self.n_outputs,
                    kernel_size=(1, self.final_conv_length),
                    bias=True,
                ).to(
                    device=previous_classifier.weight.device,
                    dtype=previous_classifier.weight.dtype,
                )
                self.final_layer.conv_classifier = classifier
            glorot_weight_zero_bias(self.final_layer)

    def _patch_feature_time_length(self) -> int:
        """Return the post-Block-2 time length for configured input size."""
        after_pool1 = self.effective_n_times // self.pool1_kernel_size
        if after_pool1 < 1:
            return 0
        depthwise_delta = (
            2 * (self.depthwise_kernel_length // 2)
            - self.depthwise_kernel_length
            + 1
        )
        after_depthwise = after_pool1 + depthwise_delta
        if after_depthwise < 1:
            return 0
        return after_depthwise // self.pool2_kernel_size


def build_causal1_group8patchlstm_eegnet(
    n_classes: int,
    *,
    n_channels: int = 128,
    n_times: int = 1153,
    **model_kwargs,
) -> causal1_group8PatchLSTM_EEGNet:
    """Build the electrode-group overlap-save Patch-LSTM EEGNet variant."""
    return causal1_group8PatchLSTM_EEGNet(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        **model_kwargs,
    )


if __name__ == "__main__":
    for patch_size, patch_stride, expected_time, expected_params in (
        (64, 64, 1152, 7764),
        (64, 32, 1152, 7764),
        (256, 128, 1152, 7764),
        (512, 256, 1024, 7508),
    ):
        model = build_causal1_group8patchlstm_eegnet(
            n_classes=4,
            patch_size=patch_size,
            patch_stride=patch_stride,
        )
        eeg = torch.randn(1, 128, 1153, requires_grad=True)
        captured = []
        handle = model.conv_temporal.register_forward_hook(
            lambda _module, _inputs, output: captured.append(output)
        )
        logits = model(eeg)
        handle.remove()

        if captured[0].shape != (1, 8, 128, expected_time):
            raise RuntimeError(
                f"Unexpected temporal shape: {tuple(captured[0].shape)}"
            )
        if logits.shape != (1, 4):
            raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")
        parameter_count = sum(
            parameter.numel() for parameter in model.parameters()
        )
        if parameter_count != expected_params:
            raise RuntimeError(
                f"Unexpected parameter count: {parameter_count}"
            )
        logits.sum().backward()

    print("causal1_group8PatchLSTM_EEGNet self-test passed")
