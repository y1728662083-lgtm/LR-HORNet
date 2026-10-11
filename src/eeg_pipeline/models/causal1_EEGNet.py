"""EEGNet with grouped, left-only causal filtering in its first layer."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from braindecode.functional import glorot_weight_zero_bias
from braindecode.models import EEGNet
from braindecode.modules import LinearWithConstraint


__all__ = ["causal1_EEGNet", "build_causal1_eegnet"]


class _LeftCausalTemporalConv2d(nn.Conv2d):
    """Conv2d padded only on the past side of the time dimension."""

    def forward(self, input: Tensor) -> Tensor:
        left_padding = self.dilation[1] * (self.kernel_size[1] - 1)
        if left_padding:
            input = F.pad(input, (left_padding, 0, 0, 0))
        return self._conv_forward(input, self.weight, self.bias)


class _GroupedLeftCausalTemporalConv2d(nn.Module):
    """Apply one causal temporal filter bank per electrode group.

    The input and output layouts remain compatible with EEGNet:
    ``[batch, 1, electrodes, time]`` is mapped to
    ``[batch, features, electrodes, time]``. Electrodes are split into
    contiguous, near-equal groups. Every electrode within a group uses the
    same temporal kernels, while different groups have independent kernels.

    With ``n_electrode_groups=1``, the weight shape and computation are the
    same as the former globally shared causal ``Conv2d`` implementation.
    """

    def __init__(
        self,
        *,
        n_chans: int,
        out_channels: int,
        kernel_length: int,
        n_electrode_groups: int,
    ) -> None:
        super().__init__()

        if isinstance(n_electrode_groups, bool) or not isinstance(
            n_electrode_groups, int
        ):
            raise TypeError("n_electrode_groups must be an integer.")
        if not 1 <= n_electrode_groups <= n_chans:
            raise ValueError(
                "n_electrode_groups must be between 1 and n_chans "
                f"(received {n_electrode_groups} for {n_chans} channels)."
            )
        if kernel_length < 1:
            raise ValueError("kernel_length must be at least 1.")

        self.n_chans = n_chans
        self.n_electrode_groups = n_electrode_groups
        self.in_channels = 1
        self.out_channels = out_channels
        self.kernel_size = (1, kernel_length)
        self.stride = (1, 1)
        self.padding = (0, 0)
        self.dilation = (1, 1)
        self.groups = 1

        # Kernels belonging to group g occupy
        # weight[g * out_channels : (g + 1) * out_channels]. Keeping groups
        # in the first dimension makes the G=1 state_dict exactly compatible
        # with the previous causal Conv2d implementation.
        self.weight = nn.Parameter(
            torch.empty(
                n_electrode_groups * out_channels,
                1,
                1,
                kernel_length,
            )
        )
        self.register_parameter("bias", None)

        base_size, remainder = divmod(n_chans, n_electrode_groups)
        group_sizes = tuple(
            base_size + int(group_index < remainder)
            for group_index in range(n_electrode_groups)
        )
        boundaries = [0]
        for group_size in group_sizes:
            boundaries.append(boundaries[-1] + group_size)
        self.electrode_group_sizes = group_sizes
        self._electrode_group_boundaries = tuple(boundaries)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize every group's filter bank as an independent Conv2d."""
        with torch.no_grad():
            for group_index in range(self.n_electrode_groups):
                weight_start = group_index * self.out_channels
                weight_stop = weight_start + self.out_channels
                nn.init.xavier_uniform_(
                    self.weight[weight_start:weight_stop]
                )

    @property
    def electrode_groups(self) -> tuple[tuple[int, ...], ...]:
        """Return the zero-based electrode indices assigned to each group."""
        return tuple(
            tuple(range(start, stop))
            for start, stop in zip(
                self._electrode_group_boundaries[:-1],
                self._electrode_group_boundaries[1:],
            )
        )

    def forward(self, input: Tensor) -> Tensor:
        """Filter each electrode with the kernels of its assigned group."""
        if input.ndim != 4:
            raise ValueError(
                "Grouped causal temporal convolution expects a 4-D tensor "
                "[batch, 1, electrodes, time]."
            )
        if input.shape[1] != 1 or input.shape[2] != self.n_chans:
            raise ValueError(
                "Expected input shape [batch, 1, "
                f"{self.n_chans}, time], received {tuple(input.shape)}."
            )

        left_padding = self.dilation[1] * (self.kernel_size[1] - 1)
        if left_padding:
            input = F.pad(input, (left_padding, 0, 0, 0))

        group_outputs = []
        for group_index, (start, stop) in enumerate(
            zip(
                self._electrode_group_boundaries[:-1],
                self._electrode_group_boundaries[1:],
            )
        ):
            weight_start = group_index * self.out_channels
            weight_stop = weight_start + self.out_channels
            group_outputs.append(
                F.conv2d(
                    input[:, :, start:stop, :],
                    self.weight[weight_start:weight_stop],
                    bias=None,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                )
            )
        return torch.cat(group_outputs, dim=2)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        """Expand an old globally shared temporal kernel across all groups."""
        weight_key = prefix + "weight"
        incoming_weight = state_dict.get(weight_key)
        old_weight_shape = (
            self.out_channels,
            1,
            1,
            self.kernel_size[1],
        )
        if (
            self.n_electrode_groups > 1
            and incoming_weight is not None
            and tuple(incoming_weight.shape) == old_weight_shape
        ):
            state_dict[weight_key] = incoming_weight.repeat(
                self.n_electrode_groups, 1, 1, 1
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def extra_repr(self) -> str:
        """Show the grouping configuration in model summaries."""
        return (
            f"n_chans={self.n_chans}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, "
            f"n_electrode_groups={self.n_electrode_groups}, "
            f"electrode_group_sizes={self.electrode_group_sizes}, "
            "causal_padding=left"
        )


class causal1_EEGNet(EEGNet):
    """EEGNet with grouped causal first-layer temporal filter banks.

    Only ``conv_temporal`` is changed. The electrode axis is divided into
    ``n_electrode_groups`` contiguous, near-equal groups. Each group learns
    its own ``F1`` causal temporal kernels, and all electrodes inside that
    group share those kernels. The default value 1 recovers global sharing.

    Every group receives ``kernel_length - 1`` zeros on the left and no
    right padding, so its output at time t depends only on input samples up
    to and including t. Its output length is therefore T instead of the
    original EEGNet's T + 1 for the default even-sized kernel.

    All later EEGNet layers are retained. In particular,
    conv_separable_depth keeps Braindecode's symmetric padding, so this
    variant changes only the first temporal convolution and is not an
    end-to-end causal network.
    """

    def __init__(
        self,
        # Signal parameters
        n_chans: Optional[int] = None,
        n_outputs: Optional[int] = None,
        n_times: Optional[int] = None,
        # Model parameters
        final_conv_length: str | int = "auto",
        pool_mode: str = "mean",
        F1: int = 8,
        D: int = 2,
        F2: Optional[int] = None,
        kernel_length: int = 64,
        *,
        n_electrode_groups: int = 1,
        depthwise_kernel_length: int = 16,
        pool1_kernel_size: int = 4,
        pool2_kernel_size: int = 8,
        conv_spatial_max_norm: int = 1,
        activation: type[nn.Module] = nn.ELU,
        batch_norm_momentum: float = 0.01,
        batch_norm_affine: bool = True,
        batch_norm_eps: float = 1e-3,
        drop_prob: float = 0.25,
        final_layer_with_constraint: bool = False,
        norm_rate: float = 0.25,
        # Alternative signal metadata
        chs_info: Optional[list[Dict]] = None,
        input_window_seconds=None,
        sfreq=None,
        **kwargs,
    ):
        auto_final_length = final_conv_length == "auto"

        super().__init__(
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_times=n_times,
            final_conv_length=final_conv_length,
            pool_mode=pool_mode,
            F1=F1,
            D=D,
            F2=F2,
            kernel_length=kernel_length,
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

        # Replacing an existing Sequential child preserves its module position
        # and the checkpoint key conv_temporal.weight. With one group, that
        # parameter also retains its original shape [F1, 1, 1, kernel_length].
        self.n_electrode_groups = n_electrode_groups
        self.conv_temporal = _GroupedLeftCausalTemporalConv2d(
            n_chans=self.n_chans,
            out_channels=self.F1,
            kernel_length=self.kernel_length,
            n_electrode_groups=n_electrode_groups,
        )

        causal_feature_time = self._causal_feature_time_length()
        if causal_feature_time < 1:
            raise ValueError(
                f"n_times={self.n_times} is too short for causal1_EEGNet "
                "after the two pooling stages."
            )

        # EEGNet builds its automatic classifier before this layer is replaced.
        # Recompute that classifier only when causal padding changes its length.
        if auto_final_length:
            previous_final_length = self.final_conv_length
            self.final_conv_length = causal_feature_time
            if previous_final_length != causal_feature_time:
                if final_layer_with_constraint:
                    self.final_layer.linearconstraint = LinearWithConstraint(
                        in_features=self.F2 * self.final_conv_length,
                        out_features=self.n_outputs,
                        max_norm=self.norm_rate,
                    )
                else:
                    self.final_layer.conv_classifier = nn.Conv2d(
                        self.F2,
                        self.n_outputs,
                        kernel_size=(1, self.final_conv_length),
                        bias=True,
                    )
                glorot_weight_zero_bias(self.final_layer)

    def _causal_feature_time_length(self) -> int:
        """Return the time length immediately before the classifier."""
        if self.n_times is None:
            raise ValueError("n_times must be known to build causal1_EEGNet.")

        after_pool1 = self.n_times // self.pool1_kernel_size
        depthwise_delta = (
            2 * (self.depthwise_kernel_length // 2)
            - self.depthwise_kernel_length
            + 1
        )
        after_depthwise = after_pool1 + depthwise_delta
        return after_depthwise // self.pool2_kernel_size


def build_causal1_eegnet(
    n_classes: int,
    *,
    n_channels: int = 128,
    n_times: int = 1153,
    n_electrode_groups: int = 1,
    **model_kwargs,
) -> causal1_EEGNet:
    """Build grouped-causal EEGNet for the requested EEG dimensions."""
    return causal1_EEGNet(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        n_electrode_groups=n_electrode_groups,
        **model_kwargs,
    )


if __name__ == "__main__":
    for n_classes in (4, 3):
        model = build_causal1_eegnet(
            n_classes=n_classes,
            n_electrode_groups=8,
        )
        eeg = torch.randn(2, 128, 1153)
        logits = model(eeg)

        if logits.shape != (2, n_classes):
            raise RuntimeError(f"Unexpected logits shape: {logits.shape}")
        if not torch.isfinite(logits).all():
            raise RuntimeError("causal1_EEGNet produced NaN or infinite logits")

        logits.sum().backward()
        if not any(
            parameter.grad is not None
            for parameter in model.parameters()
            if parameter.requires_grad
        ):
            raise RuntimeError(
                "causal1_EEGNet backward pass produced no gradients"
            )

        parameter_count = sum(
            parameter.numel() for parameter in model.parameters()
        )
        print(
            f"causal1_EEGNet test passed: classes={n_classes}, "
            f"input={tuple(eeg.shape)}, output={tuple(logits.shape)}, "
            f"groups={model.n_electrode_groups}, "
            f"group_sizes={model.conv_temporal.electrode_group_sizes}, "
            f"parameters={parameter_count}"
        )

    model.eval()
    temporal_input = torch.randn(1, 1, 128, 256)
    changed_future = temporal_input.clone()
    cutoff = 100
    changed_future[..., cutoff + 1 :] = torch.randn_like(
        changed_future[..., cutoff + 1 :]
    )

    with torch.inference_mode():
        original_output = model.conv_temporal(temporal_input)
        changed_output = model.conv_temporal(changed_future)

    if original_output.shape[-1] != temporal_input.shape[-1]:
        raise RuntimeError("Causal temporal convolution changed time length")
    if not torch.allclose(
        original_output[..., : cutoff + 1],
        changed_output[..., : cutoff + 1],
    ):
        raise RuntimeError("Future samples changed an earlier causal output")

    print("causal temporal convolution test passed")

    sharing_test = _GroupedLeftCausalTemporalConv2d(
        n_chans=5,
        out_channels=1,
        kernel_length=1,
        n_electrode_groups=2,
    )
    with torch.no_grad():
        sharing_test.weight[0].fill_(2.0)
        sharing_test.weight[1].fill_(3.0)
    sharing_output = sharing_test(torch.ones(1, 1, 5, 4))
    if sharing_test.electrode_group_sizes != (3, 2):
        raise RuntimeError("Electrodes were not split into near-equal groups")
    if not torch.allclose(
        sharing_output[:, :, :3],
        torch.full((1, 1, 3, 4), 2.0),
    ):
        raise RuntimeError("Electrodes inside the first group did not share weights")
    if not torch.allclose(
        sharing_output[:, :, 3:],
        torch.full((1, 1, 2, 4), 3.0),
    ):
        raise RuntimeError("Different electrode groups did not use independent weights")

    old_temporal_layer = _GroupedLeftCausalTemporalConv2d(
        n_chans=5,
        out_channels=2,
        kernel_length=3,
        n_electrode_groups=1,
    )
    grouped_temporal_layer = _GroupedLeftCausalTemporalConv2d(
        n_chans=5,
        out_channels=2,
        kernel_length=3,
        n_electrode_groups=2,
    )
    grouped_temporal_layer.load_state_dict(old_temporal_layer.state_dict())
    expected_weight = old_temporal_layer.weight.repeat(2, 1, 1, 1)
    if not torch.equal(grouped_temporal_layer.weight, expected_weight):
        raise RuntimeError("Old temporal weights were not copied to every group")

    print("electrode grouping and checkpoint compatibility tests passed")
