"""Causal1 EEGNet with overlapping grouped spatial windows.

The first temporal convolution remains causal, exactly as in
``causal1_EEGNet``. The full-channel spatial depthwise convolution is
replaced by three shared, 50%-overlapping spatial windows. After Block 1,
the window axis is folded into the feature axis and fused by a grouped
temporal convolution.
"""

from __future__ import annotations

from typing import Dict, Optional

from torch import Tensor, nn

from braindecode.modules import Conv2dWithConstraint

from eeg_pipeline.models.causal1_EEGNet import causal1_EEGNet


__all__ = [
    "causal1_spaceGroup_EEGNet",
    "build_causal1_spacegroup_eegnet",
]


class _FoldSpatialThenGroupedTemporalConv2d(nn.Conv2d):
    """Fold spatial windows feature-first, then apply grouped temporal conv.

    Input ``(B, A, H, T)`` is folded to ``(B, A * H, 1, T)`` in the
    order ``a0h0, a0h1, ..., a1h0, ...``. With ``groups=A`` each group
    therefore receives all spatial windows belonging to one feature.
    """

    def __init__(
        self,
        *,
        n_features: int,
        n_spatial_windows: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
        dilation: tuple[int, int],
        bias: bool,
        padding_mode: str,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            in_channels=n_features * n_spatial_windows,
            out_channels=n_features,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=n_features,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        self.n_features = n_features
        self.n_spatial_windows = n_spatial_windows

    def forward(self, input: Tensor) -> Tensor:
        if input.ndim != 4:
            raise RuntimeError(
                "grouped temporal convolution expects a 4D tensor "
                f"(B, features, spatial, time), got shape {tuple(input.shape)}"
            )

        batch, features, spatial, time = input.shape
        if features != self.n_features:
            raise RuntimeError(
                f"expected {self.n_features} features before folding, "
                f"got {features}"
            )
        if spatial != self.n_spatial_windows:
            raise RuntimeError(
                f"expected {self.n_spatial_windows} spatial windows, "
                f"got {spatial}"
            )

        # Feature-major folding keeps one feature's three windows contiguous,
        # which is required for groups=n_features to have the intended meaning.
        input = input.reshape(batch, features * spatial, 1, time)
        return self._conv_forward(input, self.weight, self.bias)


class causal1_spaceGroup_EEGNet(causal1_EEGNet):
    """Causal1 EEGNet with three overlapping grouped spatial windows.

    For ``C`` input channels, the spatial depthwise convolution uses
    ``kernel_size=(C/2, 1)`` and ``stride=(C/4, 1)``. This produces three
    50%-overlapping windows. With the default ``C=128`` they cover channels
    ``0..63``, ``32..95``, and ``64..127``.

    Block 1 remains in its original order. Its output ``(B, A, 3, L1)`` is
    folded to ``(B, 3A, 1, L1)`` immediately before the second temporal
    convolution, where ``A=F1*D``. That convolution uses ``3A -> A`` and
    ``groups=A``, so each group fuses the three windows belonging to one
    feature. It is a grouped temporal convolution rather than a strict
    depthwise convolution.

    As in causal1_EEGNet, only the first temporal convolution is causal. The
    second temporal convolution keeps Braindecode EEGNet's symmetric padding.
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
    ) -> None:
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

        if self.n_chans is None:
            raise ValueError(
                "n_chans must be known to build causal1_spaceGroup_EEGNet."
            )
        if self.n_chans % 4 != 0:
            raise ValueError(
                "n_chans must be divisible by 4 for the three overlapping "
                f"spatial windows, got n_chans={self.n_chans}."
            )

        n_features = self.F1 * self.D
        self.spatial_kernel_size = self.n_chans // 2
        self.spatial_stride = self.n_chans // 4
        self.n_spatial_windows = 3

        old_spatial = self.conv_spatial
        self.conv_spatial = Conv2dWithConstraint(
            in_channels=self.F1,
            out_channels=n_features,
            kernel_size=(self.spatial_kernel_size, 1),
            stride=(self.spatial_stride, 1),
            padding=0,
            groups=self.F1,
            bias=False,
            max_norm=self.conv_spatial_max_norm,
            device=old_spatial.weight.device,
            dtype=old_spatial.weight.dtype,
        )

        old_depth = self.conv_separable_depth
        self.conv_separable_depth = _FoldSpatialThenGroupedTemporalConv2d(
            n_features=n_features,
            n_spatial_windows=self.n_spatial_windows,
            kernel_size=old_depth.kernel_size,
            stride=old_depth.stride,
            padding=old_depth.padding,
            dilation=old_depth.dilation,
            bias=old_depth.bias is not None,
            padding_mode=old_depth.padding_mode,
            device=old_depth.weight.device,
            dtype=old_depth.weight.dtype,
        )


def build_causal1_spacegroup_eegnet(
    n_classes: int,
    *,
    n_channels: int = 128,
    n_times: int = 1153,
    **model_kwargs,
) -> causal1_spaceGroup_EEGNet:
    """Build the grouped-spatial causal1 EEGNet variant."""
    return causal1_spaceGroup_EEGNet(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        **model_kwargs,
    )


if __name__ == "__main__":
    import torch

    expected_parameters = {4: 4948, 3: 4371}
    for n_classes, expected_count in expected_parameters.items():
        model = build_causal1_spacegroup_eegnet(n_classes=n_classes)
        eeg = torch.randn(2, 128, 1153, requires_grad=True)
        logits = model(eeg)

        if logits.shape != (2, n_classes):
            raise RuntimeError(f"Unexpected logits shape: {logits.shape}")
        if not torch.isfinite(logits).all():
            raise RuntimeError(
                "causal1_spaceGroup_EEGNet produced NaN or infinite logits"
            )

        logits.square().mean().backward()
        if eeg.grad is None or not torch.isfinite(eeg.grad).all():
            raise RuntimeError("Input gradient is missing or non-finite")

        parameter_count = sum(
            parameter.numel() for parameter in model.parameters()
        )
        if parameter_count != expected_count:
            raise RuntimeError(
                f"Expected {expected_count} parameters, got {parameter_count}"
            )

        print(
            "causal1_spaceGroup_EEGNet test passed: "
            f"classes={n_classes}, input={tuple(eeg.shape)}, "
            f"output={tuple(logits.shape)}, parameters={parameter_count}"
        )

    model = build_causal1_spacegroup_eegnet(n_classes=4)
    if model.conv_spatial.kernel_size != (64, 1):
        raise RuntimeError("Unexpected spatial kernel size")
    if model.conv_spatial.stride != (32, 1):
        raise RuntimeError("Unexpected spatial stride")
    if model.conv_separable_depth.in_channels != 48:
        raise RuntimeError("Unexpected grouped temporal input channels")
    if model.conv_separable_depth.out_channels != 16:
        raise RuntimeError("Unexpected grouped temporal output channels")
    if model.conv_separable_depth.groups != 16:
        raise RuntimeError("Unexpected grouped temporal groups")

    print("grouped spatial structure test passed")
