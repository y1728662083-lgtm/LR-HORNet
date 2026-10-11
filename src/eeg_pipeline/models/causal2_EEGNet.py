"""EEGNet variant with causal first and depthwise temporal convolutions."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor, nn

from braindecode.functional import glorot_weight_zero_bias
from braindecode.modules import LinearWithConstraint
from eeg_pipeline.models.causal1_EEGNet import (
    _LeftCausalTemporalConv2d,
    causal1_EEGNet,
)


__all__ = ["causal2_EEGNet", "build_causal2_eegnet"]


class causal2_EEGNet(causal1_EEGNet):
    """EEGNet with both temporal convolutions changed to left-causal ones.

    The first temporal convolution is inherited from causal1_EEGNet.
    This class additionally replaces conv_separable_depth while preserving
    its module position, groups, parameter shape, state-dict key, and initial
    parameter values. Spatial convolution, pointwise convolution, batch
    normalization, activation, pooling, dropout, and output behavior remain
    unchanged.

    The convolutional path is causal when normalization statistics are
    frozen for inference. Training-mode BatchNorm and the whole-window
    classifier are intentionally retained from EEGNet.
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

        previous_final_length = self.final_conv_length

        # Copy the existing depthwise layer exactly, changing only its padding
        # behavior. Reusing the module name preserves order and checkpoint keys.
        old_depth = self.conv_separable_depth
        fork_devices = (
            [old_depth.weight.get_device()] if old_depth.weight.is_cuda else []
        )
        with torch.random.fork_rng(devices=fork_devices):
            causal_depth = _LeftCausalTemporalConv2d(
                in_channels=old_depth.in_channels,
                out_channels=old_depth.out_channels,
                kernel_size=old_depth.kernel_size,
                stride=old_depth.stride,
                padding=0,
                dilation=old_depth.dilation,
                groups=old_depth.groups,
                bias=old_depth.bias is not None,
                padding_mode=old_depth.padding_mode,
                device=old_depth.weight.device,
                dtype=old_depth.weight.dtype,
            )
        causal_depth.load_state_dict(old_depth.state_dict())
        self.add_module("conv_separable_depth", causal_depth)

        causal2_feature_time = self._causal2_feature_time_length()
        if causal2_feature_time < 1:
            minimum_n_times = (
                self.pool1_kernel_size * self.pool2_kernel_size
            )
            raise ValueError(
                f"n_times={self.n_times} is too short for causal2_EEGNet "
                "after the two pooling stages; this configuration requires "
                f"n_times >= {minimum_n_times}."
            )

        # causal1 has already built its automatic classifier. Adjust it only
        # when making the second temporal convolution causal crosses a pooling
        # boundary and changes the feature length.
        if auto_final_length:
            self.final_conv_length = causal2_feature_time
            if previous_final_length != causal2_feature_time:
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

    def _causal2_feature_time_length(self) -> int:
        """Return the time length immediately before the classifier."""
        if self.n_times is None:
            raise ValueError("n_times must be known to build causal2_EEGNet.")

        after_pool1 = self.n_times // self.pool1_kernel_size
        return after_pool1 // self.pool2_kernel_size


def build_causal2_eegnet(
    n_classes: int,
    *,
    n_channels: int = 128,
    n_times: int = 1153,
    **model_kwargs,
) -> causal2_EEGNet:
    """Build causal2_EEGNet for the requested EEG shape and classes."""
    return causal2_EEGNet(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        **model_kwargs,
    )


if __name__ == "__main__":

    def assert_causal(
        module: nn.Module,
        sample: Tensor,
        cutoff: int,
    ) -> None:
        changed_future = sample.clone()
        changed_future[..., cutoff + 1 :] = torch.randn_like(
            changed_future[..., cutoff + 1 :]
        )

        with torch.inference_mode():
            original_output = module(sample)
            changed_output = module(changed_future)

        if original_output.shape[-1] != sample.shape[-1]:
            raise RuntimeError("Causal convolution changed time length")
        if not torch.allclose(
            original_output[..., : cutoff + 1],
            changed_output[..., : cutoff + 1],
        ):
            raise RuntimeError("Future samples changed an earlier causal output")

    for n_classes in (4, 3):
        model = build_causal2_eegnet(n_classes=n_classes)
        eeg = torch.randn(2, 128, 1153)
        logits = model(eeg)

        if logits.shape != (2, n_classes):
            raise RuntimeError(f"Unexpected logits shape: {logits.shape}")
        if not torch.isfinite(logits).all():
            raise RuntimeError("causal2_EEGNet produced NaN or infinite logits")

        logits.sum().backward()
        if not any(
            parameter.grad is not None
            for parameter in model.parameters()
            if parameter.requires_grad
        ):
            raise RuntimeError(
                "causal2_EEGNet backward pass produced no gradients"
            )

        parameter_count = sum(
            parameter.numel() for parameter in model.parameters()
        )
        print(
            f"causal2_EEGNet test passed: classes={n_classes}, "
            f"input={tuple(eeg.shape)}, output={tuple(logits.shape)}, "
            f"parameters={parameter_count}"
        )

    model.eval()
    assert_causal(
        model.conv_temporal,
        torch.randn(1, 1, 128, 256),
        cutoff=100,
    )
    assert_causal(
        model.conv_separable_depth,
        torch.randn(1, 16, 1, 80),
        cutoff=30,
    )
    print("both temporal convolution causality tests passed")
