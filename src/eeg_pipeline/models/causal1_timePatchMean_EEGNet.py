"""Causal1 EEGNet with shared local-time patch logits and mean pooling."""

from __future__ import annotations

from typing import Dict, Optional

from torch import Tensor, nn

from braindecode.functional import glorot_weight_zero_bias

from eeg_pipeline.models.causal1_EEGNet import causal1_EEGNet


__all__ = [
    "causal1_timePatchMean_EEGNet",
    "build_causal1_timepatchmean_eegnet",
]


class _LocalTimePatchClassifier(nn.Conv2d):
    """Classify shared local time patches with an explicit length check."""

    def forward(self, input: Tensor) -> Tensor:
        if input.ndim != 4:
            raise RuntimeError(
                "local time-patch classifier expects "
                f"(B, features, 1, time), got {tuple(input.shape)}"
            )
        if input.shape[-2] != 1:
            raise RuntimeError(
                "local time-patch classifier expects one virtual spatial "
                f"channel, got shape {tuple(input.shape)}"
            )
        required_time = self.kernel_size[-1]
        if input.shape[-1] < required_time:
            raise ValueError(
                "local time-patch classifier requires at least "
                f"{required_time} pooled time steps, got {input.shape[-1]}."
            )
        return self._conv_forward(input, self.weight, self.bias)


class _TemporalMean(nn.Module):
    """Average raw local-window logits over their time axis."""

    def forward(self, input: Tensor) -> Tensor:
        if input.ndim != 4:
            raise RuntimeError(
                "temporal mean expects logits shaped "
                f"(B, classes, 1, patches), got {tuple(input.shape)}"
            )
        if input.shape[-2] != 1:
            raise RuntimeError(
                "temporal mean expects one virtual spatial channel, "
                f"got shape {tuple(input.shape)}"
            )
        return input.mean(dim=-1, keepdim=True)


class causal1_timePatchMean_EEGNet(causal1_EEGNet):
    """Causal1 EEGNet with a shared local-time classifier and logit mean.

    The causal1 feature extractor is retained unchanged. Let its configured
    output be ``(B, F2, 1, L2)``. By default, the classifier uses

    ``kernel_time = max(1, L2 // 8)`` and
    ``stride_time = max(1, L2 // 16)``.

    It produces shared local-window logits ``(B, n_outputs, 1, M)`` with
    valid convolution, where ``M = floor((L_runtime-K)/S)+1``. Raw logits are
    averaged over M before Braindecode's usual output permutation and squeeze.
    No softmax is applied.

    With the default ``n_times=1153``, ``L2=36``, ``K=4``, ``S=2``, and
    ``M=17``. These windows cover all 36 configured pooled time steps. For
    other runtime lengths, valid convolution may leave up to ``S-1`` trailing
    pooled steps unused.

    As in causal1_EEGNet, only the first temporal convolution is causal. The
    second temporal convolution retains symmetric padding, and the final mean
    requires the complete input window.
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
        valid_auto_length = final_conv_length == "auto"
        valid_explicit_length = (
            type(final_conv_length) is int and final_conv_length >= 1
        )
        if not (valid_auto_length or valid_explicit_length):
            raise ValueError(
                "causal1_timePatchMean_EEGNet expects "
                "final_conv_length='auto' or a positive integer, got "
                f"{final_conv_length!r}."
            )
        if final_layer_with_constraint:
            raise ValueError(
                "final_layer_with_constraint=True is incompatible with the "
                "local time-patch mean classifier."
            )

        # Follow causal1_EEGNet's automatic-head construction path first so
        # every trunk parameter is identical under the same random seed. The
        # original full-time classifier is replaced after the parent returns.
        super().__init__(
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_times=n_times,
            final_conv_length="auto",
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
            final_layer_with_constraint=False,
            norm_rate=norm_rate,
            chs_info=chs_info,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
            **kwargs,
        )

        self.configured_feature_time_length = self.final_conv_length
        if valid_auto_length:
            classifier_kernel_length = max(
                1, self.configured_feature_time_length // 8
            )
        else:
            classifier_kernel_length = final_conv_length

        if classifier_kernel_length > self.configured_feature_time_length:
            raise ValueError(
                "classifier kernel cannot exceed the configured pooled time "
                f"length {self.configured_feature_time_length}, got "
                f"{classifier_kernel_length}."
            )

        classifier_stride = max(
            1, self.configured_feature_time_length // 16
        )
        self.classifier_kernel_length = classifier_kernel_length
        self.classifier_stride = classifier_stride
        self.classifier_patch_count = (
            (
                self.configured_feature_time_length
                - self.classifier_kernel_length
            )
            // self.classifier_stride
            + 1
        )

        parent_final_layer = self.final_layer
        expected_modules = ("conv_classifier", "permute_back", "squeeze")
        actual_modules = tuple(parent_final_layer._modules)
        if actual_modules != expected_modules:
            raise RuntimeError(
                "Unexpected EEGNet final_layer structure: expected "
                f"{expected_modules}, got {actual_modules}."
            )

        local_classifier = _LocalTimePatchClassifier(
            in_channels=self.F2,
            out_channels=self.n_outputs,
            kernel_size=(1, self.classifier_kernel_length),
            stride=(1, self.classifier_stride),
            padding=0,
            bias=True,
            device=parent_final_layer.conv_classifier.weight.device,
            dtype=parent_final_layer.conv_classifier.weight.dtype,
        )
        glorot_weight_zero_bias(local_classifier)
        self.final_conv_length = self.classifier_kernel_length

        final_layer = nn.Sequential()
        final_layer.add_module("conv_classifier", local_classifier)
        final_layer.add_module("time_mean", _TemporalMean())
        final_layer.add_module("permute_back", parent_final_layer.permute_back)
        final_layer.add_module("squeeze", parent_final_layer.squeeze)
        self.final_layer = final_layer


def build_causal1_timepatchmean_eegnet(
    n_classes: int,
    *,
    n_channels: int = 128,
    n_times: int = 1153,
    **model_kwargs,
) -> causal1_timePatchMean_EEGNet:
    """Build the local time-patch mean causal1 EEGNet variant."""
    return causal1_timePatchMean_EEGNet(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        **model_kwargs,
    )


if __name__ == "__main__":
    import torch

    expected_parameters = {4: 3412, 3: 3347}
    for n_classes, expected_count in expected_parameters.items():
        model = build_causal1_timepatchmean_eegnet(n_classes=n_classes)
        captured_logits = []
        handle = model.final_layer.conv_classifier.register_forward_hook(
            lambda _module, _inputs, output: captured_logits.append(output)
        )

        eeg = torch.randn(2, 128, 1153, requires_grad=True)
        logits = model(eeg)
        handle.remove()

        if logits.shape != (2, n_classes):
            raise RuntimeError(f"Unexpected logits shape: {logits.shape}")
        if not torch.isfinite(logits).all():
            raise RuntimeError(
                "causal1_timePatchMean_EEGNet produced non-finite logits"
            )
        if len(captured_logits) != 1:
            raise RuntimeError("Expected exactly one classifier hook output")

        raw_patch_logits = captured_logits[0]
        expected_logits = raw_patch_logits.mean(dim=-1).squeeze(-1)
        expected_raw_shape = (2, n_classes, 1, 17)
        if raw_patch_logits.shape != expected_raw_shape:
            raise RuntimeError(
                "Unexpected local-patch logits shape: "
                f"{raw_patch_logits.shape}"
            )
        if not torch.equal(logits, expected_logits):
            raise RuntimeError("Public logits do not equal mean patch logits")

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
            "causal1_timePatchMean_EEGNet test passed: "
            f"classes={n_classes}, input={tuple(eeg.shape)}, "
            f"patch_logits={tuple(raw_patch_logits.shape)}, "
            f"output={tuple(logits.shape)}, parameters={parameter_count}"
        )

    model = build_causal1_timepatchmean_eegnet(n_classes=4).eval()
    with torch.inference_mode():
        for runtime_time in (124, 512, 1153, 1200):
            logits = model(torch.randn(1, 128, runtime_time))
            if logits.shape != (1, 4):
                raise RuntimeError(
                    f"Unexpected output for T={runtime_time}: {logits.shape}"
                )

    print("variable-time local-patch classifier test passed")
