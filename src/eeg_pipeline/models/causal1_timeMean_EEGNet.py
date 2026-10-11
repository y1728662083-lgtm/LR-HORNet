"""Causal1 EEGNet with shared per-time logits and temporal mean pooling."""

from __future__ import annotations

from typing import Dict, Optional

from torch import Tensor, nn

from braindecode.functional import glorot_weight_zero_bias

from eeg_pipeline.models.causal1_EEGNet import causal1_EEGNet


__all__ = [
    "causal1_timeMean_EEGNet",
    "build_causal1_timemean_eegnet",
]


class _TemporalMean(nn.Module):
    """Average raw class logits over time while retaining four dimensions."""

    def forward(self, input: Tensor) -> Tensor:
        if input.ndim != 4:
            raise RuntimeError(
                "temporal mean expects logits shaped "
                f"(B, classes, 1, time), got {tuple(input.shape)}"
            )
        if input.shape[-2] != 1:
            raise RuntimeError(
                "temporal mean expects one virtual spatial channel, "
                f"got shape {tuple(input.shape)}"
            )
        return input.mean(dim=-1, keepdim=True)


class causal1_timeMean_EEGNet(causal1_EEGNet):
    """Causal1 EEGNet with a shared pointwise temporal-mean classifier.

    The causal1 feature extractor is retained unchanged. Given its final
    feature map ``(B, F2, 1, L2)``, a shared ``1 x 1`` convolution produces
    raw per-time logits ``(B, n_outputs, 1, L2)``. Those logits are averaged
    over time before Braindecode's usual permute-and-squeeze output handling,
    so the public output remains ``(B, n_outputs)``.

    Because the same classifier is used at every time step, this head is
    mathematically equivalent to temporal mean pooling of the F2 features
    followed by a linear classifier. No softmax is applied.

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
        valid_pointwise_length = (
            type(final_conv_length) is int and final_conv_length == 1
        )
        if not (valid_auto_length or valid_pointwise_length):
            raise ValueError(
                "causal1_timeMean_EEGNet uses a fixed 1x1 classifier and "
                "only accepts final_conv_length='auto' or 1, got "
                f"{final_conv_length!r}."
            )
        if final_layer_with_constraint:
            raise ValueError(
                "final_layer_with_constraint=True is incompatible with the "
                "shared 1x1 temporal-mean classifier."
            )

        # Follow causal1_EEGNet's automatic-head construction path first. This
        # keeps every trunk parameter, including the causal temporal layer that
        # the parent replaces last, identical under the same random seed. The
        # full-time classifier is replaced below by the requested shared head.
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

        # Keep the parent's permutation and squeeze modules. Replacing the
        # classifier under the same name preserves its state-dict key.
        parent_final_layer = self.final_layer
        expected_modules = ("conv_classifier", "permute_back", "squeeze")
        actual_modules = tuple(parent_final_layer._modules)
        if actual_modules != expected_modules:
            raise RuntimeError(
                "Unexpected EEGNet final_layer structure: expected "
                f"{expected_modules}, got {actual_modules}."
            )

        pointwise_classifier = nn.Conv2d(
            in_channels=self.F2,
            out_channels=self.n_outputs,
            kernel_size=(1, 1),
            bias=True,
            device=parent_final_layer.conv_classifier.weight.device,
            dtype=parent_final_layer.conv_classifier.weight.dtype,
        )
        glorot_weight_zero_bias(pointwise_classifier)
        self.final_conv_length = 1

        final_layer = nn.Sequential()
        final_layer.add_module(
            "conv_classifier", pointwise_classifier
        )
        final_layer.add_module("time_mean", _TemporalMean())
        final_layer.add_module("permute_back", parent_final_layer.permute_back)
        final_layer.add_module("squeeze", parent_final_layer.squeeze)
        self.final_layer = final_layer


def build_causal1_timemean_eegnet(
    n_classes: int,
    *,
    n_channels: int = 128,
    n_times: int = 1153,
    **model_kwargs,
) -> causal1_timeMean_EEGNet:
    """Build the temporal-mean causal1 EEGNet variant."""
    return causal1_timeMean_EEGNet(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        **model_kwargs,
    )


if __name__ == "__main__":
    import torch

    expected_parameters = {4: 3220, 3: 3203}
    for n_classes, expected_count in expected_parameters.items():
        model = build_causal1_timemean_eegnet(n_classes=n_classes)
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
                "causal1_timeMean_EEGNet produced NaN or infinite logits"
            )
        if len(captured_logits) != 1:
            raise RuntimeError("Expected exactly one classifier hook output")

        raw_time_logits = captured_logits[0]
        expected_logits = raw_time_logits.mean(dim=-1).squeeze(-1)
        if raw_time_logits.shape != (2, n_classes, 1, 36):
            raise RuntimeError(
                f"Unexpected per-time logits shape: {raw_time_logits.shape}"
            )
        if not torch.allclose(logits, expected_logits):
            raise RuntimeError("Public logits do not equal mean time logits")

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
            "causal1_timeMean_EEGNet test passed: "
            f"classes={n_classes}, input={tuple(eeg.shape)}, "
            f"time_logits={tuple(raw_time_logits.shape)}, "
            f"output={tuple(logits.shape)}, parameters={parameter_count}"
        )

    # A 1x1 shared classifier followed by a mean accepts runtime time lengths
    # that differ from the configured n_times, as long as the trunk is valid.
    model = build_causal1_timemean_eegnet(n_classes=4).eval()
    with torch.inference_mode():
        for runtime_time in (512, 1153, 1200):
            logits = model(torch.randn(1, 128, runtime_time))
            if logits.shape != (1, 4):
                raise RuntimeError(
                    f"Unexpected output for T={runtime_time}: {logits.shape}"
                )

    print("variable-time temporal-mean classifier test passed")
