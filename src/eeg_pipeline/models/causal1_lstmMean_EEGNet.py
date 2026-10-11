"""Causal1 EEGNet with a recurrent temporal-mean classification head."""

from __future__ import annotations

from typing import Dict, Optional

from torch import Tensor, nn

from braindecode.functional import glorot_weight_zero_bias

from eeg_pipeline.models.causal1_EEGNet import causal1_EEGNet


__all__ = [
    "causal1_lstmMean_EEGNet",
    "build_causal1_lstmmean_eegnet",
]


class _LSTMMeanClassificationHead(nn.Module):
    """Classify every recurrent time step and average the raw logits."""

    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        n_outputs: int,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_outputs = n_outputs

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
            **factory_kwargs,
        )
        self.classifier = nn.Linear(
            hidden_size,
            n_outputs,
            bias=True,
            **factory_kwargs,
        )
        glorot_weight_zero_bias(self.classifier)

    def forward_time_logits(self, input: Tensor) -> Tensor:
        """Return raw per-time logits shaped ``(B, classes, time)``."""
        if input.ndim != 4:
            raise RuntimeError(
                "LSTM head expects features shaped (B, features, 1, time), "
                f"got {tuple(input.shape)}."
            )
        if input.shape[1] != self.input_size:
            raise RuntimeError(
                f"LSTM head expects {self.input_size} features, "
                f"got {input.shape[1]}."
            )
        if input.shape[-2] != 1:
            raise RuntimeError(
                "LSTM head expects one virtual spatial channel, "
                f"got shape {tuple(input.shape)}."
            )
        if input.shape[-1] < 1:
            raise RuntimeError("LSTM head requires at least one time step.")

        sequence = input.squeeze(-2).transpose(1, 2).contiguous()
        recurrent_features, _ = self.lstm(sequence)
        time_logits = self.classifier(recurrent_features)
        return time_logits.transpose(1, 2).contiguous()

    def forward(self, input: Tensor) -> Tensor:
        time_logits = self.forward_time_logits(input)
        mean_logits = time_logits.mean(dim=-1, keepdim=True)
        return mean_logits.squeeze(-1)


class causal1_lstmMean_EEGNet(causal1_EEGNet):
    """Causal1 EEGNet with a single-layer unidirectional LSTM head.

    The causal1 feature extractor is retained unchanged. Its Block 2 output
    ``(B, F2, 1, L2)`` is interpreted as an ``L2``-step sequence with ``F2``
    features. A one-layer forward LSTM produces recurrent features at every
    step, a shared linear classifier produces raw per-time class logits, and
    those logits are averaged over time. The public output is ``(B, classes)``
    and contains no softmax.

    As in causal1_EEGNet, only the first temporal convolution is causal. The
    second temporal convolution keeps symmetric padding, and the final mean
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
        lstm_hidden_size: int = 16,
        # Alternative signal metadata
        chs_info: Optional[list[Dict]] = None,
        input_window_seconds=None,
        sfreq=None,
        **kwargs,
    ) -> None:
        if final_conv_length != "auto":
            raise ValueError(
                "causal1_lstmMean_EEGNet replaces the convolutional "
                "classifier and only accepts final_conv_length='auto', got "
                f"{final_conv_length!r}."
            )
        if final_layer_with_constraint:
            raise ValueError(
                "final_layer_with_constraint=True is incompatible with the "
                "LSTM classification head."
            )
        if type(lstm_hidden_size) is not int or lstm_hidden_size < 1:
            raise ValueError(
                "lstm_hidden_size must be a positive integer, got "
                f"{lstm_hidden_size!r}."
            )

        # Construct the complete causal1 parent, including its automatic head,
        # before replacing final_layer. This preserves every trunk tensor under
        # the same initialization seed as the causal1_EEGNet baseline.
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

        parent_final_layer = self.final_layer
        expected_modules = ("conv_classifier", "permute_back", "squeeze")
        actual_modules = tuple(parent_final_layer._modules)
        if actual_modules != expected_modules:
            raise RuntimeError(
                "Unexpected EEGNet final_layer structure: expected "
                f"{expected_modules}, got {actual_modules}."
            )

        classifier_weight = parent_final_layer.conv_classifier.weight
        self.lstm_hidden_size = lstm_hidden_size
        self.configured_feature_time_length = self.final_conv_length
        self.final_layer = _LSTMMeanClassificationHead(
            input_size=self.F2,
            hidden_size=lstm_hidden_size,
            n_outputs=self.n_outputs,
            device=classifier_weight.device,
            dtype=classifier_weight.dtype,
        )

    def forward_time_logits(self, input: Tensor) -> Tensor:
        """Return raw per-time logits shaped ``(B, classes, L2)``."""
        features = input
        for name, module in self._modules.items():
            if name == "final_layer":
                break
            features = module(features)
        return self.final_layer.forward_time_logits(features)


def build_causal1_lstmmean_eegnet(
    n_classes: int,
    *,
    n_channels: int = 128,
    n_times: int = 1153,
    **model_kwargs,
) -> causal1_lstmMean_EEGNet:
    """Build the single-layer unidirectional LSTM causal1 variant."""
    return causal1_lstmMean_EEGNet(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        **model_kwargs,
    )


if __name__ == "__main__":
    import torch

    expected_parameters = {4: 5396, 3: 5379}
    for n_classes, expected_count in expected_parameters.items():
        model = build_causal1_lstmmean_eegnet(n_classes=n_classes).eval()
        captured_logits = []
        handle = model.final_layer.classifier.register_forward_hook(
            lambda _module, _inputs, output: captured_logits.append(output)
        )

        eeg = torch.randn(2, 128, 1153, requires_grad=True)
        logits = model(eeg)
        handle.remove()

        if len(captured_logits) != 1:
            raise RuntimeError("Expected exactly one classifier hook output")
        time_logits = captured_logits[0].transpose(1, 2).contiguous()
        if time_logits.shape != (2, n_classes, 36):
            raise RuntimeError(
                f"Unexpected per-time logits shape: {time_logits.shape}"
            )
        if logits.shape != (2, n_classes):
            raise RuntimeError(f"Unexpected logits shape: {logits.shape}")
        if not torch.equal(logits, time_logits.mean(dim=-1)):
            raise RuntimeError("Public logits do not equal mean time logits")
        if not torch.isfinite(logits).all():
            raise RuntimeError(
                "causal1_lstmMean_EEGNet produced NaN or infinite logits"
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
            "causal1_lstmMean_EEGNet test passed: "
            f"classes={n_classes}, input={tuple(eeg.shape)}, "
            f"time_logits={tuple(time_logits.shape)}, "
            f"output={tuple(logits.shape)}, parameters={parameter_count}"
        )

    head = model.final_layer.eval()
    features = torch.randn(1, model.F2, 1, 12)
    changed_future = features.clone()
    cutoff = 5
    changed_future[..., cutoff + 1 :] = torch.randn_like(
        changed_future[..., cutoff + 1 :]
    )
    with torch.inference_mode():
        original_time_logits = head.forward_time_logits(features)
        changed_time_logits = head.forward_time_logits(changed_future)
    if not torch.allclose(
        original_time_logits[..., : cutoff + 1],
        changed_time_logits[..., : cutoff + 1],
    ):
        raise RuntimeError("Future features changed an earlier LSTM output")

    print("single-layer unidirectional LSTM prefix test passed")
