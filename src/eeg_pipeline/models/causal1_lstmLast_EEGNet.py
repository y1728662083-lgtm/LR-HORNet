"""Causal1 EEGNet classified from the final output of a recurrent head."""

from __future__ import annotations

from typing import Dict, Optional

from torch import Tensor, nn

from braindecode.functional import glorot_weight_zero_bias

from eeg_pipeline.models.causal1_EEGNet import causal1_EEGNet


__all__ = [
    "causal1_lstmLast_EEGNet",
    "build_causal1_lstmlast_eegnet",
]


class _LSTMLastClassificationHead(nn.Module):
    """Classify only the final output of a unidirectional LSTM."""

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

    def forward(self, input: Tensor) -> Tensor:
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
        recurrent_sequence, _ = self.lstm(sequence)
        last_output = recurrent_sequence[:, -1, :]
        return self.classifier(last_output)


class causal1_lstmLast_EEGNet(causal1_EEGNet):
    """Causal1 EEGNet with a final-LSTM-output classification head.

    The causal1 feature extractor is retained unchanged. Its Block 2 output
    ``(B, F2, 1, L2)`` is interpreted as an ``L2``-step sequence with ``F2``
    features. A single-layer forward LSTM processes the sequence, and only its
    final time-step output is passed through ``Linear(hidden_size, classes)``.
    The public output is raw logits shaped ``(B, classes)``; no softmax or
    temporal averaging is applied.

    As in causal1_EEGNet, only the first temporal convolution is causal. The
    second temporal convolution retains symmetric padding, so the complete
    EEG-to-logits model is not strictly causal at every intermediate time.
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
                "causal1_lstmLast_EEGNet replaces the convolutional "
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

        # Follow the full causal1 automatic-head construction path before
        # replacing final_layer. This preserves every trunk tensor under the
        # same initialization seed as the causal1_EEGNet baseline.
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

        # The parent assigns a legacy mapping to its convolutional classifier
        # during construction. That target no longer exists in this model.
        self.mapping = {}

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
        self.final_layer = _LSTMLastClassificationHead(
            input_size=self.F2,
            hidden_size=lstm_hidden_size,
            n_outputs=self.n_outputs,
            device=classifier_weight.device,
            dtype=classifier_weight.dtype,
        )


def build_causal1_lstmlast_eegnet(
    n_classes: int,
    *,
    n_channels: int = 128,
    n_times: int = 1153,
    **model_kwargs,
) -> causal1_lstmLast_EEGNet:
    """Build the final-LSTM-output causal1 EEGNet variant."""
    return causal1_lstmLast_EEGNet(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        **model_kwargs,
    )


if __name__ == "__main__":
    import torch

    expected_parameters = {4: 5396, 3: 5379}
    for n_classes, expected_count in expected_parameters.items():
        model = build_causal1_lstmlast_eegnet(n_classes=n_classes).eval()
        captured_recurrent_sequences = []
        captured_classifier_inputs = []
        lstm_handle = model.final_layer.lstm.register_forward_hook(
            lambda _module, _inputs, output: (
                captured_recurrent_sequences.append(output[0])
            )
        )
        classifier_handle = model.final_layer.classifier.register_forward_pre_hook(
            lambda _module, inputs: captured_classifier_inputs.append(inputs[0])
        )

        eeg = torch.randn(2, 128, 1153, requires_grad=True)
        logits = model(eeg)
        lstm_handle.remove()
        classifier_handle.remove()

        if logits.shape != (2, n_classes):
            raise RuntimeError(f"Unexpected logits shape: {logits.shape}")
        if len(captured_recurrent_sequences) != 1:
            raise RuntimeError("Expected exactly one LSTM hook output")
        if len(captured_classifier_inputs) != 1:
            raise RuntimeError("Expected exactly one classifier hook input")

        recurrent_sequence = captured_recurrent_sequences[0]
        classifier_input = captured_classifier_inputs[0]
        if recurrent_sequence.shape != (2, 36, 16):
            raise RuntimeError(
                f"Unexpected recurrent shape: {recurrent_sequence.shape}"
            )
        if classifier_input.shape != (2, 16):
            raise RuntimeError(
                f"Unexpected classifier input shape: {classifier_input.shape}"
            )
        if not torch.equal(classifier_input, recurrent_sequence[:, -1, :]):
            raise RuntimeError("Classifier did not receive the final LSTM output")
        if not torch.isfinite(logits).all():
            raise RuntimeError(
                "causal1_lstmLast_EEGNet produced NaN or infinite logits"
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
            "causal1_lstmLast_EEGNet test passed: "
            f"classes={n_classes}, input={tuple(eeg.shape)}, "
            f"recurrent={tuple(recurrent_sequence.shape)}, "
            f"classifier_input={tuple(classifier_input.shape)}, "
            f"output={tuple(logits.shape)}, parameters={parameter_count}"
        )
