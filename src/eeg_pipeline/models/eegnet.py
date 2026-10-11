"""Braindecode EEGNet model construction."""

from __future__ import annotations

from braindecode.models import EEGNet


def build_eegnet(
    n_classes: int,
    *,
    n_channels: int = 128,
    n_times: int = 1153,
    **model_kwargs,
) -> EEGNet:
    """Build Braindecode EEGNet for the requested EEG shape and classes."""
    return EEGNet(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        **model_kwargs,
    )


if __name__ == "__main__":
    import torch

    for n_classes in (4, 3):
        model = build_eegnet(n_classes=n_classes)
        eeg = torch.randn(2, 128, 1153)
        logits = model(eeg)

        if logits.shape != (2, n_classes):
            raise RuntimeError(f"Unexpected logits shape: {logits.shape}")
        if not torch.isfinite(logits).all():
            raise RuntimeError("EEGNet produced NaN or infinite logits")

        logits.sum().backward()
        if not any(
            parameter.grad is not None
            for parameter in model.parameters()
            if parameter.requires_grad
        ):
            raise RuntimeError("EEGNet backward pass produced no gradients")

        parameter_count = sum(
            parameter.numel() for parameter in model.parameters()
        )
        print(
            f"EEGNet test passed: classes={n_classes}, "
            f"input={tuple(eeg.shape)}, output={tuple(logits.shape)}, "
            f"parameters={parameter_count}"
        )
