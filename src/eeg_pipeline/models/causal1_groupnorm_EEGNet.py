"""causal1_EEGNet with its three BatchNorm layers replaced by GroupNorm."""

from __future__ import annotations

import operator

import torch
from torch import nn

from eeg_pipeline.models.causal1_EEGNet import causal1_EEGNet


__all__ = [
    "causal1_groupnorm_EEGNet",
    "build_causal1_groupnorm_eegnet",
]


def _positive_integer(name: str, value: int) -> int:
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


def _group_norm_from_batch_norm(
    name: str,
    batch_norm: nn.BatchNorm2d,
    n_norm_groups: int,
) -> nn.GroupNorm:
    n_channels = batch_norm.num_features
    if n_channels % n_norm_groups != 0:
        raise ValueError(
            f"{name} has {n_channels} feature channels, which is not "
            f"divisible by n_norm_groups={n_norm_groups}."
        )
    reference = batch_norm.running_mean
    factory_kwargs = {}
    if reference is not None:
        factory_kwargs = {
            "device": reference.device,
            "dtype": reference.dtype,
        }
    return nn.GroupNorm(
        num_groups=n_norm_groups,
        num_channels=n_channels,
        eps=batch_norm.eps,
        affine=batch_norm.affine,
        **factory_kwargs,
    )


class causal1_groupnorm_EEGNet(causal1_EEGNet):
    """Replace only causal1_EEGNet's three BatchNorm layers with GroupNorm."""

    def __init__(
        self,
        *args,
        n_norm_groups: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        n_norm_groups = _positive_integer("n_norm_groups", n_norm_groups)

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
                "n_norm_groups must divide every normalized feature count; "
                f"got n_norm_groups={n_norm_groups} and {', '.join(invalid)}."
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
        self.n_norm_groups = n_norm_groups


def build_causal1_groupnorm_eegnet(
    n_classes: int,
    *,
    n_channels: int = 128,
    n_times: int = 1153,
    **model_kwargs,
) -> causal1_groupnorm_EEGNet:
    """Build causal1_EEGNet with GroupNorm in all three norm positions."""
    return causal1_groupnorm_EEGNet(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        **model_kwargs,
    )


if __name__ == "__main__":
    model = build_causal1_groupnorm_eegnet(n_classes=4)
    eeg = torch.randn(2, 128, 1153, requires_grad=True)
    logits = model(eeg)
    if logits.shape != (2, 4):
        raise RuntimeError(f"Unexpected logits shape: {logits.shape}")
    if any(isinstance(module, nn.BatchNorm2d) for module in model.modules()):
        raise RuntimeError("GroupNorm model still contains BatchNorm2d")
    if sum(isinstance(module, nn.GroupNorm) for module in model.modules()) != 3:
        raise RuntimeError("GroupNorm model must contain exactly three GroupNorms")
    logits.sum().backward()
    print("causal1_groupnorm_EEGNet self-test passed")
