"""Deformable Patch-CfC with a training-only TF-C auxiliary branch.

The classification path is exactly ``deformable_patchCfC``. During training,
the original and augmented CfC1 outputs are contrasted with cached frequency-
band features using independent pointwise projectors. Evaluation and inference
call ``forward(eeg)`` and therefore do not execute the frequency branch.
"""

from __future__ import annotations

import operator
from typing import NamedTuple

import torch
from torch import Tensor, nn

from eeg_pipeline.models.deformable_patchCfC import (
    deformable_patchCfC,
)


__all__ = [
    "FilterNFCTrainingOutput",
    "deformable_patchCfC_filterNFC",
    "build_deformable_patchcfc_filternfc",
]


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        parsed = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


class FilterNFCTrainingOutput(NamedTuple):
    """Representations required by the TF-C auxiliary training objective.

    h_time_augmented has shape (B,F,C,P) for the full compatibility path, or
    (B,F,S) when time positions were selected before augmented Patch-CfC.
    """

    logits: Tensor
    h_time: Tensor
    h_time_augmented: Tensor
    h_frequency: Tensor
    h_frequency_augmented: Tensor
    z_time: Tensor
    z_frequency: Tensor


class _PointwiseProjector(nn.Sequential):
    """TF-C two-layer projector adapted to structured EEG feature maps."""

    def __init__(
        self,
        input_features: int,
        hidden_features: int,
        output_features: int,
        *,
        batch_norm_momentum: float,
        batch_norm_eps: float,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                input_features,
                hidden_features,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                hidden_features,
                momentum=batch_norm_momentum,
                eps=batch_norm_eps,
            ),
            nn.ReLU(),
            nn.Conv2d(
                hidden_features,
                output_features,
                kernel_size=1,
                bias=True,
            ),
        )


class deformable_patchCfC_filterNFC(deformable_patchCfC):
    """Add training-only structured time-frequency contrast to the backbone."""

    def __init__(
        self,
        *args,
        patch_size: int = 64,
        patch_stride: int = 4,
        n_electrode_groups: int = 16,
        n_frequency_bands: int = 8,
        projection_hidden_size: int = 16,
        projection_size: int = 8,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            patch_size=patch_size,
            patch_stride=patch_stride,
            n_electrode_groups=n_electrode_groups,
            **kwargs,
        )
        self.n_frequency_bands = _positive_integer(
            "n_frequency_bands", n_frequency_bands
        )
        projection_hidden_size = _positive_integer(
            "projection_hidden_size", projection_hidden_size
        )
        projection_size = _positive_integer(
            "projection_size", projection_size
        )

        # Capture the inherited Sequential path before registering auxiliary
        # modules. This guarantees forward(eeg) remains the original backbone.
        self._backbone_module_names = tuple(self._modules.keys())

        self.frequency_encoder = nn.Sequential(
            nn.Conv2d(
                self.n_frequency_bands,
                self.F1,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                self.F1,
                momentum=self.batch_norm_momentum,
                eps=self.batch_norm_eps,
            ),
            nn.ELU(),
        )
        self.time_projector = _PointwiseProjector(
            self.F1,
            projection_hidden_size,
            projection_size,
            batch_norm_momentum=self.batch_norm_momentum,
            batch_norm_eps=self.batch_norm_eps,
        )
        self.frequency_projector = _PointwiseProjector(
            self.F1,
            projection_hidden_size,
            projection_size,
            batch_norm_momentum=self.batch_norm_momentum,
            batch_norm_eps=self.batch_norm_eps,
        )
        self.projection_hidden_size = projection_hidden_size
        self.projection_size = projection_size

    def _forward_backbone_with_time_features(
        self, eeg: Tensor
    ) -> tuple[Tensor, Tensor]:
        features = eeg
        time_features = None
        for module_name in self._backbone_module_names:
            features = self._modules[module_name](features)
            if module_name == "conv_temporal":
                time_features = features
        if time_features is None:
            raise RuntimeError("conv_temporal was not found in the backbone")
        return features, time_features

    def _encode_augmented_time(
        self,
        eeg: Tensor,
        position_indices: Tensor | None,
    ) -> Tensor:
        features = self.ensuredims(eeg)
        features = self.dimshuffle(features)
        if position_indices is None:
            return self.conv_temporal(features)
        return self.conv_temporal.forward_selected(
            features,
            position_indices,
        )

    def _encode_frequency(self, frequency: Tensor) -> Tensor:
        if frequency.ndim != 4:
            raise ValueError(
                "frequency input must have shape (B,C,K,P), got "
                f"{tuple(frequency.shape)}"
            )
        if frequency.shape[1] != self.n_chans:
            raise ValueError(
                f"expected {self.n_chans} frequency channels, got "
                f"{frequency.shape[1]}"
            )
        if frequency.shape[2] != self.n_frequency_bands:
            raise ValueError(
                f"expected {self.n_frequency_bands} bands, got "
                f"{frequency.shape[2]}"
            )
        if frequency.shape[3] != self.patch_count:
            raise ValueError(
                f"expected {self.patch_count} frequency patches, got "
                f"{frequency.shape[3]}"
            )
        return self.frequency_encoder(
            frequency.permute(0, 2, 1, 3).contiguous()
        )

    def forward(self, eeg: Tensor) -> Tensor:
        """Run only the unchanged classification backbone."""
        # Braindecode calls self.forward while the parent EEGNet is still
        # being assembled to infer its output shape. The frozen backbone
        # module-name list does not exist at that point.
        if not hasattr(self, "_backbone_module_names"):
            return nn.Sequential.forward(self, eeg)
        logits, _ = self._forward_backbone_with_time_features(eeg)
        return logits

    def forward_contrastive(
        self,
        eeg: Tensor,
        augmented_eeg: Tensor,
        frequency: Tensor,
        augmented_frequency: Tensor,
        *,
        time_position_indices: Tensor | None = None,
    ) -> FilterNFCTrainingOutput:
        """Run the classification path and TF-C representations for training."""
        logits, h_time = self._forward_backbone_with_time_features(eeg)
        h_time_augmented = self._encode_augmented_time(
            augmented_eeg,
            time_position_indices,
        )
        h_frequency = self._encode_frequency(frequency)
        h_frequency_augmented = self._encode_frequency(
            augmented_frequency
        )
        z_time = self.time_projector(h_time)
        z_frequency = self.frequency_projector(h_frequency)
        return FilterNFCTrainingOutput(
            logits=logits,
            h_time=h_time,
            h_time_augmented=h_time_augmented,
            h_frequency=h_frequency,
            h_frequency_augmented=h_frequency_augmented,
            z_time=z_time,
            z_frequency=z_frequency,
        )


def build_deformable_patchcfc_filternfc(
    n_classes: int,
    *,
    n_channels: int = 128,
    n_times: int = 1153,
    **model_kwargs,
) -> deformable_patchCfC_filterNFC:
    """Build the training-only TF-C regularized deformable Patch-CfC."""
    return deformable_patchCfC_filterNFC(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        **model_kwargs,
    )


if __name__ == "__main__":
    model = build_deformable_patchcfc_filternfc(
        n_classes=4,
        n_channels=16,
        n_times=129,
        patch_size=32,
        patch_stride=4,
        n_electrode_groups=4,
        n_frequency_bands=2,
    )
    eeg = torch.randn(2, 16, 129, requires_grad=True)
    frequency = torch.rand(2, 16, 2, 25)
    augmented_eeg = eeg + 0.05 * torch.randn_like(eeg)
    augmented_frequency = frequency + 0.01 * torch.rand_like(frequency)

    model.eval()
    with torch.no_grad():
        inference_logits = model(eeg)
    if inference_logits.shape != (2, 4):
        raise RuntimeError(f"unexpected inference shape: {inference_logits.shape}")

    model.train()
    output = model.forward_contrastive(
        eeg,
        augmented_eeg,
        frequency,
        augmented_frequency,
    )
    expected_feature_shape = (2, 8, 16, 25)
    for name, feature in (
        ("h_time", output.h_time),
        ("h_time_augmented", output.h_time_augmented),
        ("h_frequency", output.h_frequency),
        ("h_frequency_augmented", output.h_frequency_augmented),
        ("z_time", output.z_time),
        ("z_frequency", output.z_frequency),
    ):
        if feature.shape != expected_feature_shape:
            raise RuntimeError(f"unexpected {name} shape: {feature.shape}")
    output.logits.sum().backward()
    print("deformable_patchCfC_filterNFC self-test passed")
