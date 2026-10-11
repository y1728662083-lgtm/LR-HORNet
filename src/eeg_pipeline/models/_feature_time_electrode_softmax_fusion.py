"""Feature-wise, time-wise dynamic Softmax fusion of EEG electrodes."""

from __future__ import annotations

import operator
from typing import Optional

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


__all__ = ["FeatureTimeElectrodeSoftmaxFusion"]


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


class FeatureTimeElectrodeSoftmaxFusion(nn.Module):
    """Replace a signed spatial convolution with dynamic electrode pooling.

    For every sample, temporal feature, fusion head, and time point, an
    independent length-C score vector is produced by a feature-grouped
    pointwise FFN. Softmax is applied only over electrodes and the original
    feature values are fused with the resulting non-negative weights.

    The scoring function is shared over electrodes and time, but the F
    feature routes have independent parameters. It therefore does not mix
    features, electrodes, or time before the electrode-axis Softmax.
    """

    def __init__(
        self,
        *,
        n_features: int,
        n_heads: int,
        hidden_features_per_feature: int = 8,
        time_chunk_size: Optional[int] = 32,
        activation_checkpoint: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.n_features = _positive_integer("n_features", n_features)
        self.n_heads = _positive_integer("n_heads", n_heads)
        self.hidden_features_per_feature = _positive_integer(
            "hidden_features_per_feature", hidden_features_per_feature
        )
        if time_chunk_size is not None:
            time_chunk_size = _positive_integer(
                "time_chunk_size", time_chunk_size
            )
        if not isinstance(activation_checkpoint, bool):
            raise ValueError("activation_checkpoint must be a boolean.")

        factory_kwargs = {"device": device, "dtype": dtype}
        hidden_channels = (
            self.n_features * self.hidden_features_per_feature
        )
        score_channels = self.n_features * self.n_heads
        self.hidden_projection = nn.Conv2d(
            self.n_features,
            hidden_channels,
            kernel_size=1,
            groups=self.n_features,
            bias=True,
            **factory_kwargs,
        )
        self.activation = nn.GELU()
        self.score_projection = nn.Conv2d(
            hidden_channels,
            score_channels,
            kernel_size=1,
            groups=self.n_features,
            bias=False,
            **factory_kwargs,
        )

        # Start close to uniform electrode fusion without making the hidden
        # projection gradient exactly zero on the first optimization step.
        nn.init.normal_(self.score_projection.weight, mean=0.0, std=1e-3)

        self.time_chunk_size = time_chunk_size
        self.activation_checkpoint = activation_checkpoint
        self.softmax_dimension = "electrode"

    def _validate_input(self, input: Tensor) -> None:
        if input.ndim != 4:
            raise ValueError(
                "electrode fusion expects (B,F,C,T), got "
                f"{tuple(input.shape)}."
            )
        if input.shape[1] != self.n_features:
            raise ValueError(
                f"expected {self.n_features} features, got {input.shape[1]}."
            )
        if input.shape[2] < 1 or input.shape[3] < 1:
            raise ValueError("electrode and time dimensions must be non-empty.")

    def score_electrodes(self, input: Tensor) -> Tensor:
        """Return dynamic logits with shape (B,F,D,C,T)."""
        self._validate_input(input)
        batch_size, _, n_chans, n_times = input.shape
        hidden = self.activation(self.hidden_projection(input))
        scores = self.score_projection(hidden)
        return scores.reshape(
            batch_size,
            self.n_features,
            self.n_heads,
            n_chans,
            n_times,
        )

    def electrode_weights(self, input: Tensor) -> Tensor:
        """Return electrode-normalized weights with shape (B,F,D,C,T)."""
        scores = self.score_electrodes(input)
        if scores.dtype in (torch.float16, torch.bfloat16):
            return torch.softmax(scores.float(), dim=3)
        return torch.softmax(scores, dim=3)

    def _fuse_chunk(self, input: Tensor) -> Tensor:
        weights = self.electrode_weights(input)
        values = input.unsqueeze(2).to(dtype=weights.dtype)
        fused = torch.sum(weights * values, dim=3)
        fused = fused.to(dtype=input.dtype)
        return fused.reshape(
            input.shape[0],
            self.n_features * self.n_heads,
            1,
            input.shape[-1],
        )

    def forward(self, input: Tensor) -> Tensor:
        self._validate_input(input)
        chunk_size = self.time_chunk_size or input.shape[-1]
        outputs = []
        for start in range(0, input.shape[-1], chunk_size):
            chunk = input[..., start : start + chunk_size]
            if (
                self.activation_checkpoint
                and self.training
                and torch.is_grad_enabled()
            ):
                output = checkpoint(
                    self._fuse_chunk,
                    chunk,
                    use_reentrant=False,
                )
            else:
                output = self._fuse_chunk(chunk)
            outputs.append(output)
        return torch.cat(outputs, dim=-1)

    def extra_repr(self) -> str:
        return (
            f"n_features={self.n_features}, n_heads={self.n_heads}, "
            "hidden_features_per_feature="
            f"{self.hidden_features_per_feature}, "
            f"time_chunk_size={self.time_chunk_size}, "
            f"activation_checkpoint={self.activation_checkpoint}, "
            "softmax_dim=electrode"
        )
