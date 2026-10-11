"""TF-C augmentations and dynamic structured NT-Xent loss."""

from __future__ import annotations

import math
import operator
from typing import NamedTuple

import torch
from torch import Tensor
from torch.nn import functional as F


class TFCLosses(NamedTuple):
    total: Tensor
    time: Tensor
    frequency: Tensor
    cross_domain: Tensor


def _non_negative_float(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and non-negative")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return parsed


def augment_time(
    eeg: Tensor,
    *,
    jitter_std: float,
    scaling_std: float,
) -> Tensor:
    """Apply TF-C-style jitter and optional channel-wise scaling."""
    jitter_std = _non_negative_float("time_jitter_std", jitter_std)
    scaling_std = _non_negative_float("time_scaling_std", scaling_std)
    augmented = eeg
    if scaling_std > 0.0:
        scale = 1.0 + scaling_std * torch.randn(
            eeg.shape[0],
            eeg.shape[1],
            1,
            device=eeg.device,
            dtype=eeg.dtype,
        )
        augmented = augmented * scale
    if jitter_std > 0.0:
        augmented = augmented + jitter_std * torch.randn_like(eeg)
    return augmented


def augment_frequency(
    frequency: Tensor,
    *,
    perturb_ratio: float,
    add_scale: float,
) -> Tensor:
    """Randomly remove or add small frequency components per trial.

    This follows the TF-C frequency-augmentation bank while avoiding the
    original implementation's hard-coded CUDA tensor allocation.
    """
    perturb_ratio = _non_negative_float(
        "frequency_perturb_ratio", perturb_ratio
    )
    add_scale = _non_negative_float("frequency_add_scale", add_scale)
    if perturb_ratio > 1.0:
        raise ValueError("frequency_perturb_ratio must not exceed 1")
    if perturb_ratio == 0.0:
        return frequency.clone()

    remove_mask = torch.rand_like(frequency) < perturb_ratio
    removed = frequency.masked_fill(remove_mask, 0.0)

    add_mask = torch.rand_like(frequency) < perturb_ratio
    maximum = frequency.amax(dim=(1, 2, 3), keepdim=True)
    addition = (
        torch.rand_like(frequency)
        * maximum
        * add_scale
        * add_mask.to(dtype=frequency.dtype)
    )
    added = frequency + addition

    choose_remove = (
        torch.rand(
            frequency.shape[0],
            1,
            1,
            1,
            device=frequency.device,
        )
        < 0.5
    )
    return torch.where(choose_remove, removed, added)


def _flatten_structured_features(features: Tensor) -> Tensor:
    if features.ndim != 4:
        raise ValueError(
            "structured contrastive features must have shape (B,F,C,P), "
            f"got {tuple(features.shape)}"
        )
    # Positions are never mixed or pooled. Shape becomes (C*P, B, F).
    return features.permute(2, 3, 0, 1).reshape(
        features.shape[2] * features.shape[3],
        features.shape[0],
        features.shape[1],
    )


def sample_position_indices(
    n_positions: int,
    positions_per_batch: int | None,
    *,
    device: torch.device | str,
) -> Tensor | None:
    """Sample shared channel-major positions before augmented Patch-CfC."""
    if isinstance(n_positions, bool):
        raise ValueError("n_positions must be a positive integer")
    try:
        n_positions = operator.index(n_positions)
    except TypeError as error:
        raise ValueError("n_positions must be a positive integer") from error
    if n_positions < 1:
        raise ValueError("n_positions must be a positive integer")
    if positions_per_batch is None:
        return None
    if isinstance(positions_per_batch, bool):
        raise ValueError("positions_per_batch must be positive or null")
    try:
        positions_per_batch = operator.index(positions_per_batch)
    except TypeError as error:
        raise ValueError(
            "positions_per_batch must be positive or null"
        ) from error
    if positions_per_batch < 1:
        raise ValueError("positions_per_batch must be positive or null")
    if positions_per_batch >= n_positions:
        return None
    return torch.randperm(
        n_positions,
        device=device,
    )[:positions_per_batch]


def _validate_position_indices(
    position_indices: Tensor,
    *,
    n_positions: int,
    device: torch.device,
) -> Tensor:
    if position_indices.ndim != 1:
        raise ValueError("position_indices must be one-dimensional")
    if position_indices.numel() < 1:
        raise ValueError("position_indices must not be empty")
    if position_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("position_indices must contain integer indices")
    position_indices = position_indices.to(device=device, dtype=torch.long)
    if torch.any(
        (position_indices < 0) | (position_indices >= n_positions)
    ).item():
        raise IndexError(f"position_indices must be in [0, {n_positions})")
    return position_indices


def positionwise_nt_xent(
    first: Tensor,
    second: Tensor,
    *,
    temperature: float,
    position_indices: Tensor | None = None,
    chunk_size: int = 256,
) -> Tensor:
    """Calculate SimCLR NT-Xent independently at matching EEG positions."""
    if first.shape != second.shape:
        raise ValueError(
            f"contrastive view mismatch: {first.shape} != {second.shape}"
        )
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    chunk_size = operator.index(chunk_size)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    first_positions = _flatten_structured_features(first)
    second_positions = _flatten_structured_features(second)
    if position_indices is not None:
        first_positions = first_positions.index_select(0, position_indices)
        second_positions = second_positions.index_select(0, position_indices)

    n_positions, batch_size, _ = first_positions.shape
    if batch_size < 2:
        return first.sum() * 0.0

    paired_targets = (
        torch.arange(2 * batch_size, device=first.device) + batch_size
    ) % (2 * batch_size)
    diagonal_mask = torch.eye(
        2 * batch_size,
        device=first.device,
        dtype=torch.bool,
    )
    total_loss = first.new_zeros(())
    total_anchors = 0

    for start in range(0, n_positions, chunk_size):
        stop = min(start + chunk_size, n_positions)
        representations = torch.cat(
            (first_positions[start:stop], second_positions[start:stop]),
            dim=1,
        )
        representations = F.normalize(representations, p=2, dim=-1)
        logits = torch.matmul(
            representations,
            representations.transpose(-1, -2),
        ) / temperature
        logits = logits.masked_fill(diagonal_mask.unsqueeze(0), float("-inf"))
        current_positions = stop - start
        loss = F.cross_entropy(
            logits.reshape(
                current_positions * 2 * batch_size,
                2 * batch_size,
            ),
            paired_targets.repeat(current_positions),
            reduction="sum",
        )
        total_loss = total_loss + loss
        total_anchors += current_positions * 2 * batch_size
    return total_loss / total_anchors


def compute_tfc_losses(
    *,
    h_time: Tensor,
    h_time_augmented: Tensor,
    h_frequency: Tensor,
    h_frequency_augmented: Tensor,
    z_time: Tensor,
    z_frequency: Tensor,
    temperature: float,
    intra_domain_weight: float,
    positions_per_batch: int | None,
    chunk_size: int,
    position_indices: Tensor | None = None,
) -> TFCLosses:
    """Compute TF-C time, frequency and cross-domain NT-Xent losses."""
    if h_time.ndim != 4:
        raise ValueError(
            "h_time must have shape (B,F,C,P), got "
            f"{tuple(h_time.shape)}"
        )
    reference_shape = h_time.shape
    for name, feature in (
        ("h_frequency", h_frequency),
        ("h_frequency_augmented", h_frequency_augmented),
    ):
        if feature.shape != reference_shape:
            raise ValueError(
                f"{name} shape {feature.shape} does not match "
                f"h_time {reference_shape}"
            )

    selected_augmented_time = h_time_augmented.ndim == 3
    if selected_augmented_time:
        if h_time_augmented.shape[:2] != h_time.shape[:2]:
            raise ValueError(
                "selected h_time_augmented must have shape (B,F,S), got "
                f"{tuple(h_time_augmented.shape)}"
            )
    elif h_time_augmented.shape != reference_shape:
        raise ValueError(
            f"h_time_augmented shape {h_time_augmented.shape} does not "
            f"match h_time {reference_shape}"
        )

    if z_time.shape != z_frequency.shape:
        raise ValueError("time and frequency projection shapes must match")
    if z_time.shape[0] != h_time.shape[0] or z_time.shape[2:] != h_time.shape[2:]:
        raise ValueError("projected and encoder positions must match")

    intra_domain_weight = _non_negative_float(
        "intra_domain_weight", intra_domain_weight
    )
    n_positions = h_time.shape[2] * h_time.shape[3]
    if position_indices is None:
        if selected_augmented_time:
            raise ValueError(
                "position_indices are required for selected "
                "h_time_augmented"
            )
        position_indices = sample_position_indices(
            n_positions,
            positions_per_batch,
            device=h_time.device,
        )
    else:
        position_indices = _validate_position_indices(
            position_indices,
            n_positions=n_positions,
            device=h_time.device,
        )

    common_kwargs = {
        "temperature": temperature,
        "position_indices": position_indices,
        "chunk_size": chunk_size,
    }
    if selected_augmented_time:
        expected_shape = (
            h_time.shape[0],
            h_time.shape[1],
            position_indices.numel(),
        )
        if h_time_augmented.shape != expected_shape:
            raise ValueError(
                "selected h_time_augmented shape "
                f"{h_time_augmented.shape} does not match {expected_shape}"
            )
        selected_h_time = _flatten_structured_features(h_time).index_select(
            0,
            position_indices,
        ).permute(1, 2, 0).contiguous()
        time_loss = positionwise_nt_xent(
            selected_h_time.unsqueeze(-1),
            h_time_augmented.unsqueeze(-1),
            temperature=temperature,
            chunk_size=chunk_size,
        )
    else:
        time_loss = positionwise_nt_xent(
            h_time,
            h_time_augmented,
            **common_kwargs,
        )
    frequency_loss = positionwise_nt_xent(
        h_frequency, h_frequency_augmented, **common_kwargs
    )
    cross_domain_loss = positionwise_nt_xent(
        z_time, z_frequency, **common_kwargs
    )
    total = (
        intra_domain_weight * (time_loss + frequency_loss)
        + cross_domain_loss
    )
    return TFCLosses(
        total=total,
        time=time_loss,
        frequency=frequency_loss,
        cross_domain=cross_domain_loss,
    )

if __name__ == "__main__":
    generator = torch.Generator().manual_seed(42)
    first = torch.randn(4, 8, 3, 7, generator=generator, requires_grad=True)
    second = first.detach() + 0.05 * torch.randn(
        first.shape, generator=generator
    )
    different = torch.randn(first.shape, generator=generator)
    close_loss = positionwise_nt_xent(first, second, temperature=0.2)
    far_loss = positionwise_nt_xent(first, different, temperature=0.2)
    if not torch.isfinite(close_loss) or not torch.isfinite(far_loss):
        raise RuntimeError("NT-Xent produced a non-finite value")
    if close_loss >= far_loss:
        raise RuntimeError("matching views were not preferred")
    close_loss.backward()
    print("structured TF-C loss self-test passed")
