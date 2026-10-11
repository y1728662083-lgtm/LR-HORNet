"""Dual-memory WaveClock CfC for causal EEG trial classification.

The model combines causal Haar checkpoints, continuous electrode-group CfCs,
a trial-local wavelet event memory, a monotone liquid clock, and a stable
complex Laplace pole memory. Both caches are local variables of ``forward``:
they reset for every batch, never enter checkpoints, and use read-before-write.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from braindecode.modules import LinearWithConstraint
from ncps.torch import CfC
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from eeg_pipeline.models.causal1_group8PatchLSTM_EEGNet import (
    _positive_integer,
)
from eeg_pipeline.models.deformable_patchCfC import (
    _dropout_probability,
    deformable_patchCfC,
)


__all__ = ["DualMem_WaveClock_CfC", "build_dualmem_waveclock_cfc"]


def _positive_finite(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return parsed


def _nonnegative_finite(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative finite number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be a non-negative finite number")
    return parsed


def _unit_interval(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and in [0, 1]")
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return parsed


def _inverse_softplus(value: Tensor) -> Tensor:
    return value + torch.log(-torch.expm1(-value))


def _logit(value: Tensor) -> Tensor:
    eps = torch.finfo(value.dtype).eps
    value = value.clamp(eps, 1.0 - eps)
    return torch.log(value) - torch.log1p(-value)


def _expanded_cfc_timespans(
    delta_time: Tensor,
    *,
    batch_size: int,
    hidden_size: int,
) -> Tensor:
    """Adapt physical intervals to the vendored ncps squeeze convention."""
    if delta_time.ndim != 1:
        raise ValueError("delta_time must be one-dimensional")
    return delta_time.view(1, -1, 1).expand(
        batch_size, delta_time.numel(), hidden_size
    )


def _run_cfc_sequence(
    cfc: CfC,
    sequence: Tensor,
    delta_time: Tensor,
    *,
    hidden_size: int,
    activation_checkpoint: bool,
) -> Tensor:
    """Run CfC with dtype-safe state and explicit physical intervals."""
    if sequence.ndim != 3:
        raise ValueError("CfC sequence must have shape (N,P,D)")
    n_sequences = sequence.shape[0]
    hx = sequence.new_zeros((n_sequences, hidden_size))
    timespans = _expanded_cfc_timespans(
        delta_time.to(device=sequence.device, dtype=sequence.dtype),
        batch_size=n_sequences,
        hidden_size=hidden_size,
    )

    def run(current: Tensor, state: Tensor, spans: Tensor) -> Tensor:
        outputs, _ = cfc(current, hx=state, timespans=spans)
        return outputs

    if activation_checkpoint and sequence.requires_grad and torch.is_grad_enabled():
        return checkpoint(run, sequence, hx, timespans, use_reentrant=False)
    return run(sequence, hx, timespans)


def _gather_cache(values: Tensor, indices: Tensor, valid_ndim: int) -> Tensor:
    """Gather cache slots while retaining arbitrary trailing value axes."""
    slot_axis = valid_ndim - 1
    expanded = indices
    for _ in range(values.ndim - valid_ndim):
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand(*indices.shape, *values.shape[valid_ndim:])
    return torch.gather(values, dim=slot_axis, index=expanded)


def _read_fixed_cache(
    query: Tensor,
    keys: Tensor,
    values: Tensor,
    valid: Tensor,
    *,
    topk: int,
    temperature: float,
    score_bias: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Cosine top-k read; invalid slots always receive zero weight."""
    if query.shape[:-1] != keys.shape[:-2]:
        raise ValueError("query and cache leading dimensions do not match")
    normalized_query = F.normalize(query, dim=-1, eps=1e-6)
    normalized_keys = F.normalize(keys, dim=-1, eps=1e-6)
    scores = (normalized_keys * normalized_query.unsqueeze(-2)).sum(dim=-1)
    if score_bias is None:
        retrieval_scores = scores
    else:
        if score_bias.shape != scores.shape:
            raise ValueError("cache score_bias must match cache scores")
        retrieval_scores = scores + score_bias
    active = valid.any(dim=-1)
    floor = torch.finfo(scores.dtype).min
    masked_scores = retrieval_scores.masked_fill(~valid, floor)
    k = min(topk, keys.shape[-2])
    selected_scores, indices = torch.topk(masked_scores, k=k, dim=-1)
    # Softmax(all -inf) is NaN for an empty cache. Replace that row before
    # softmax; the validity mask below still makes its final weights exactly 0.
    selected_scores = torch.where(
        active.unsqueeze(-1), selected_scores, torch.zeros_like(selected_scores)
    )
    weights = torch.softmax(selected_scores / temperature, dim=-1)
    selected_valid = torch.gather(valid, dim=-1, index=indices)
    weights = weights * selected_valid.to(dtype=weights.dtype)
    denominator = weights.sum(dim=-1, keepdim=True)
    denominator = torch.where(
        active.unsqueeze(-1), denominator, torch.ones_like(denominator)
    )
    weights = weights / denominator.clamp_min(torch.finfo(weights.dtype).eps)
    weights = torch.where(active.unsqueeze(-1), weights, torch.zeros_like(weights))

    selected_values = _gather_cache(values, indices, valid.ndim)
    value_weights = weights
    for _ in range(values.ndim - valid.ndim):
        value_weights = value_weights.unsqueeze(-1)
    slot_axis = valid.ndim - 1
    retrieved_value = (selected_values * value_weights).sum(dim=slot_axis)
    selected_keys = _gather_cache(keys, indices, valid.ndim)
    retrieved_key = (selected_keys * weights.unsqueeze(-1)).sum(dim=slot_axis)

    novelty_scores = scores.masked_fill(~valid, floor)
    best_similarity = novelty_scores.max(dim=-1).values.clamp(-1.0, 1.0)
    novelty = torch.where(
        active, 1.0 - best_similarity, torch.ones_like(best_similarity)
    )
    return retrieved_value, retrieved_key, indices, weights, novelty


def _write_fixed_cache(
    keys: Tensor,
    values: Tensor,
    valid: Tensor,
    pointer: Tensor,
    new_key: Tensor,
    new_value: Tensor,
    should_write: Tensor,
    *,
    detach: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Out-of-place ring write, independently gated per trial/group."""
    if detach:
        new_key = new_key.detach()
        new_value = new_value.detach()
    memory_slots = keys.shape[-2]
    slot_axis = valid.ndim - 1
    slot = torch.remainder(pointer, memory_slots)
    selector = F.one_hot(slot, num_classes=memory_slots).to(dtype=torch.bool)
    selector = selector & should_write.unsqueeze(-1)
    keys = torch.where(selector.unsqueeze(-1), new_key.unsqueeze(slot_axis), keys)
    value_selector = selector
    for _ in range(values.ndim - valid.ndim):
        value_selector = value_selector.unsqueeze(-1)
    values = torch.where(value_selector, new_value.unsqueeze(slot_axis), values)
    valid = valid | selector
    pointer = pointer + should_write.to(dtype=pointer.dtype)
    return keys, values, valid, pointer, selector


class _ElectrodeGroupWaveletMemoryCfC(nn.Module):
    """Causal Haar checkpoints, sparse event memory and continuous group CfCs."""

    def __init__(
        self,
        *,
        n_chans: int,
        n_features: int,
        patch_size: int,
        patch_stride: int,
        n_electrode_groups: int,
        wavelet_levels: int,
        memory_slots: int,
        memory_key_dim: int,
        memory_topk: int,
        memory_temperature: float,
        memory_novelty_threshold: float,
        sampling_rate_hz: float,
        backbone_units: int,
        backbone_layers: int,
        backbone_dropout: float,
        activation_checkpoint: bool,
        memory_detach: bool,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.n_chans = _positive_integer("n_chans", n_chans)
        self.n_features = _positive_integer("n_features", n_features)
        if self.n_features == 1:
            raise ValueError(
                "n_features must exceed 1 because the vendored CfC squeezes "
                "singleton timespan dimensions"
            )
        self.patch_size = _positive_integer("patch_size", patch_size)
        self.patch_stride = _positive_integer("patch_stride", patch_stride)
        self.n_electrode_groups = _positive_integer(
            "n_electrode_groups", n_electrode_groups
        )
        if self.patch_stride > self.patch_size:
            raise ValueError("patch_stride must not exceed patch_size")
        if self.n_chans % self.n_electrode_groups != 0:
            raise ValueError("n_chans must be divisible by n_electrode_groups")
        self.chans_per_group = self.n_chans // self.n_electrode_groups

        self.wavelet_levels = _positive_integer("wavelet_levels", wavelet_levels)
        if 2**self.wavelet_levels != self.patch_size:
            raise ValueError(
                "patch_size must equal 2**wavelet_levels so every checkpoint "
                "covers the complete configured patch"
            )
        self.n_wavelet_features = self.wavelet_levels + 1
        self.memory_slots = _positive_integer("wavelet_memory_slots", memory_slots)
        self.memory_key_dim = _positive_integer(
            "wavelet_memory_key_dim", memory_key_dim
        )
        self.memory_topk = _positive_integer("wavelet_memory_topk", memory_topk)
        if self.memory_topk > self.memory_slots:
            raise ValueError("wavelet_memory_topk must not exceed memory slots")
        self.memory_temperature = _positive_finite(
            "wavelet_memory_temperature", memory_temperature
        )
        self.memory_novelty_threshold = _unit_interval(
            "wavelet_memory_novelty_threshold", memory_novelty_threshold
        )
        self.sampling_rate_hz = _positive_finite(
            "sampling_rate_hz", sampling_rate_hz
        )
        if not isinstance(activation_checkpoint, bool):
            raise ValueError("activation_checkpoint must be a boolean")
        if not isinstance(memory_detach, bool):
            raise ValueError("memory_detach must be a boolean")
        self.activation_checkpoint = activation_checkpoint
        self.memory_detach = memory_detach

        event_dim = self.chans_per_group * self.n_wavelet_features
        self.event_encoder = nn.Sequential(
            nn.Linear(event_dim, self.memory_key_dim),
            nn.GELU(),
            nn.LayerNorm(self.memory_key_dim),
        )
        # A shared key/query projection remains trainable even when cache
        # contents are detached between timesteps.
        self.key_query_head = nn.Linear(
            self.memory_key_dim, self.memory_key_dim, bias=False
        )
        self.value_head = nn.Linear(
            self.memory_key_dim, self.memory_key_dim, bias=False
        )
        self.scale_head = nn.Linear(2 * self.memory_key_dim, event_dim)
        nn.init.zeros_(self.scale_head.weight)
        nn.init.zeros_(self.scale_head.bias)

        backbone_units = _positive_integer("cfc_backbone_units", backbone_units)
        backbone_layers = _positive_integer("cfc_backbone_layers", backbone_layers)
        backbone_dropout = _dropout_probability(
            "cfc_backbone_dropout", backbone_dropout
        )
        self.group_cfcs = nn.ModuleList(
            [
                CfC(
                    input_size=self.n_wavelet_features,
                    units=self.n_features,
                    proj_size=None,
                    return_sequences=True,
                    batch_first=True,
                    mixed_memory=False,
                    mode="default",
                    activation="lecun_tanh",
                    backbone_units=backbone_units,
                    backbone_layers=backbone_layers,
                    backbone_dropout=backbone_dropout,
                )
                for _ in range(self.n_electrode_groups)
            ]
        )
        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)
        self.backbone_units = backbone_units
        self.backbone_layers = backbone_layers
        self.backbone_dropout = backbone_dropout

        self.last_memory_context: Optional[Tensor] = None
        self.last_memory_active: Optional[Tensor] = None
        self.last_memory_novelty: Optional[Tensor] = None
        self._last_diagnostics: dict[str, Tensor] = {}

    def _validate_input(self, input: Tensor) -> tuple[int, int, Tensor]:
        if input.ndim != 4:
            raise ValueError(
                "wavelet group CfC expects (B,1,C,T), "
                f"got {tuple(input.shape)}"
            )
        if input.dtype not in (torch.float32, torch.float64):
            raise TypeError(
                "DualMem_WaveClock_CfC requires float32/float64 input"
            )
        batch_size, features, channels, n_times = input.shape
        if features != 1 or channels != self.n_chans:
            raise ValueError(
                f"expected (B,1,{self.n_chans},T), got {tuple(input.shape)}"
            )
        if n_times < self.patch_size:
            raise ValueError("input is shorter than patch_size")
        n_patches = (n_times - self.patch_size) // self.patch_stride + 1
        effective = self.patch_size + (n_patches - 1) * self.patch_stride
        return batch_size, n_patches, input[:, 0, :, :effective]

    def _causal_haar(self, signal: Tensor, n_patches: int) -> Tensor:
        """Return ``(B,C,P,J+1)`` causal details and a coarse rolling mean."""
        prefix = F.pad(signal.cumsum(dim=-1), (1, 0))
        ends = self.patch_size + torch.arange(
            n_patches, device=signal.device, dtype=torch.long
        ) * self.patch_stride

        def interval_mean(end: Tensor, width: int) -> Tensor:
            right = prefix.index_select(-1, end)
            left = prefix.index_select(-1, end - width)
            return (right - left) / float(width)

        coefficients = []
        for level in range(1, self.wavelet_levels + 1):
            width = 2**level
            half = width // 2
            recent = interval_mean(ends, half)
            previous = interval_mean(ends - half, half)
            coefficients.append(recent - previous)
        coefficients.append(interval_mean(ends, 2**self.wavelet_levels))
        return torch.stack(coefficients, dim=-1)

    def forward(self, input: Tensor) -> Tensor:
        batch_size, n_patches, signal = self._validate_input(input)
        haar = self._causal_haar(signal, n_patches)
        grouped = haar.reshape(
            batch_size,
            self.n_electrode_groups,
            self.chans_per_group,
            n_patches,
            self.n_wavelet_features,
        )
        events = grouped.permute(0, 1, 3, 2, 4).reshape(
            batch_size,
            self.n_electrode_groups,
            n_patches,
            self.chans_per_group * self.n_wavelet_features,
        )

        # Cache leading axes are (trial, electrode_group). Nothing survives
        # this forward call, so samples and train/validation phases are isolated.
        keys = input.new_zeros(
            batch_size,
            self.n_electrode_groups,
            self.memory_slots,
            self.memory_key_dim,
        )
        values = torch.zeros_like(keys)
        valid = torch.zeros(
            batch_size,
            self.n_electrode_groups,
            self.memory_slots,
            device=input.device,
            dtype=torch.bool,
        )
        pointer = torch.zeros(
            batch_size,
            self.n_electrode_groups,
            device=input.device,
            dtype=torch.long,
        )

        gated_steps = []
        context_steps = []
        active_steps = []
        novelty_steps = []
        for patch_index in range(n_patches):
            encoded = self.event_encoder(events[:, :, patch_index, :])
            current_key = F.normalize(
                self.key_query_head(encoded), dim=-1, eps=1e-6
            )
            current_value = self.value_head(encoded)
            retrieved, _, _, _, novelty = _read_fixed_cache(
                current_key,
                keys,
                values,
                valid,
                topk=self.memory_topk,
                temperature=self.memory_temperature,
            )
            active = valid.any(dim=-1)
            context = current_value + retrieved
            scale = 1.0 + torch.tanh(
                self.scale_head(torch.cat((encoded, retrieved), dim=-1))
            )
            scale = scale.reshape(
                batch_size,
                self.n_electrode_groups,
                self.chans_per_group,
                self.n_wavelet_features,
            )
            gated_steps.append(grouped[:, :, :, patch_index, :] * scale)
            context_steps.append(context)
            active_steps.append(active)
            novelty_steps.append(novelty)

            # Read precedes this write: a checkpoint cannot retrieve itself.
            should_write = (~active) | (
                novelty >= self.memory_novelty_threshold
            )
            keys, values, valid, pointer, _ = _write_fixed_cache(
                keys,
                values,
                valid,
                pointer,
                current_key,
                current_value,
                should_write,
                detach=self.memory_detach,
            )

        gated = torch.stack(gated_steps, dim=3)
        delta_time = input.new_full(
            (n_patches,), self.patch_stride / self.sampling_rate_hz
        )
        # The first checkpoint has observed a full patch from the trial origin.
        delta_time[0] = self.patch_size / self.sampling_rate_hz
        group_outputs = []
        for group_index, cfc in enumerate(self.group_cfcs):
            sequence = gated[:, group_index].reshape(
                batch_size * self.chans_per_group,
                n_patches,
                self.n_wavelet_features,
            )
            recurrent = _run_cfc_sequence(
                cfc,
                sequence,
                delta_time,
                hidden_size=self.n_features,
                activation_checkpoint=self.activation_checkpoint,
            )
            group_outputs.append(
                recurrent.reshape(
                    batch_size,
                    self.chans_per_group,
                    n_patches,
                    self.n_features,
                )
            )
        output = (
            torch.cat(group_outputs, dim=1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        all_context = torch.stack(context_steps, dim=2)
        all_active = torch.stack(active_steps, dim=2)
        all_novelty = torch.stack(novelty_steps, dim=2)
        self.last_memory_context = all_context.mean(dim=1)
        self.last_memory_active = all_active.any(dim=1)
        self.last_memory_novelty = all_novelty.mean(dim=1).detach()
        self._last_diagnostics = {
            # Retain one trial for visualization without pinning an entire
            # batch-sized diagnostic tensor in GPU memory.
            "wavelet_coefficients": haar[:1].detach(),
            "wavelet_memory_active": self.last_memory_active.detach(),
            "wavelet_memory_novelty": self.last_memory_novelty.detach(),
            "wavelet_cache_fill": valid.float().mean(dim=(-1, -2)).detach(),
        }
        return output

    def forward_selected(self, input: Tensor, position_indices: Tensor) -> Tensor:
        """Encode the full causal history, then gather channel-major positions."""
        output = self.forward(input)
        batch_size, features, channels, n_patches = output.shape
        if position_indices.ndim != 1 or position_indices.numel() < 1:
            raise ValueError("position_indices must be a non-empty vector")
        indices = position_indices.to(device=output.device, dtype=torch.long)
        if torch.any((indices < 0) | (indices >= channels * n_patches)).item():
            raise IndexError("position_indices are outside the channel-patch grid")
        flattened = output.permute(0, 2, 3, 1).reshape(
            batch_size, channels * n_patches, features
        )
        return (
            flattened.index_select(1, indices)
            .permute(0, 2, 1)
            .contiguous()
        )

    def diagnostics(self) -> dict[str, Tensor]:
        return dict(self._last_diagnostics)


class DualMem_WaveClock_CfC(deformable_patchCfC):
    """Causal Wavelet-KV plus liquid-clock Laplace-KV EEG classifier."""

    def __init__(
        self,
        *args,
        patch_size: int = 64,
        patch_stride: int = 4,
        n_electrode_groups: int = 16,
        cfc_hidden_size: Optional[int] = None,
        cfc_backbone_units: int = 8,
        cfc_backbone_layers: int = 1,
        cfc_backbone_dropout: float = 0.0,
        cfc_activation_checkpoint: bool = True,
        wavelet_levels: int = 6,
        sampling_rate_hz: float = 256.0,
        wavelet_memory_slots: int = 32,
        wavelet_memory_key_dim: int = 16,
        wavelet_memory_topk: int = 4,
        wavelet_memory_temperature: float = 0.2,
        wavelet_memory_novelty_threshold: float = 0.2,
        clock_hidden_size: int = 8,
        clock_min_speed: float = 0.25,
        clock_max_speed: float = 4.0,
        clock_backbone_units: int = 8,
        clock_backbone_layers: int = 1,
        clock_backbone_dropout: float = 0.0,
        n_laplace_poles: int = 4,
        min_pole_decay: float = 0.05,
        max_pole_frequency_hz: float = 8.0,
        laplace_memory_slots: int = 16,
        laplace_memory_key_dim: int = 32,
        laplace_memory_topk: int = 4,
        laplace_memory_temperature: float = 0.2,
        laplace_memory_novelty_threshold: float = 0.2,
        laplace_memory_age_penalty: float = 0.1,
        memory_detach: bool = True,
        cross_view_consistency_weight: float = 1.0,
        clock_identity_weight: float = 0.1,
        clock_smoothness_weight: float = 0.1,
        key_alignment_weight: float = 0.1,
        auxiliary_loss_weight: float = 0.01,
        normalization: str = "batchnorm",
        n_norm_groups: int = 4,
        drop_prob: float = 0.25,
        final_layer_with_constraint: bool = False,
        norm_rate: float = 0.25,
        **kwargs,
    ) -> None:
        # Braindecode may run forward while a parent constructor infers shapes.
        object.__setattr__(self, "_dualmem_ready", False)
        super().__init__(
            *args,
            patch_size=patch_size,
            patch_stride=patch_stride,
            n_electrode_groups=n_electrode_groups,
            cfc_hidden_size=cfc_hidden_size,
            cfc_backbone_units=cfc_backbone_units,
            cfc_backbone_layers=cfc_backbone_layers,
            cfc_backbone_dropout=cfc_backbone_dropout,
            cfc_activation_checkpoint=cfc_activation_checkpoint,
            offset_cfc_hidden_size=clock_hidden_size,
            offset_cfc_backbone_units=clock_backbone_units,
            offset_cfc_backbone_layers=clock_backbone_layers,
            offset_cfc_backbone_dropout=clock_backbone_dropout,
            normalization=normalization,
            n_norm_groups=n_norm_groups,
            drop_prob=drop_prob,
            final_layer_with_constraint=final_layer_with_constraint,
            norm_rate=norm_rate,
            **kwargs,
        )
        if cfc_hidden_size is None:
            cfc_hidden_size = self.F1
        if _positive_integer("cfc_hidden_size", cfc_hidden_size) != self.F1:
            raise ValueError("cfc_hidden_size must equal EEGNet F1")
        auxiliary_loss_weight = float(auxiliary_loss_weight)
        if not math.isfinite(auxiliary_loss_weight) or auxiliary_loss_weight < 0.0:
            raise ValueError("auxiliary_loss_weight must be finite and non-negative")

        reference = next(self.conv_temporal.parameters())
        device, dtype = reference.device, reference.dtype
        self.conv_temporal = _ElectrodeGroupWaveletMemoryCfC(
            n_chans=self.n_chans,
            n_features=self.F1,
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
            n_electrode_groups=self.n_electrode_groups,
            wavelet_levels=wavelet_levels,
            memory_slots=wavelet_memory_slots,
            memory_key_dim=wavelet_memory_key_dim,
            memory_topk=wavelet_memory_topk,
            memory_temperature=wavelet_memory_temperature,
            memory_novelty_threshold=wavelet_memory_novelty_threshold,
            sampling_rate_hz=sampling_rate_hz,
            backbone_units=cfc_backbone_units,
            backbone_layers=cfc_backbone_layers,
            backbone_dropout=cfc_backbone_dropout,
            activation_checkpoint=cfc_activation_checkpoint,
            memory_detach=memory_detach,
            device=device,
            dtype=dtype,
        )
        self.pool_1 = _DualMemoryLiquidLaplaceHead(
            n_feature_maps=self.F1 * self.D,
            wavelet_context_dim=wavelet_memory_key_dim,
            sampling_rate_hz=sampling_rate_hz,
            checkpoint_size=self.patch_size,
            checkpoint_stride=self.patch_stride,
            clock_hidden_size=clock_hidden_size,
            clock_min_speed=clock_min_speed,
            clock_max_speed=clock_max_speed,
            n_laplace_poles=n_laplace_poles,
            min_pole_decay=min_pole_decay,
            max_pole_frequency_hz=max_pole_frequency_hz,
            memory_slots=laplace_memory_slots,
            memory_key_dim=laplace_memory_key_dim,
            memory_topk=laplace_memory_topk,
            memory_temperature=laplace_memory_temperature,
            memory_novelty_threshold=laplace_memory_novelty_threshold,
            memory_age_penalty=laplace_memory_age_penalty,
            backbone_units=clock_backbone_units,
            backbone_layers=clock_backbone_layers,
            backbone_dropout=clock_backbone_dropout,
            activation_checkpoint=cfc_activation_checkpoint,
            memory_detach=memory_detach,
            consistency_weight=cross_view_consistency_weight,
            clock_identity_weight=clock_identity_weight,
            clock_smoothness_weight=clock_smoothness_weight,
            key_alignment_weight=key_alignment_weight,
            device=device,
            dtype=dtype,
        )

        # The new head jointly replaces the old temporal deformation, both
        # pooling operations and the separable temporal convolution.
        self.drop_1 = nn.Identity()
        self.conv_separable_depth = nn.Identity()
        self.conv_separable_point = nn.Identity()
        self.bnorm_2 = nn.Identity()
        self.elu_2 = nn.Identity()
        self.pool_2 = nn.Identity()
        self.drop_2 = nn.Identity()

        if final_layer_with_constraint:
            classifier: nn.Module = LinearWithConstraint(
                in_features=self.pool_1.embedding_dim,
                out_features=self.n_outputs,
                max_norm=self.norm_rate,
            )
        else:
            classifier = nn.Linear(self.pool_1.embedding_dim, self.n_outputs)
        classifier = classifier.to(device=device, dtype=dtype)
        nn.init.xavier_uniform_(classifier.weight)
        if classifier.bias is not None:
            nn.init.zeros_(classifier.bias)
        self.final_layer = nn.Sequential(nn.Dropout(drop_prob), classifier)
        self.final_conv_length = 1
        self.memory_auxiliary_loss_weight = auxiliary_loss_weight
        self.memory_scope = "trial_local_read_before_write"
        self.temporal_mechanism = (
            "causal_rolling_haar_contrasts_continuous_group_cfc"
        )
        self.deformation_mechanism = "monotone_liquid_clock_stable_laplace"
        self.algorithmic_latency_samples = self.patch_size
        self.causality_scope = (
            "causal state updates with offline trial normalization/classification"
        )
        self._dualmem_ready = True

    def forward(self, input: Tensor) -> Tensor:
        if not getattr(self, "_dualmem_ready", False):
            return super().forward(input)

        features = self.ensuredims(input)
        features = self.dimshuffle(features)
        features = self.conv_temporal(features)
        features = self.bnorm_temporal(features)
        features = self.conv_spatial(features)
        features = self.bnorm_1(features)
        features = self.elu_1(features)
        wavelet_context = self.conv_temporal.last_memory_context
        wavelet_active = self.conv_temporal.last_memory_active
        if wavelet_context is None or wavelet_active is None:
            raise RuntimeError("wavelet memory context was not produced")
        # Locals retain the current graph; clearing module attributes prevents
        # the completed batch graph from surviving into the next forward pass.
        self.conv_temporal.last_memory_context = None
        self.conv_temporal.last_memory_active = None
        embedding = self.pool_1(features, wavelet_context, wavelet_active)
        return self.final_layer(embedding)

    def memory_auxiliary_loss(self) -> Tensor:
        return self.pool_1.auxiliary_loss(clear=True)

    def memory_auxiliary_components(self) -> dict[str, Tensor]:
        return self.pool_1.auxiliary_components()

    def memory_diagnostics(self) -> dict[str, Tensor]:
        diagnostics = self.conv_temporal.diagnostics()
        diagnostics.update(self.pool_1.diagnostics())
        return diagnostics


def build_dualmem_waveclock_cfc(
    n_classes: int,
    *,
    n_channels: int = 128,
    n_times: int = 1153,
    **model_kwargs,
) -> DualMem_WaveClock_CfC:
    """Build the dual-memory WaveClock CfC classifier."""
    return DualMem_WaveClock_CfC(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        **model_kwargs,
    )


class _DualMemoryLiquidLaplaceHead(nn.Module):
    """Monotone liquid clock and exact stable complex-pole recurrence."""

    def __init__(
        self,
        *,
        n_feature_maps: int,
        wavelet_context_dim: int,
        sampling_rate_hz: float,
        checkpoint_size: int,
        checkpoint_stride: int,
        clock_hidden_size: int,
        clock_min_speed: float,
        clock_max_speed: float,
        n_laplace_poles: int,
        min_pole_decay: float,
        max_pole_frequency_hz: float,
        memory_slots: int,
        memory_key_dim: int,
        memory_topk: int,
        memory_temperature: float,
        memory_novelty_threshold: float,
        memory_age_penalty: float,
        backbone_units: int,
        backbone_layers: int,
        backbone_dropout: float,
        activation_checkpoint: bool,
        memory_detach: bool,
        consistency_weight: float,
        clock_identity_weight: float,
        clock_smoothness_weight: float,
        key_alignment_weight: float,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.n_feature_maps = _positive_integer("n_feature_maps", n_feature_maps)
        self.wavelet_context_dim = _positive_integer(
            "wavelet_context_dim", wavelet_context_dim
        )
        self.sampling_rate_hz = _positive_finite(
            "sampling_rate_hz", sampling_rate_hz
        )
        self.checkpoint_size = _positive_integer(
            "checkpoint_size", checkpoint_size
        )
        self.checkpoint_stride = _positive_integer(
            "checkpoint_stride", checkpoint_stride
        )
        self.clock_hidden_size = _positive_integer(
            "clock_hidden_size", clock_hidden_size
        )
        if self.clock_hidden_size == 1:
            raise ValueError(
                "clock_hidden_size must exceed 1 because the vendored CfC "
                "squeezes singleton timespan dimensions"
            )
        self.clock_min_speed = _positive_finite(
            "clock_min_speed", clock_min_speed
        )
        self.clock_max_speed = _positive_finite(
            "clock_max_speed", clock_max_speed
        )
        if not self.clock_min_speed < 1.0 < self.clock_max_speed:
            raise ValueError("clock speed bounds must strictly contain 1.0")

        self.n_laplace_poles = _positive_integer(
            "n_laplace_poles", n_laplace_poles
        )
        self.min_pole_decay = _positive_finite("min_pole_decay", min_pole_decay)
        self.max_pole_frequency_hz = _positive_finite(
            "max_pole_frequency_hz", max_pole_frequency_hz
        )
        checkpoint_nyquist = self.sampling_rate_hz / (
            2.0 * self.checkpoint_stride
        )
        latent_frequency_limit = checkpoint_nyquist / self.clock_max_speed
        if self.max_pole_frequency_hz > latent_frequency_limit:
            raise ValueError(
                "max_pole_frequency_hz * clock_max_speed must not exceed "
                f"checkpoint Nyquist ({checkpoint_nyquist:g} Hz); maximum "
                f"latent frequency is {latent_frequency_limit:g} Hz"
            )

        self.memory_slots = _positive_integer("laplace_memory_slots", memory_slots)
        self.memory_key_dim = _positive_integer(
            "laplace_memory_key_dim", memory_key_dim
        )
        self.memory_topk = _positive_integer("laplace_memory_topk", memory_topk)
        if self.memory_topk > self.memory_slots:
            raise ValueError("laplace_memory_topk must not exceed memory slots")
        self.memory_temperature = _positive_finite(
            "laplace_memory_temperature", memory_temperature
        )
        self.memory_novelty_threshold = _unit_interval(
            "laplace_memory_novelty_threshold", memory_novelty_threshold
        )
        self.memory_age_penalty = _nonnegative_finite(
            "laplace_memory_age_penalty", memory_age_penalty
        )
        if not isinstance(activation_checkpoint, bool):
            raise ValueError("activation_checkpoint must be a boolean")
        if not isinstance(memory_detach, bool):
            raise ValueError("memory_detach must be a boolean")
        if not memory_detach:
            raise ValueError(
                "memory_detach=False is intentionally disabled: retaining "
                "all out-of-place cache versions has quadratic graph memory"
            )
        self.activation_checkpoint = activation_checkpoint
        self.memory_detach = memory_detach

        self.consistency_weight = _unit_interval(
            "cross_view_consistency_weight", consistency_weight
        )
        self.clock_identity_weight = _unit_interval(
            "clock_identity_weight", clock_identity_weight
        )
        self.clock_smoothness_weight = _unit_interval(
            "clock_smoothness_weight", clock_smoothness_weight
        )
        self.key_alignment_weight = _unit_interval(
            "key_alignment_weight", key_alignment_weight
        )

        backbone_units = _positive_integer("clock_backbone_units", backbone_units)
        backbone_layers = _positive_integer("clock_backbone_layers", backbone_layers)
        backbone_dropout = _dropout_probability(
            "clock_backbone_dropout", backbone_dropout
        )
        self.clock_cfc = CfC(
            input_size=self.n_feature_maps,
            units=self.clock_hidden_size,
            proj_size=None,
            return_sequences=True,
            batch_first=True,
            mixed_memory=False,
            mode="default",
            activation="lecun_tanh",
            backbone_units=backbone_units,
            backbone_layers=backbone_layers,
            backbone_dropout=backbone_dropout,
        )
        self.speed_head = nn.Linear(self.clock_hidden_size, 1)
        nn.init.zeros_(self.speed_head.weight)
        speed_fraction = torch.tensor(
            (1.0 - self.clock_min_speed)
            / (self.clock_max_speed - self.clock_min_speed),
            dtype=torch.float32,
        )
        nn.init.constant_(self.speed_head.bias, float(_logit(speed_fraction)))

        target_decay = torch.linspace(
            0.15, 1.5, self.n_laplace_poles, dtype=torch.float32
        )
        target_decay = (target_decay - self.min_pole_decay).clamp_min(1e-4)
        self.raw_pole_decay = nn.Parameter(_inverse_softplus(target_decay))
        target_frequency = torch.linspace(
            0.5,
            0.8 * self.max_pole_frequency_hz,
            self.n_laplace_poles,
            dtype=torch.float32,
        )
        self.raw_pole_frequency = nn.Parameter(
            _logit(target_frequency / self.max_pole_frequency_hz)
        )
        initial_gain = torch.full(
            (self.n_feature_maps, self.n_laplace_poles),
            1.0 / math.sqrt(self.n_laplace_poles),
        )
        self.raw_pole_input_gain = nn.Parameter(
            torch.atanh((initial_gain / 2.0).clamp(-0.999, 0.999))
        )

        pole_summary_dim = 2 * self.n_laplace_poles
        self.laplace_query = nn.Linear(
            self.clock_hidden_size, self.memory_key_dim, bias=False
        )
        self.laplace_key = nn.Linear(
            pole_summary_dim, self.memory_key_dim, bias=False
        )
        self.laplace_readout = nn.Sequential(
            nn.Linear(pole_summary_dim, self.memory_key_dim),
            nn.GELU(),
            nn.LayerNorm(self.memory_key_dim),
        )
        self.input_gain_head = nn.Linear(
            self.clock_hidden_size + self.memory_key_dim,
            self.n_feature_maps,
        )
        # At initialization cache retrieval cannot destabilize the recurrence.
        nn.init.zeros_(self.input_gain_head.weight)
        nn.init.zeros_(self.input_gain_head.bias)
        self.wavelet_drive_head = nn.Linear(
            self.wavelet_context_dim, self.n_laplace_poles
        )

        self.embedding_dim = (
            2 * self.n_feature_maps * self.n_laplace_poles
            + self.memory_key_dim
            + self.wavelet_context_dim
        )
        self.embedding_norm = nn.LayerNorm(self.embedding_dim)
        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)

        self._auxiliary_loss: Optional[Tensor] = None
        self._auxiliary_components: dict[str, Tensor] = {}
        self._last_diagnostics: dict[str, Tensor] = {}

    def _poles(self, reference: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        decay = F.softplus(self.raw_pole_decay) + self.min_pole_decay
        frequency = self.max_pole_frequency_hz * torch.sigmoid(
            self.raw_pole_frequency
        )
        omega = 2.0 * math.pi * frequency
        return (
            decay.to(device=reference.device, dtype=reference.dtype),
            omega.to(device=reference.device, dtype=reference.dtype),
            frequency.to(device=reference.device, dtype=reference.dtype),
        )

    @staticmethod
    def _transition_factor(
        delta_time: Tensor, decay: Tensor, omega: Tensor
    ) -> tuple[Tensor, Tensor]:
        magnitude = torch.exp(-delta_time.unsqueeze(-1) * decay)
        angle = delta_time.unsqueeze(-1) * omega
        return magnitude * torch.cos(angle), -magnitude * torch.sin(angle)

    @staticmethod
    def _zoh_factor(
        delta_time: Tensor, decay: Tensor, omega: Tensor
    ) -> tuple[Tensor, Tensor]:
        er, ei = _DualMemoryLiquidLaplaceHead._transition_factor(
            delta_time, decay, omega
        )
        numerator_real = 1.0 - er
        numerator_imag = -ei
        denominator = decay.square() + omega.square()
        real = (
            numerator_real * decay + numerator_imag * omega
        ) / denominator
        imag = (
            numerator_imag * decay - numerator_real * omega
        ) / denominator
        return real, imag

    def _transport_cached(
        self,
        cached: Tensor,
        age: Tensor,
        decay: Tensor,
        omega: Tensor,
    ) -> Tensor:
        er, ei = self._transition_factor(age.clamp_min(0.0), decay, omega)
        er = er.unsqueeze(-2)
        ei = ei.unsqueeze(-2)
        real = cached[..., 0] * er - cached[..., 1] * ei
        imag = cached[..., 0] * ei + cached[..., 1] * er
        return torch.stack((real, imag), dim=-1)

    def forward(
        self,
        input: Tensor,
        wavelet_context: Tensor,
        wavelet_memory_active: Tensor,
    ) -> Tensor:
        if input.ndim != 4:
            raise ValueError(
                "liquid Laplace head expects (B,F,1,P), "
                f"got {tuple(input.shape)}"
            )
        if input.dtype not in (torch.float32, torch.float64):
            raise TypeError(
                "liquid Laplace dynamics require float32/float64; disable "
                "FP16/BF16 autocast for this model"
            )
        batch_size, feature_maps, spatial_size, n_patches = input.shape
        if feature_maps != self.n_feature_maps or spatial_size != 1:
            raise ValueError(
                f"expected (B,{self.n_feature_maps},1,P), got "
                f"{tuple(input.shape)}"
            )
        if wavelet_context.shape != (
            batch_size,
            n_patches,
            self.wavelet_context_dim,
        ):
            raise ValueError("wavelet context shape does not match spatial features")
        if wavelet_memory_active.shape != (batch_size, n_patches):
            raise ValueError("wavelet memory mask must have shape (B,P)")

        sequence = input[:, :, 0, :].transpose(1, 2).contiguous()
        physical_dt = input.new_full(
            (n_patches,), self.checkpoint_stride / self.sampling_rate_hz
        )
        physical_dt[0] = self.checkpoint_size / self.sampling_rate_hz
        clock_hidden = _run_cfc_sequence(
            self.clock_cfc,
            sequence,
            physical_dt,
            hidden_size=self.clock_hidden_size,
            activation_checkpoint=self.activation_checkpoint,
        )
        speed = self.clock_min_speed + (
            self.clock_max_speed - self.clock_min_speed
        ) * torch.sigmoid(self.speed_head(clock_hidden).squeeze(-1))
        warped_dt = speed * physical_dt.view(1, -1)
        warped_time = warped_dt.cumsum(dim=1)

        decay, omega, frequency = self._poles(input)
        # Bounded drive plus positive decay gives a BIBO-stable pole branch.
        pole_gain = 2.0 * torch.tanh(self.raw_pole_input_gain)
        pole_gain = pole_gain.to(device=input.device, dtype=input.dtype)
        state = input.new_zeros(
            batch_size,
            self.n_feature_maps,
            self.n_laplace_poles,
            2,
        )
        wavelet_state = input.new_zeros(
            batch_size, self.n_laplace_poles, 2
        )

        # Exact pole values are stored with their warped timestamps. They are
        # analytically transported to the current time before aggregation.
        cache_keys = input.new_zeros(
            batch_size, self.memory_slots, self.memory_key_dim
        )
        cache_values = input.new_zeros(
            batch_size,
            self.memory_slots,
            self.n_feature_maps,
            self.n_laplace_poles,
            2,
        )
        cache_times = input.new_zeros(batch_size, self.memory_slots)
        cache_valid = torch.zeros(
            batch_size,
            self.memory_slots,
            device=input.device,
            dtype=torch.bool,
        )
        cache_pointer = torch.zeros(
            batch_size, device=input.device, dtype=torch.long
        )

        consistency_terms = []
        consistency_masks = []
        alignment_terms = []
        read_descriptors = []
        laplace_active_steps = []
        laplace_novelty_steps = []
        for patch_index in range(n_patches):
            query = F.normalize(
                self.laplace_query(clock_hidden[:, patch_index]),
                dim=-1,
                eps=1e-6,
            )
            cache_age = (
                warped_time[:, patch_index].unsqueeze(-1) - cache_times
            ).clamp_min(0.0)
            _, _, indices, weights, novelty = _read_fixed_cache(
                query,
                cache_keys,
                # Only indices/weights are needed before timestamp transport;
                # passing compact keys avoids gathering the large pole tensor
                # twice at every checkpoint.
                cache_keys,
                cache_valid,
                topk=self.memory_topk,
                temperature=self.memory_temperature,
                score_bias=-self.memory_age_penalty * cache_age,
            )
            active = cache_valid.any(dim=-1)
            selected_values = _gather_cache(
                cache_values, indices, cache_valid.ndim
            )
            selected_times = torch.gather(cache_times, dim=-1, index=indices)
            age = warped_time[:, patch_index].unsqueeze(-1) - selected_times
            transported = self._transport_cached(
                selected_values, age, decay, omega
            )
            transported_weights = weights.view(
                batch_size, weights.shape[-1], 1, 1, 1
            )
            retrieved_state = (transported * transported_weights).sum(dim=1)
            retrieved_summary = retrieved_state.mean(dim=1).reshape(
                batch_size, -1
            )
            read_descriptor = self.laplace_readout(retrieved_summary)
            read_descriptor = read_descriptor * active.unsqueeze(-1).to(
                dtype=read_descriptor.dtype
            )
            read_descriptors.append(read_descriptor)

            # Memory can only modulate a bounded input gain. It never alters
            # the stable homogeneous transition exp(-(a+i*w)*dt).
            bounded_input_gain = 2.0 * torch.sigmoid(
                self.input_gain_head(
                    torch.cat(
                        (clock_hidden[:, patch_index], read_descriptor), dim=-1
                    )
                )
            )
            dt = warped_dt[:, patch_index]
            er, ei = self._transition_factor(dt, decay, omega)
            br, bi = self._zoh_factor(dt, decay, omega)
            er_f = er.unsqueeze(1)
            ei_f = ei.unsqueeze(1)
            drive = (
                torch.tanh(sequence[:, patch_index]).unsqueeze(-1)
                * bounded_input_gain.unsqueeze(-1)
                * pole_gain.unsqueeze(0)
            )
            next_real = (
                state[..., 0] * er_f
                - state[..., 1] * ei_f
                + drive * br.unsqueeze(1)
            )
            next_imag = (
                state[..., 0] * ei_f
                + state[..., 1] * er_f
                + drive * bi.unsqueeze(1)
            )
            state = torch.stack((next_real, next_imag), dim=-1)

            # A second pole state is driven only by the wavelet-memory stream,
            # but uses the identical clock and poles. Their agreement is the
            # cross-domain regularizer; it is not fed into the stable dynamics.
            wavelet_drive = self.wavelet_drive_head(
                wavelet_context[:, patch_index]
            )
            wavelet_real = (
                wavelet_state[..., 0] * er
                - wavelet_state[..., 1] * ei
                + wavelet_drive * br
            )
            wavelet_imag = (
                wavelet_state[..., 0] * ei
                + wavelet_state[..., 1] * er
                + wavelet_drive * bi
            )
            wavelet_state = torch.stack((wavelet_real, wavelet_imag), dim=-1)
            actual_summary = state.mean(dim=1)
            wavelet_magnitude = torch.sqrt(
                wavelet_state.square().sum(dim=-1) + 1e-8
            )
            actual_magnitude = torch.sqrt(
                state.square().sum(dim=-1) + 1e-8
            ).mean(dim=1)
            consistency_terms.append(
                F.smooth_l1_loss(
                    wavelet_magnitude,
                    actual_magnitude.detach(),
                    reduction="none",
                ).mean(dim=-1)
            )
            consistency_masks.append(wavelet_memory_active[:, patch_index])

            current_summary = actual_summary.reshape(batch_size, -1)
            current_key = F.normalize(
                self.laplace_key(current_summary), dim=-1, eps=1e-6
            )
            centered_query = query - query.mean(dim=0, keepdim=True)
            centered_key = current_key - current_key.mean(dim=0, keepdim=True)
            centered_alignment = 1.0 - F.cosine_similarity(
                centered_query, centered_key, dim=-1, eps=1e-6
            ).mean()
            variance_floor = 0.05
            variance_penalty = F.relu(
                variance_floor
                - torch.sqrt(query.var(dim=0, unbiased=False) + 1e-4)
            ).mean() + F.relu(
                variance_floor
                - torch.sqrt(current_key.var(dim=0, unbiased=False) + 1e-4)
            ).mean()
            alignment_terms.append(centered_alignment + variance_penalty)
            should_write = (~active) | (
                novelty >= self.memory_novelty_threshold
            )
            (
                cache_keys,
                cache_values,
                cache_valid,
                cache_pointer,
                selector,
            ) = _write_fixed_cache(
                cache_keys,
                cache_values,
                cache_valid,
                cache_pointer,
                current_key,
                state,
                should_write,
                detach=self.memory_detach,
            )
            write_time = warped_time[:, patch_index]
            if self.memory_detach:
                write_time = write_time.detach()
            cache_times = torch.where(
                selector, write_time.unsqueeze(-1), cache_times
            )
            laplace_active_steps.append(active)
            laplace_novelty_steps.append(novelty)

        consistency = torch.stack(consistency_terms, dim=1)
        consistency_mask = torch.stack(consistency_masks, dim=1).to(
            dtype=consistency.dtype
        )
        consistency_loss = (
            (consistency * consistency_mask).sum()
            / consistency_mask.sum().clamp_min(1.0)
        )
        identity_loss = (speed - 1.0).square().mean()
        if n_patches > 1:
            smoothness_loss = (speed[:, 1:] - speed[:, :-1]).square().mean()
        else:
            smoothness_loss = speed.new_zeros(())
        alignment_loss = torch.stack(alignment_terms).mean()
        auxiliary_loss = (
            self.consistency_weight * consistency_loss
            + self.clock_identity_weight * identity_loss
            + self.clock_smoothness_weight * smoothness_loss
            + self.key_alignment_weight * alignment_loss
        )
        self._auxiliary_loss = auxiliary_loss
        self._auxiliary_components = {
            "cross_view_consistency": consistency_loss.detach(),
            "clock_identity": identity_loss.detach(),
            "clock_smoothness": smoothness_loss.detach(),
            "key_alignment": alignment_loss.detach(),
        }

        mean_read = torch.stack(read_descriptors, dim=1).mean(dim=1)
        mean_wavelet = wavelet_context.mean(dim=1)
        embedding = torch.cat(
            (state.reshape(batch_size, -1), mean_read, mean_wavelet), dim=-1
        )
        embedding = self.embedding_norm(embedding)
        self._last_diagnostics = {
            "clock_speed": speed.detach(),
            "warped_time": warped_time.detach(),
            "pole_decay": decay.detach(),
            "latent_pole_frequency_hz": frequency.detach(),
            "effective_pole_decay": (
                speed.unsqueeze(-1) * decay.view(1, 1, -1)
            ).detach(),
            "effective_pole_frequency_hz": (
                speed.unsqueeze(-1) * frequency.view(1, 1, -1)
            ).detach(),
            "laplace_memory_active": torch.stack(
                laplace_active_steps, dim=1
            ).detach(),
            "laplace_memory_novelty": torch.stack(
                laplace_novelty_steps, dim=1
            ).detach(),
            "laplace_cache_fill": cache_valid.float().mean(dim=-1).detach(),
            "final_pole_state": state.detach(),
        }
        return embedding

    def auxiliary_loss(self, *, clear: bool = False) -> Tensor:
        if self._auxiliary_loss is None:
            return self.raw_pole_decay.new_zeros(())
        loss = self._auxiliary_loss
        if clear:
            self._auxiliary_loss = None
        return loss

    def auxiliary_components(self) -> dict[str, Tensor]:
        return dict(self._auxiliary_components)

    def diagnostics(self) -> dict[str, Tensor]:
        return dict(self._last_diagnostics)


if __name__ == "__main__":
    torch.manual_seed(42)
    model = build_dualmem_waveclock_cfc(
        n_classes=4,
        n_channels=16,
        n_times=129,
        patch_size=16,
        patch_stride=4,
        n_electrode_groups=4,
        wavelet_levels=4,
        wavelet_memory_slots=8,
        wavelet_memory_topk=2,
        laplace_memory_slots=8,
        laplace_memory_topk=2,
        max_pole_frequency_hz=6.0,
        sampling_rate_hz=256.0,
    )
    eeg = torch.randn(3, 16, 129, requires_grad=True)
    logits = model(eeg)
    if logits.shape != (3, 4):
        raise RuntimeError(f"unexpected logits shape: {tuple(logits.shape)}")
    if not torch.isfinite(logits).all():
        raise RuntimeError("non-finite logits")
    loss = logits.square().mean() + model.memory_auxiliary_loss_weight * (
        model.memory_auxiliary_loss()
    )
    loss.backward()
    if model.pool_1.raw_pole_decay.grad is None:
        raise RuntimeError("Laplace poles did not receive gradients")
    diagnostics = model.memory_diagnostics()
    if torch.any(diagnostics["clock_speed"] <= 0):
        raise RuntimeError("liquid clock is not strictly monotone")
    print("DualMem_WaveClock_CfC self-test passed")
