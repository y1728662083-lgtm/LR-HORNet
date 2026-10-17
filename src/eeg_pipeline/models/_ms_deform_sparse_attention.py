"""Vector-token EEG adapters for Deformable-DETR's MSDeformAttn operator.

The sampling/aggregation kernel follows Deformable DETR (SenseTime, 2020,
Apache-2.0).  The EEG tensor adaptation, one-dimensional spatial restriction,
orthogonal-axis position encoding, chunking, and gated residual connection are
specific to this project.  One token is the complete feature vector at one
``(electrode, time)`` location; feature maps are never split into independent
scalar attention routes.

The compiled CUDA extension is loaded lazily so importing the model factory
does not break unrelated models.  ``backend="pytorch"`` uses the reference
implementation published with Deformable DETR and is intended for CPU tests;
production configurations request the official compiled CUDA operator.
"""

from __future__ import annotations

import importlib
import math
import operator
import sys
from pathlib import Path
from types import ModuleType
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


__all__ = [
    "FeatureWiseMSDeformSparseAttention",
    "VectorTokenMSDeformSparseAttention",
]


AttentionMode = Literal["space_1d", "spacetime_2d"]
AttentionBackend = Literal["cuda", "pytorch"]

_OFFICIAL_FUNCTION = None
_OFFICIAL_IMPORT_ERROR: BaseException | None = None
_OFFICIAL_IMPORT_ATTEMPTED = False


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


def _load_official_msda_function():
    """Load the vendored official CUDA autograd function on first use."""
    global _OFFICIAL_FUNCTION
    global _OFFICIAL_IMPORT_ATTEMPTED
    global _OFFICIAL_IMPORT_ERROR

    if _OFFICIAL_IMPORT_ATTEMPTED:
        return _OFFICIAL_FUNCTION
    _OFFICIAL_IMPORT_ATTEMPTED = True

    ops_models_dir = (
        Path(__file__).resolve().parents[2]
        / "Deformable-DETR-main"
        / "models"
    )
    # The vendored package is imported as ops.*, while its legacy autograd
    # function imports the in-place-built extension as a top-level module.
    # Both directories are therefore required for direct main.py runs.
    import_paths = (ops_models_dir, ops_models_dir / "ops")
    for import_path in reversed(import_paths):
        if str(import_path) not in sys.path:
            sys.path.insert(0, str(import_path))
    try:
        module: ModuleType = importlib.import_module(
            "ops.functions.ms_deform_attn_func"
        )
        _OFFICIAL_FUNCTION = module.MSDeformAttnFunction
    except BaseException as error:  # preserve the extension's real error
        _OFFICIAL_IMPORT_ERROR = error
        _OFFICIAL_FUNCTION = None
    return _OFFICIAL_FUNCTION


def _ms_deform_attn_core_pytorch(
    value: Tensor,
    value_spatial_shapes: Tensor,
    sampling_locations: Tensor,
    attention_weights: Tensor,
) -> Tensor:
    """Deformable-DETR's published PyTorch reference core.

    Copyright (c) 2020 SenseTime. All Rights Reserved.
    Licensed under the Apache License, Version 2.0.
    Modified from Deformable-Convolution-V2-PyTorch by chengdazhi.
    """
    batch_size, _, n_heads, head_dim = value.shape
    _, n_queries, _, n_levels, n_points, _ = sampling_locations.shape
    spatial_shapes = [
        (int(height), int(width))
        for height, width in value_spatial_shapes.detach().cpu().tolist()
    ]
    split_sizes = [height * width for height, width in spatial_shapes]
    value_list = value.split(split_sizes, dim=1)
    sampling_grids = 2.0 * sampling_locations - 1.0
    sampled_values = []
    for level, (height, width) in enumerate(spatial_shapes):
        value_level = (
            value_list[level]
            .flatten(2)
            .transpose(1, 2)
            .reshape(batch_size * n_heads, head_dim, height, width)
        )
        sampling_grid = (
            sampling_grids[:, :, :, level]
            .transpose(1, 2)
            .flatten(0, 1)
        )
        sampled_values.append(
            F.grid_sample(
                value_level,
                sampling_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
        )
    weights = attention_weights.transpose(1, 2).reshape(
        batch_size * n_heads,
        1,
        n_queries,
        n_levels * n_points,
    )
    output = (
        torch.stack(sampled_values, dim=-2).flatten(-2) * weights
    ).sum(-1)
    return output.view(
        batch_size,
        n_heads * head_dim,
        n_queries,
    ).transpose(1, 2).contiguous()


def _run_msdeform_operator(
    value: Tensor,
    spatial_shapes: Tensor,
    level_start_index: Tensor,
    sampling_locations: Tensor,
    attention_weights: Tensor,
    *,
    backend: AttentionBackend,
    im2col_step: int,
) -> Tensor:
    if backend == "pytorch":
        return _ms_deform_attn_core_pytorch(
            value,
            spatial_shapes,
            sampling_locations,
            attention_weights,
        )

    if not value.is_cuda:
        raise RuntimeError(
            "attention_backend='cuda' requires CUDA tensors; use "
            "attention_backend='pytorch' only for CPU tests."
        )
    official_function = _load_official_msda_function()
    if official_function is None:
        command = (
            "cd src/Deformable-DETR-main/models/ops && "
            "python setup.py build_ext --inplace"
        )
        raise RuntimeError(
            "The official MultiScaleDeformableAttention CUDA extension could "
            f"not be imported. Build it with: {command}. Original error: "
            f"{_OFFICIAL_IMPORT_ERROR!r}"
        ) from _OFFICIAL_IMPORT_ERROR
    # The official CUDA kernel asserts batch % im2col_step == 0. Batch size
    # varies on final train/validation/test batches and, for spatial-only
    # attention, also equals B times the current time-chunk length.
    effective_im2col_step = math.gcd(value.shape[0], im2col_step)
    return official_function.apply(
        value,
        spatial_shapes,
        level_start_index,
        sampling_locations,
        attention_weights,
        effective_im2col_step,
    )


def _axis_sinusoidal_encoding(
    length: int,
    dimension: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if dimension % 2 != 0:
        raise ValueError("each orthogonal position subspace must be even")
    positions = torch.arange(length, device=device, dtype=torch.float32)
    frequencies = 1.0 / (
        10000.0
        ** (
            torch.arange(0, dimension, 2, device=device, dtype=torch.float32)
            / dimension
        )
    )
    angles = positions[:, None] * frequencies[None, :]
    encoding = torch.empty(
        length,
        dimension,
        device=device,
        dtype=torch.float32,
    )
    encoding[:, 0::2] = angles.sin()
    encoding[:, 1::2] = angles.cos()
    return encoding.to(dtype=dtype)


class _MSDeformFeatureRoute(nn.Module):
    """One independent scalar-feature route through MSDeformAttn.

    This is the original route used by the pre-existing FilterNFC SATT/STATT
    families.  Its parameter names are part of their checkpoint contract, so
    it intentionally remains separate from the newer vector-token route.
    """

    def __init__(
        self,
        *,
        mode: AttentionMode,
        d_model: int,
        n_heads: int,
        n_points: int,
        time_chunk_size: int,
        backend: AttentionBackend,
        im2col_step: int,
        position_encoding: str,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if mode not in {"space_1d", "spacetime_2d"}:
            raise ValueError(f"unknown attention mode: {mode!r}")
        if backend not in {"cuda", "pytorch"}:
            raise ValueError(f"unknown attention backend: {backend!r}")
        if position_encoding not in {"orthogonal", "none"}:
            raise ValueError(
                "position_encoding must be 'orthogonal' or 'none'"
            )
        self.mode = mode
        self.d_model = _positive_integer("attention_dim", d_model)
        self.n_heads = _positive_integer("attention_heads", n_heads)
        self.n_points = _positive_integer("attention_points", n_points)
        self.time_chunk_size = _positive_integer(
            "attention_time_chunk_size", time_chunk_size
        )
        self.im2col_step = _positive_integer(
            "attention_im2col_step", im2col_step
        )
        if self.d_model % self.n_heads != 0:
            raise ValueError("attention_dim must be divisible by attention_heads")
        if position_encoding == "orthogonal" and self.d_model % 4 != 0:
            raise ValueError(
                "orthogonal space/time encoding requires attention_dim "
                "divisible by 4"
            )
        self.backend = backend
        self.position_encoding = position_encoding

        factory_kwargs = {"device": device, "dtype": dtype}
        self.content_projection = nn.Linear(1, self.d_model, **factory_kwargs)
        offset_dimensions = 1 if mode == "space_1d" else 2
        self.offset_dimensions = offset_dimensions
        self.sampling_offsets = nn.Linear(
            self.d_model,
            self.n_heads * self.n_points * offset_dimensions,
            **factory_kwargs,
        )
        self.attention_weights = nn.Linear(
            self.d_model,
            self.n_heads * self.n_points,
            **factory_kwargs,
        )
        self.value_projection = nn.Linear(
            self.d_model, self.d_model, **factory_kwargs
        )
        self.output_projection = nn.Linear(
            self.d_model, self.d_model, **factory_kwargs
        )
        self.scalar_readout = nn.Linear(self.d_model, 1, **factory_kwargs)
        self.gate_projection = nn.Linear(1, 1, **factory_kwargs)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.content_projection.weight)
        nn.init.zeros_(self.content_projection.bias)
        nn.init.zeros_(self.sampling_offsets.weight)
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
        nn.init.xavier_uniform_(self.value_projection.weight)
        nn.init.zeros_(self.value_projection.bias)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        nn.init.xavier_uniform_(self.scalar_readout.weight)
        nn.init.zeros_(self.scalar_readout.bias)
        nn.init.zeros_(self.gate_projection.weight)
        nn.init.constant_(self.gate_projection.bias, -2.0)

        if self.mode == "space_1d":
            directions = torch.where(
                torch.arange(self.n_heads) % 2 == 0, 1.0, -1.0
            ).view(self.n_heads, 1, 1)
            radii = torch.arange(1, self.n_points + 1, dtype=torch.float32)
            bias = directions * radii.view(1, self.n_points, 1)
        else:
            angles = (
                torch.arange(self.n_heads, dtype=torch.float32)
                * (2.0 * math.pi / self.n_heads)
            )
            directions = torch.stack((angles.cos(), angles.sin()), dim=-1)
            directions = directions / directions.abs().amax(
                dim=-1, keepdim=True
            )
            bias = directions[:, None, :].repeat(1, self.n_points, 1)
            for point_index in range(self.n_points):
                bias[:, point_index, :] *= point_index + 1
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(
                bias.reshape(-1).to(
                    device=self.sampling_offsets.bias.device,
                    dtype=self.sampling_offsets.bias.dtype,
                )
            )

    def _position_encoding(
        self,
        n_times: int,
        n_chans: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if self.position_encoding == "none":
            return torch.zeros(
                n_times,
                n_chans,
                self.d_model,
                device=device,
                dtype=dtype,
            )
        half = self.d_model // 2
        spatial = _axis_sinusoidal_encoding(
            n_chans, half, device=device, dtype=dtype
        )
        if self.mode == "space_1d":
            temporal = torch.zeros(
                n_times, half, device=device, dtype=dtype
            )
        else:
            temporal = _axis_sinusoidal_encoding(
                n_times, half, device=device, dtype=dtype
            )
        return torch.cat(
            (
                spatial[None, :, :].expand(n_times, -1, -1),
                temporal[:, None, :].expand(-1, n_chans, -1),
            ),
            dim=-1,
        )

    def _attention_output(
        self,
        *,
        query: Tensor,
        value: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
    ) -> Tensor:
        batch_size, n_queries, _ = query.shape
        raw_offsets = self.sampling_offsets(query).view(
            batch_size,
            n_queries,
            self.n_heads,
            1,
            self.n_points,
            self.offset_dimensions,
        )
        if self.mode == "space_1d":
            raw_offsets = torch.cat(
                (raw_offsets, torch.zeros_like(raw_offsets)), dim=-1
            )
        weight_logits = self.attention_weights(query).view(
            batch_size, n_queries, self.n_heads, self.n_points
        )
        attention_weights = F.softmax(weight_logits, dim=-1).unsqueeze(3)
        normalizer = torch.stack(
            (spatial_shapes[:, 1], spatial_shapes[:, 0]), dim=-1
        ).to(device=query.device, dtype=query.dtype)
        sampling_locations = (
            reference_points[:, :, None, :, None, :]
            + raw_offsets / normalizer[None, None, None, :, None, :]
        )
        output = _run_msdeform_operator(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            backend=self.backend,
            im2col_step=self.im2col_step,
        )
        return self.output_projection(output)

    def _project_value(self, source: Tensor) -> Tensor:
        batch_size, source_length, _ = source.shape
        return self.value_projection(source).view(
            batch_size,
            source_length,
            self.n_heads,
            self.d_model // self.n_heads,
        )

    def _forward_space_1d(self, input: Tensor) -> Tensor:
        batch_size, n_chans, n_times = input.shape
        spatial_shapes = torch.tensor(
            [[1, n_chans]], device=input.device, dtype=torch.long
        )
        level_start_index = torch.zeros(
            1, device=input.device, dtype=torch.long
        )
        position = self._position_encoding(
            1, n_chans, device=input.device, dtype=input.dtype
        )[0]
        reference_x = (
            torch.arange(n_chans, device=input.device, dtype=input.dtype) + 0.5
        ) / n_chans
        reference = torch.stack(
            (reference_x, torch.full_like(reference_x, 0.5)), dim=-1
        )[None, :, None, :]

        chunks = []
        for start in range(0, n_times, self.time_chunk_size):
            stop = min(start + self.time_chunk_size, n_times)
            chunk_size = stop - start
            scalar = input[:, :, start:stop].permute(0, 2, 1).reshape(
                batch_size * chunk_size, n_chans, 1
            )
            content = self.content_projection(scalar)
            query = content + position[None, :, :]
            value = self._project_value(content)
            attended = self._attention_output(
                query=query,
                value=value,
                reference_points=reference.expand(
                    batch_size * chunk_size, -1, -1, -1
                ),
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
            )
            scalar_output = self.scalar_readout(attended)
            gate = torch.sigmoid(self.gate_projection(scalar))
            output = scalar + gate * scalar_output
            chunks.append(
                output.reshape(batch_size, chunk_size, n_chans).permute(
                    0, 2, 1
                )
            )
        return torch.cat(chunks, dim=-1)

    def _forward_spacetime_2d(self, input: Tensor) -> Tensor:
        batch_size, n_chans, n_times = input.shape
        scalar = input.permute(0, 2, 1).reshape(
            batch_size, n_times * n_chans, 1
        )
        content = self.content_projection(scalar)
        value = self._project_value(content)
        position = self._position_encoding(
            n_times, n_chans, device=input.device, dtype=input.dtype
        ).reshape(n_times * n_chans, self.d_model)
        spatial_shapes = torch.tensor(
            [[n_times, n_chans]], device=input.device, dtype=torch.long
        )
        level_start_index = torch.zeros(
            1, device=input.device, dtype=torch.long
        )

        chunks = []
        channel_reference = (
            torch.arange(n_chans, device=input.device, dtype=input.dtype) + 0.5
        ) / n_chans
        for start in range(0, n_times, self.time_chunk_size):
            stop = min(start + self.time_chunk_size, n_times)
            query_indices = torch.arange(
                start * n_chans, stop * n_chans, device=input.device
            )
            query_content = content.index_select(1, query_indices)
            query = query_content + position.index_select(
                0, query_indices
            )[None, :, :]
            time_reference = (
                torch.arange(
                    start, stop, device=input.device, dtype=input.dtype
                )
                + 0.5
            ) / n_times
            reference_x = channel_reference[None, :].expand(stop - start, -1)
            reference_y = time_reference[:, None].expand(-1, n_chans)
            reference = torch.stack(
                (reference_x, reference_y), dim=-1
            ).reshape(1, (stop - start) * n_chans, 1, 2)
            attended = self._attention_output(
                query=query,
                value=value,
                reference_points=reference.expand(batch_size, -1, -1, -1),
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
            )
            scalar_output = self.scalar_readout(attended)
            query_scalar = scalar.index_select(1, query_indices)
            gate = torch.sigmoid(self.gate_projection(query_scalar))
            chunks.append(query_scalar + gate * scalar_output)
        output = torch.cat(chunks, dim=1)
        return output.reshape(batch_size, n_times, n_chans).permute(0, 2, 1)

    def forward(self, input: Tensor) -> Tensor:
        if input.ndim != 3:
            raise ValueError(
                "feature route expects (B,C,T), got "
                f"{tuple(input.shape)}"
            )
        if self.mode == "space_1d":
            return self._forward_space_1d(input)
        return self._forward_spacetime_2d(input)


class _MSDeformVectorRoute(nn.Module):
    """One MSDeformAttn route over full feature-vector EEG tokens."""

    def __init__(
        self,
        *,
        n_features: int,
        mode: AttentionMode,
        d_model: int,
        n_heads: int,
        n_points: int,
        time_chunk_size: int,
        backend: AttentionBackend,
        im2col_step: int,
        position_encoding: str,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if mode not in {"space_1d", "spacetime_2d"}:
            raise ValueError(f"unknown attention mode: {mode!r}")
        if backend not in {"cuda", "pytorch"}:
            raise ValueError(f"unknown attention backend: {backend!r}")
        if position_encoding not in {"orthogonal", "none"}:
            raise ValueError(
                "position_encoding must be 'orthogonal' or 'none'"
            )
        self.mode = mode
        self.n_features = _positive_integer("n_features", n_features)
        self.d_model = _positive_integer("attention_dim", d_model)
        self.n_heads = _positive_integer("attention_heads", n_heads)
        self.n_points = _positive_integer("attention_points", n_points)
        self.time_chunk_size = _positive_integer(
            "attention_time_chunk_size", time_chunk_size
        )
        self.im2col_step = _positive_integer("attention_im2col_step", im2col_step)
        if self.d_model % self.n_heads != 0:
            raise ValueError("attention_dim must be divisible by attention_heads")
        if position_encoding == "orthogonal" and self.d_model % 4 != 0:
            raise ValueError(
                "orthogonal space/time encoding requires attention_dim "
                "divisible by 4"
            )
        self.backend = backend
        self.position_encoding = position_encoding

        factory_kwargs = {"device": device, "dtype": dtype}
        self.content_projection = nn.Linear(
            self.n_features,
            self.d_model,
            **factory_kwargs,
        )
        offset_dimensions = 1 if mode == "space_1d" else 2
        self.offset_dimensions = offset_dimensions
        self.sampling_offsets = nn.Linear(
            self.d_model,
            self.n_heads * self.n_points * offset_dimensions,
            **factory_kwargs,
        )
        self.attention_weights = nn.Linear(
            self.d_model,
            self.n_heads * self.n_points,
            **factory_kwargs,
        )
        self.value_projection = nn.Linear(
            self.d_model,
            self.d_model,
            **factory_kwargs,
        )
        self.output_projection = nn.Linear(
            self.d_model,
            self.d_model,
            **factory_kwargs,
        )
        self.feature_readout = nn.Linear(
            self.d_model,
            self.n_features,
            **factory_kwargs,
        )
        self.gate_projection = nn.Linear(
            self.n_features,
            self.n_features,
            **factory_kwargs,
        )
        self._geometry_key = None
        self.register_buffer(
            "_cached_position",
            torch.empty(0),
            persistent=False,
        )
        self.register_buffer(
            "_cached_reference",
            torch.empty(0),
            persistent=False,
        )
        self.register_buffer(
            "_cached_spatial_shapes",
            torch.empty(0, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_cached_level_start_index",
            torch.empty(0, dtype=torch.long),
            persistent=False,
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.content_projection.weight)
        nn.init.zeros_(self.content_projection.bias)
        nn.init.zeros_(self.sampling_offsets.weight)
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
        nn.init.xavier_uniform_(self.value_projection.weight)
        nn.init.zeros_(self.value_projection.bias)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        nn.init.xavier_uniform_(self.feature_readout.weight)
        nn.init.zeros_(self.feature_readout.bias)
        nn.init.zeros_(self.gate_projection.weight)
        nn.init.constant_(self.gate_projection.bias, -2.0)

        if self.mode == "space_1d":
            directions = torch.where(
                torch.arange(self.n_heads) % 2 == 0,
                1.0,
                -1.0,
            ).view(self.n_heads, 1, 1)
            radii = torch.arange(1, self.n_points + 1, dtype=torch.float32)
            bias = directions * radii.view(1, self.n_points, 1)
        else:
            angles = (
                torch.arange(self.n_heads, dtype=torch.float32)
                * (2.0 * math.pi / self.n_heads)
            )
            directions = torch.stack((angles.cos(), angles.sin()), dim=-1)
            directions = directions / directions.abs().amax(
                dim=-1,
                keepdim=True,
            )
            bias = directions[:, None, :].repeat(1, self.n_points, 1)
            for point_index in range(self.n_points):
                bias[:, point_index, :] *= point_index + 1
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(
                bias.reshape(-1).to(
                    device=self.sampling_offsets.bias.device,
                    dtype=self.sampling_offsets.bias.dtype,
                )
            )

    def _position_encoding(
        self,
        n_times: int,
        n_chans: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if self.position_encoding == "none":
            return torch.zeros(
                n_times,
                n_chans,
                self.d_model,
                device=device,
                dtype=dtype,
            )
        half = self.d_model // 2
        spatial = _axis_sinusoidal_encoding(
            n_chans,
            half,
            device=device,
            dtype=dtype,
        )
        if self.mode == "space_1d":
            temporal = torch.zeros(
                n_times,
                half,
                device=device,
                dtype=dtype,
            )
        else:
            temporal = _axis_sinusoidal_encoding(
                n_times,
                half,
                device=device,
                dtype=dtype,
            )
        return torch.cat(
            (
                spatial[None, :, :].expand(n_times, -1, -1),
                temporal[:, None, :].expand(-1, n_chans, -1),
            ),
            dim=-1,
        )

    def _geometry(
        self,
        n_times: int,
        n_chans: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return cached position/reference tensors for the current geometry."""
        key = (
            self.mode,
            n_times,
            n_chans,
            device.type,
            device.index,
            dtype,
        )
        if self._geometry_key != key:
            position_times = 1 if self.mode == "space_1d" else n_times
            position = self._position_encoding(
                position_times,
                n_chans,
                device=device,
                dtype=dtype,
            )
            channel_reference = (
                torch.arange(n_chans, device=device, dtype=dtype) + 0.5
            ) / n_chans
            if self.mode == "space_1d":
                position = position[0]
                reference = torch.stack(
                    (
                        channel_reference,
                        torch.full_like(channel_reference, 0.5),
                    ),
                    dim=-1,
                )[None, :, None, :]
                spatial_shapes = torch.tensor(
                    [[1, n_chans]],
                    device=device,
                    dtype=torch.long,
                )
            else:
                position = position.reshape(
                    n_times * n_chans,
                    self.d_model,
                )
                time_reference = (
                    torch.arange(n_times, device=device, dtype=dtype) + 0.5
                ) / n_times
                reference_x = channel_reference[None, :].expand(n_times, -1)
                reference_y = time_reference[:, None].expand(-1, n_chans)
                reference = torch.stack(
                    (reference_x, reference_y),
                    dim=-1,
                ).reshape(1, n_times * n_chans, 1, 2)
                spatial_shapes = torch.tensor(
                    [[n_times, n_chans]],
                    device=device,
                    dtype=torch.long,
                )
            self._cached_position = position
            self._cached_reference = reference
            self._cached_spatial_shapes = spatial_shapes
            self._cached_level_start_index = torch.zeros(
                1,
                device=device,
                dtype=torch.long,
            )
            self._geometry_key = key
        return (
            self._cached_position,
            self._cached_reference,
            self._cached_spatial_shapes,
            self._cached_level_start_index,
        )

    def _attention_output(
        self,
        *,
        query: Tensor,
        value: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
    ) -> Tensor:
        batch_size, n_queries, _ = query.shape
        raw_offsets = self.sampling_offsets(query)
        raw_offsets = raw_offsets.view(
            batch_size,
            n_queries,
            self.n_heads,
            1,
            self.n_points,
            self.offset_dimensions,
        )
        if self.mode == "space_1d":
            raw_offsets = torch.cat(
                (raw_offsets, torch.zeros_like(raw_offsets)),
                dim=-1,
            )
        weight_logits = self.attention_weights(query).view(
            batch_size,
            n_queries,
            self.n_heads,
            self.n_points,
        )
        attention_weights = F.softmax(weight_logits, dim=-1).unsqueeze(3)
        normalizer = torch.stack(
            (spatial_shapes[:, 1], spatial_shapes[:, 0]),
            dim=-1,
        ).to(device=query.device, dtype=query.dtype)
        sampling_locations = (
            reference_points[:, :, None, :, None, :]
            + raw_offsets
            / normalizer[None, None, None, :, None, :]
        )
        output = _run_msdeform_operator(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            backend=self.backend,
            im2col_step=self.im2col_step,
        )
        return self.output_projection(output)

    def _project_value(self, source: Tensor) -> Tensor:
        batch_size, source_length, _ = source.shape
        return self.value_projection(source).view(
            batch_size,
            source_length,
            self.n_heads,
            self.d_model // self.n_heads,
        )

    def _forward_space_1d(self, input: Tensor) -> Tensor:
        batch_size, n_features, n_chans, n_times = input.shape
        position, reference, spatial_shapes, level_start_index = self._geometry(
            n_times,
            n_chans,
            device=input.device,
            dtype=input.dtype,
        )

        chunks = []
        for start in range(0, n_times, self.time_chunk_size):
            stop = min(start + self.time_chunk_size, n_times)
            chunk_size = stop - start
            tokens = input[:, :, :, start:stop].permute(0, 3, 2, 1).reshape(
                batch_size * chunk_size,
                n_chans,
                n_features,
            )
            content = self.content_projection(tokens)
            query = content + position[None, :, :]
            value = self._project_value(content)
            attended = self._attention_output(
                query=query,
                value=value,
                reference_points=reference.expand(
                    batch_size * chunk_size,
                    -1,
                    -1,
                    -1,
                ),
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
            )
            feature_output = self.feature_readout(attended)
            gate = torch.sigmoid(self.gate_projection(tokens))
            output = tokens + gate * feature_output
            chunks.append(
                output.reshape(
                    batch_size,
                    chunk_size,
                    n_chans,
                    n_features,
                ).permute(
                    0,
                    3,
                    2,
                    1,
                )
            )
        return torch.cat(chunks, dim=-1)

    def _forward_spacetime_2d(self, input: Tensor) -> Tensor:
        batch_size, n_features, n_chans, n_times = input.shape
        tokens = input.permute(0, 3, 2, 1).reshape(
            batch_size,
            n_times * n_chans,
            n_features,
        )
        content = self.content_projection(tokens)
        value = self._project_value(content)
        position, reference, spatial_shapes, level_start_index = self._geometry(
            n_times,
            n_chans,
            device=input.device,
            dtype=input.dtype,
        )

        chunks = []
        for start in range(0, n_times, self.time_chunk_size):
            stop = min(start + self.time_chunk_size, n_times)
            token_start = start * n_chans
            token_stop = stop * n_chans
            query_content = content[:, token_start:token_stop]
            query = query_content + position[token_start:token_stop][None]
            attended = self._attention_output(
                query=query,
                value=value,
                reference_points=reference[:, token_start:token_stop].expand(
                    batch_size,
                    -1,
                    -1,
                    -1,
                ),
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
            )
            feature_output = self.feature_readout(attended)
            query_tokens = tokens[:, token_start:token_stop]
            gate = torch.sigmoid(self.gate_projection(query_tokens))
            chunks.append(query_tokens + gate * feature_output)
        output = torch.cat(chunks, dim=1)
        return output.reshape(
            batch_size,
            n_times,
            n_chans,
            n_features,
        ).permute(0, 3, 2, 1)

    def forward(self, input: Tensor) -> Tensor:
        if input.ndim != 4:
            raise ValueError(
                "vector-token route expects (B,F,C,T), got "
                f"{tuple(input.shape)}"
            )
        _, n_features, _, _ = input.shape
        if n_features != self.n_features:
            raise ValueError(
                f"expected F={self.n_features}, got F={n_features}"
            )
        if self.mode == "space_1d":
            return self._forward_space_1d(input)
        return self._forward_spacetime_2d(input)


class VectorTokenMSDeformSparseAttention(nn.Module):
    """Attend full feature vectors at each electrode-time location."""

    def __init__(
        self,
        *,
        n_features: int,
        n_chans: int,
        mode: AttentionMode,
        d_model: int = 8,
        n_heads: int = 4,
        n_points: int = 4,
        time_chunk_size: int = 64,
        checkpoint_attention: bool = False,
        backend: AttentionBackend = "cuda",
        im2col_step: int = 64,
        position_encoding: str = "orthogonal",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.n_features = _positive_integer("n_features", n_features)
        self.n_chans = _positive_integer("n_chans", n_chans)
        if not isinstance(checkpoint_attention, bool):
            raise TypeError("checkpoint_attention must be a boolean")
        self.mode = mode
        self.checkpoint_attention = checkpoint_attention
        self.route = _MSDeformVectorRoute(
            n_features=self.n_features,
            mode=mode,
            d_model=d_model,
            n_heads=n_heads,
            n_points=n_points,
            time_chunk_size=time_chunk_size,
            backend=backend,
            im2col_step=im2col_step,
            position_encoding=position_encoding,
            device=device,
            dtype=dtype,
        )
        self.d_model = self.route.d_model
        self.n_heads = self.route.n_heads
        self.n_points = self.route.n_points
        self.time_chunk_size = self.route.time_chunk_size
        self.backend = self.route.backend
        self.im2col_step = self.route.im2col_step
        self.position_encoding = self.route.position_encoding
        self.route_count = 1
        self.token_definition = "full_feature_vector_per_electrode_time"

    def _run_route(self, input: Tensor) -> Tensor:
        if (
            self.checkpoint_attention
            and self.training
            and torch.is_grad_enabled()
        ):
            return checkpoint(
                self.route,
                input,
                use_reentrant=False,
            )
        return self.route(input)

    def estimated_operator_calls(self, n_times: int) -> int:
        """Number of sparse-attention operator launches in one forward."""
        return math.ceil(_positive_integer("n_times", n_times) / self.time_chunk_size)

    def forward(self, input: Tensor) -> Tensor:
        if input.ndim != 4:
            raise ValueError(
                "MSDeform sparse attention expects (B,F,C,T), got "
                f"{tuple(input.shape)}"
            )
        _, n_features, n_chans, _ = input.shape
        if n_features != self.n_features or n_chans != self.n_chans:
            raise ValueError(
                f"expected F={self.n_features}, C={self.n_chans}; got "
                f"F={n_features}, C={n_chans}"
            )
        return self._run_route(input)


class FeatureWiseMSDeformSparseAttention(nn.Module):
    """Backward-compatible scalar routes for pre-existing model families.

    Historical FilterNFC SAtt/StAtt modules import this class by name.  Keeping
    their original independent scalar routes prevents this update from
    silently changing those already-defined experiments.  The new Group4
    comparison imports :class:`VectorTokenMSDeformSparseAttention` directly.
    """

    def __init__(
        self,
        *,
        n_features: int,
        n_chans: int,
        mode: AttentionMode,
        d_model: int = 8,
        n_heads: int = 4,
        n_points: int = 4,
        time_chunk_size: int = 4,
        checkpoint_attention: bool = True,
        backend: AttentionBackend = "cuda",
        im2col_step: int = 64,
        position_encoding: str = "orthogonal",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.n_features = _positive_integer("n_features", n_features)
        self.n_chans = _positive_integer("n_chans", n_chans)
        if not isinstance(checkpoint_attention, bool):
            raise TypeError("checkpoint_attention must be a boolean")
        self.mode = mode
        self.checkpoint_attention = checkpoint_attention
        self.routes = nn.ModuleList(
            [
                _MSDeformFeatureRoute(
                    mode=mode,
                    d_model=d_model,
                    n_heads=n_heads,
                    n_points=n_points,
                    time_chunk_size=time_chunk_size,
                    backend=backend,
                    im2col_step=im2col_step,
                    position_encoding=position_encoding,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(self.n_features)
            ]
        )
        self.d_model = self.routes[0].d_model
        self.n_heads = self.routes[0].n_heads
        self.n_points = self.routes[0].n_points
        self.time_chunk_size = self.routes[0].time_chunk_size
        self.backend = self.routes[0].backend
        self.position_encoding = self.routes[0].position_encoding

    def _run_route(self, route: nn.Module, feature: Tensor) -> Tensor:
        if (
            self.checkpoint_attention
            and self.training
            and torch.is_grad_enabled()
        ):
            return checkpoint(
                route,
                feature,
                use_reentrant=False,
            )
        return route(feature)

    def forward(self, input: Tensor) -> Tensor:
        if input.ndim != 4:
            raise ValueError(
                "MSDeform sparse attention expects (B,F,C,T), got "
                f"{tuple(input.shape)}"
            )
        _, n_features, n_chans, _ = input.shape
        if n_features != self.n_features or n_chans != self.n_chans:
            raise ValueError(
                f"expected F={self.n_features}, C={self.n_chans}; got "
                f"F={n_features}, C={n_chans}"
            )
        outputs = [
            self._run_route(route, input[:, feature_index])[:, None]
            for feature_index, route in enumerate(self.routes)
        ]
        return torch.cat(outputs, dim=1)


