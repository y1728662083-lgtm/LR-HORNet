"""Paper4-native serial Patch-LSTM/Mamba augmentation of EEGNet.

The model is intentionally registered under a new name because it is not the
existing ``group4_patchLstm_deformableLstm``.  The existing model *replaces*
the first temporal convolution with Group-4 Patch-LSTMs.  This model keeps the
temporal convolution and augments its input instead::

    z = x + alpha(x) * PatchLSTM(x) + beta(x) * Mamba(x)
    logits = EEGNet_with_post_spatial_deformation(z)

Patch and Mamba branches are independently removable.  Mamba always reads the
original full trial, never the Patch output.  Electrode grouping assumes that
Paper4's dataset has already reordered channels into four contiguous spatial
groups, which is exactly what ``data.n_electrode_groups: 4`` provides.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


__all__ = [
    "group4_serialPatchLstmMamba_deformableLstm",
    "build_group4_serialpatchlstmmamba_deformablelstm",
]


_GATE_MODES = {"off", "global", "electrode", "patch", "patch_mlp"}


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _probability(name: str, value: float) -> float:
    value = float(value)
    if not 0.0 <= value < 1.0:
        raise ValueError(f"{name} must lie in [0, 1), got {value}")
    return value


def _patch_count(length: int, size: int, stride: int) -> int:
    return 1 + max(0, (length - size + stride - 1) // stride)


def _split_patches(x: Tensor, size: int, stride: int) -> Tensor:
    """Convert ``(B,C,T)`` to ``(B,C,N,K)`` with right replicate padding."""
    covered = size + (_patch_count(x.shape[-1], size, stride) - 1) * stride
    if covered > x.shape[-1]:
        x = F.pad(x, (0, covered - x.shape[-1]), mode="replicate")
    return x.unfold(-1, size, stride)


def _overlap_mean(patches: Tensor, length: int, stride: int) -> Tensor:
    """Overlap-add ``(B,C,N,K)`` and divide only by geometric coverage."""
    batch, channels, count, size = patches.shape
    covered = size + (count - 1) * stride
    columns = patches.reshape(batch * channels, count, size).transpose(1, 2)
    arguments = {
        "output_size": (1, covered),
        "kernel_size": (1, size),
        "stride": (1, stride),
    }
    total = F.fold(columns, **arguments)
    coverage = F.fold(patches.new_ones(1, size, count), **arguments)
    return (total / coverage.clamp_min(1.0)).reshape(
        batch, channels, covered
    )[..., :length]


class _FusionCoefficient(nn.Module):
    """Global, electrode, static Patch, or contextual Patch coefficient."""

    def __init__(
        self,
        mode: str,
        channels: int,
        count: int,
        size: int,
        *,
        initial: float = 0.1,
        hidden: int = 16,
    ) -> None:
        super().__init__()
        if mode not in _GATE_MODES - {"off"}:
            raise ValueError(f"Unsupported active coefficient mode {mode!r}")
        if not 0.0 < float(initial) < 1.0:
            raise ValueError("coefficient initialization must lie in (0, 1)")
        self.mode = mode
        initial_logit = math.log(float(initial) / (1.0 - float(initial)))
        if mode == "patch_mlp":
            self.mlp = nn.Sequential(
                nn.Linear(2 * size, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.constant_(self.mlp[-1].bias, initial_logit)
        else:
            shapes = {
                "global": (1, 1, 1, 1),
                "electrode": (1, channels, 1, 1),
                "patch": (1, channels, count, 1),
            }
            self.logit = nn.Parameter(torch.full(shapes[mode], initial_logit))

    def forward(
        self, raw_patch: Tensor | None = None, feature_patch: Tensor | None = None
    ) -> Tensor:
        if self.mode == "patch_mlp":
            if raw_patch is None or feature_patch is None:
                raise ValueError("patch_mlp requires raw and feature patches")
            logits = self.mlp(torch.cat((raw_patch, feature_patch), dim=-1))
        else:
            logits = self.logit
        return torch.sigmoid(logits)


def _make_lstm(output_width: int, cell: str) -> nn.LSTM:
    if cell not in {"lstm", "bilstm"}:
        raise ValueError("cell must be 'lstm' or 'bilstm'")
    directions = 2 if cell == "bilstm" else 1
    if output_width % directions:
        raise ValueError("Bi-LSTM output width must be even")
    return nn.LSTM(
        input_size=1,
        hidden_size=output_width // directions,
        batch_first=True,
        bidirectional=directions == 2,
    )


class _Group4PatchEncoder(nn.Module):
    """Four shared LSTMs; every electrode/Patch has independent state."""

    def __init__(self, channels: int, width: int = 8, cell: str = "lstm") -> None:
        super().__init__()
        base, remainder = divmod(channels, 4)
        self.group_sizes = tuple(base + int(i < remainder) for i in range(4))
        self.lstms = nn.ModuleList([_make_lstm(width, cell) for _ in range(4)])
        self.projection = nn.Linear(width, 1, bias=False)
        nn.init.xavier_uniform_(self.projection.weight)

    def forward(self, patches: Tensor) -> Tensor:
        outputs = []
        for group, recurrent in zip(
            patches.split(self.group_sizes, dim=1), self.lstms
        ):
            batch, channels, count, size = group.shape
            hidden, _ = recurrent(
                group.reshape(batch * channels * count, size, 1)
            )
            outputs.append(
                self.projection(hidden).reshape(
                    batch, channels, count, size
                )
            )
        return torch.cat(outputs, dim=1)


class _ElectrodeMamba(nn.Module):
    """Official full-trial Mamba shared across electrodes."""

    def __init__(
        self, d_model: int = 16, d_state: int = 16, d_conv: int = 4, expand: int = 2
    ) -> None:
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except ImportError as error:
            raise ImportError(
                "A beta_mode other than 'off' requires the official mamba_ssm package"
            ) from error
        self.input_projection = nn.Linear(1, d_model)
        self.mamba = Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.output_projection = nn.Linear(d_model, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, length = x.shape
        sequence = x.reshape(batch * channels, length, 1)
        hidden = self.mamba(self.input_projection(sequence))
        return self.output_projection(hidden).reshape(batch, channels, length)


class _SerialInputFusion(nn.Module):
    """Independently gated local Patch and full-trial Mamba residuals."""

    def __init__(
        self,
        *,
        n_chans: int,
        n_times: int,
        patch_size: int,
        patch_stride: int,
        width: int,
        alpha_mode: str,
        beta_mode: str,
        alpha_init: float,
        beta_init: float,
        patch_cell: str,
        mamba_kwargs: dict | None,
    ) -> None:
        super().__init__()
        if alpha_mode not in _GATE_MODES or beta_mode not in _GATE_MODES:
            raise ValueError(
                f"alpha_mode/beta_mode must lie in {sorted(_GATE_MODES)}"
            )
        if n_chans < 4 or not 1 <= patch_stride <= patch_size:
            raise ValueError("Require n_chans >= 4 and 1 <= stride <= size")
        self.size = patch_size
        self.stride = patch_stride
        self.alpha_mode = alpha_mode
        self.beta_mode = beta_mode
        count = _patch_count(n_times, patch_size, patch_stride)
        self.patch: _Group4PatchEncoder | None = None
        self.mamba: _ElectrodeMamba | None = None
        if alpha_mode != "off":
            self.patch = _Group4PatchEncoder(n_chans, width, patch_cell)
            self.alpha = _FusionCoefficient(
                alpha_mode, n_chans, count, patch_size, initial=alpha_init
            )
        if beta_mode != "off":
            self.mamba = _ElectrodeMamba(**(mamba_kwargs or {}))
            self.beta = _FusionCoefficient(
                beta_mode, n_chans, count, patch_size, initial=beta_init
            )

    def forward(self, x: Tensor) -> Tensor:
        output = x
        raw = (
            _split_patches(x, self.size, self.stride)
            if self.patch is not None or self.beta_mode == "patch_mlp"
            else None
        )
        if self.patch is not None:
            patch_feature = self.patch(raw)
            alpha = self.alpha(raw, patch_feature)
            output = output + _overlap_mean(
                alpha * patch_feature, x.shape[-1], self.stride
            )
        if self.mamba is not None:
            mamba_feature = self.mamba(x)
            contextual = (
                _split_patches(mamba_feature, self.size, self.stride)
                if self.beta_mode == "patch_mlp"
                else None
            )
            beta = self.beta(raw, contextual)
            if self.beta_mode in {"patch", "patch_mlp"}:
                beta = _overlap_mean(
                    beta.expand(*beta.shape[:-1], self.size),
                    x.shape[-1],
                    self.stride,
                )
            else:
                beta = beta.squeeze(-1)
            output = output + beta * mamba_feature
        return output


class _TemporalDeformLSTM(nn.Module):
    """Shared recurrent predictor for post-spatial temporal displacement."""

    def __init__(self, hidden_size: int = 8, cell: str = "lstm") -> None:
        super().__init__()
        self.recurrent = _make_lstm(hidden_size, cell)
        self.offset = nn.Linear(hidden_size, 1)
        self.gate = nn.Linear(hidden_size, 1)
        for head in (self.offset, self.gate):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, x: Tensor) -> Tensor:
        batch, feature_maps, spatial, length = x.shape
        if spatial != 1:
            raise ValueError(f"Expected post-spatial size 1, got {spatial}")
        hidden, _ = self.recurrent(
            x.reshape(batch * feature_maps, length, 1)
        )
        displacement = (
            self.offset(hidden) * torch.sigmoid(self.gate(hidden))
        ).squeeze(-1)
        time = torch.arange(length, device=x.device, dtype=x.dtype) + displacement
        if length == 1:
            time = torch.zeros_like(time)
        else:
            time = 2.0 * time / float(length - 1) - 1.0
        grid = torch.stack((time, torch.zeros_like(time)), dim=-1).unsqueeze(1)
        warped = F.grid_sample(
            x.reshape(batch * feature_maps, 1, 1, length),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return warped.reshape_as(x)


class _SpatialConv(nn.Conv2d):
    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            self.weight.copy_(torch.renorm(self.weight, p=2, dim=0, maxnorm=1.0))
        return super().forward(x)


def _normalization(kind: str, channels: int, groups: int) -> nn.Module:
    normalized = str(kind).casefold()
    if normalized == "batchnorm":
        return nn.BatchNorm2d(channels, momentum=0.01, eps=1e-3)
    if normalized == "groupnorm":
        if channels % groups:
            raise ValueError(
                f"n_norm_groups={groups} must divide feature count {channels}"
            )
        return nn.GroupNorm(groups, channels)
    raise ValueError("normalization must be 'batchnorm' or 'groupnorm'")


class group4_serialPatchLstmMamba_deformableLstm(nn.Module):
    """Standalone Paper4 classifier implementing the serial core."""

    def __init__(
        self,
        n_chans: int,
        n_times: int,
        n_outputs: int,
        *,
        patch_size: int = 32,
        patch_stride: int = 16,
        n_electrode_groups: int = 4,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 128,
        depthwise_kernel_length: int | None = None,
        depthwise_kernel: int | None = None,
        alpha_mode: str = "global",
        beta_mode: str = "off",
        alpha_init: float = 0.1,
        beta_init: float = 0.1,
        patch_cell: str = "lstm",
        deform_cell: str = "lstm",
        drop_prob: float | None = None,
        dropout: float | None = None,
        offset_hidden: int = 8,
        normalization: str = "batchnorm",
        n_norm_groups: int = 4,
        mamba_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        n_chans = _positive_integer("n_chans", n_chans)
        n_times = _positive_integer("n_times", n_times)
        n_outputs = _positive_integer("n_outputs", n_outputs)
        patch_size = _positive_integer("patch_size", patch_size)
        patch_stride = _positive_integer("patch_stride", patch_stride)
        if n_electrode_groups != 4:
            raise ValueError("This model requires n_electrode_groups=4")
        if patch_stride > patch_size:
            raise ValueError("patch_stride must not exceed patch_size")
        if depthwise_kernel_length is not None and depthwise_kernel is not None:
            if int(depthwise_kernel_length) != int(depthwise_kernel):
                raise ValueError(
                    "depthwise_kernel and depthwise_kernel_length disagree"
                )
        depthwise = _positive_integer(
            "depthwise_kernel_length",
            16
            if depthwise_kernel_length is None and depthwise_kernel is None
            else (
                depthwise_kernel_length
                if depthwise_kernel_length is not None
                else depthwise_kernel
            ),
        )
        if drop_prob is not None and dropout is not None:
            if not math.isclose(float(drop_prob), float(dropout)):
                raise ValueError("drop_prob and dropout disagree")
        probability = _probability(
            "drop_prob",
            0.25 if drop_prob is None and dropout is None else (
                drop_prob if drop_prob is not None else dropout
            ),
        )
        self.n_chans = n_chans
        self.n_times = n_times
        self.n_outputs = n_outputs
        self.fusion = _SerialInputFusion(
            n_chans=n_chans,
            n_times=n_times,
            patch_size=patch_size,
            patch_stride=patch_stride,
            width=F1,
            alpha_mode=alpha_mode,
            beta_mode=beta_mode,
            alpha_init=alpha_init,
            beta_init=beta_init,
            patch_cell=patch_cell,
            mamba_kwargs=mamba_kwargs,
        )
        self.conv_temporal = nn.Conv2d(
            1,
            F1,
            (1, kernel_length),
            padding=(0, kernel_length // 2),
            bias=False,
        )
        self.temporal_norm = _normalization(normalization, F1, n_norm_groups)
        self.conv_spatial = _SpatialConv(
            F1, F1 * D, (n_chans, 1), groups=F1, bias=False
        )
        self.spatial_norm = _normalization(
            normalization, F1 * D, n_norm_groups
        )
        self.deformation = _TemporalDeformLSTM(offset_hidden, deform_cell)
        self.pool_1 = nn.AvgPool2d((1, 4))
        self.dropout_1 = nn.Dropout(probability)
        self.depthwise = nn.Conv2d(
            F1 * D,
            F1 * D,
            (1, depthwise),
            groups=F1 * D,
            padding=(0, depthwise // 2),
            bias=False,
        )
        self.pointwise = nn.Conv2d(F1 * D, F2, 1, bias=False)
        self.feature_norm = _normalization(normalization, F2, n_norm_groups)
        self.pool_2 = nn.AvgPool2d((1, 8))
        self.dropout_2 = nn.Dropout(probability)

        temporal_times = n_times + 2 * (kernel_length // 2) - kernel_length + 1
        pool_1_times = temporal_times // 4
        depthwise_times = pool_1_times + 2 * (depthwise // 2) - depthwise + 1
        feature_times = depthwise_times // 8
        if feature_times < 1:
            raise ValueError(
                "The configured temporal kernels/pools collapse the time axis"
            )
        self.classifier = nn.Linear(F2 * feature_times, n_outputs)
        self.activation = nn.ELU()
        self.architecture_formula = (
            "CNN(x + alpha*Group4PatchLSTM(x) + beta*FullTrialMamba(x)); "
            "post-spatial DeformableLSTM"
        )
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.n_electrode_groups = 4
        self.alpha_mode = alpha_mode
        self.beta_mode = beta_mode
        self.normalization = normalization

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[1:] != (self.n_chans, self.n_times):
            raise ValueError(
                f"Expected (B,{self.n_chans},{self.n_times}), got {tuple(x.shape)}"
            )
        x = self.conv_temporal(self.fusion(x).unsqueeze(1))
        x = self.temporal_norm(x)
        x = self.activation(self.spatial_norm(self.conv_spatial(x)))
        x = self.deformation(x)
        x = self.dropout_1(self.pool_1(x))
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.activation(self.feature_norm(x))
        x = self.dropout_2(self.pool_2(x))
        return self.classifier(x.flatten(1))


def build_group4_serialpatchlstmmamba_deformablelstm(
    n_classes: int,
    *,
    n_channels: int = 128,
    n_times: int = 1153,
    **model_kwargs,
) -> group4_serialPatchLstmMamba_deformableLstm:
    return group4_serialPatchLstmMamba_deformableLstm(
        n_chans=n_channels,
        n_times=n_times,
        n_outputs=n_classes,
        **model_kwargs,
    )
