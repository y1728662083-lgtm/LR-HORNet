"""Controlled recurrent/Mamba substitutions for group-four deformable EEGNet.

Only the patch temporal encoder or the post-spatial offset generator is
replaced. Mamba is imported lazily so non-Mamba experiments do not require it.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from eeg_pipeline.models.group4_patchLstm_deformableLstm import (
    group4_patchLstm_deformableLstm,
)


def _mamba_class():
    try:
        from mamba_ssm import Mamba
    except ImportError as error:
        raise ImportError(
            "Mamba ablation requires the official mamba-ssm package. "
            "Install it in the Linux/CUDA training environment; no "
            "surrogate implementation is used."
        ) from error
    return Mamba


class _GroupPatchRecurrent(nn.Module):
    def __init__(self, original: nn.Module, kind: str) -> None:
        super().__init__()
        self.kind = kind
        self.n_chans = original.n_chans
        self.n_features = original.n_features
        self.patch_size = original.patch_size
        self.patch_stride = original.patch_stride
        self.n_electrode_groups = original.n_electrode_groups
        self.chans_per_group = original.chans_per_group
        recurrent = {"gru": nn.GRU, "rnn": nn.RNN}[kind]
        ref = next(original.parameters())
        self.groups = nn.ModuleList([
            recurrent(
                input_size=1, hidden_size=self.n_features,
                num_layers=1, batch_first=True,
                device=ref.device, dtype=ref.dtype,
            )
            for _ in range(self.n_electrode_groups)
        ])

    def forward(self, input: Tensor) -> Tensor:
        if input.ndim != 4 or input.shape[1:3] != (1, self.n_chans):
            raise ValueError("Expected (B,1,C,T) with configured C")
        batch, _, _, times = input.shape
        count = (times - self.patch_size) // self.patch_stride + 1
        if count < 1:
            raise ValueError("Signal is shorter than patch_size")
        length = self.patch_size + (count - 1) * self.patch_stride
        results = []
        for group_index, recurrent in enumerate(self.groups):
            first = group_index * self.chans_per_group
            signal = input[:, 0, first:first + self.chans_per_group, :length]
            patches = signal.unfold(-1, self.patch_size, self.patch_stride)
            sequence = patches.reshape(-1, self.patch_size, 1)
            hidden, _ = recurrent(sequence)
            hidden = hidden.reshape(
                batch, self.chans_per_group, count,
                self.patch_size, self.n_features,
            )
            if count == 1:
                result = hidden[:, :, 0]
            else:
                result = torch.cat((
                    hidden[:, :, 0],
                    hidden[:, :, 1:, self.patch_size-self.patch_stride:]
                    .reshape(batch, self.chans_per_group,
                             (count-1)*self.patch_stride, self.n_features),
                ), dim=2)
            results.append(result)
        return torch.cat(results, dim=1).permute(0, 3, 1, 2).contiguous()


class _GroupFullMamba(nn.Module):
    """One official Mamba block per electrode group, no patch reset."""

    def __init__(self, original: nn.Module, *, d_state: int = 16) -> None:
        super().__init__()
        Mamba = _mamba_class()
        self.n_chans = original.n_chans
        self.n_features = original.n_features
        self.n_electrode_groups = original.n_electrode_groups
        self.chans_per_group = original.chans_per_group
        ref = next(original.parameters())
        self.input_projection = nn.ModuleList([
            nn.Linear(1, self.n_features, device=ref.device, dtype=ref.dtype)
            for _ in range(self.n_electrode_groups)
        ])
        self.groups = nn.ModuleList([
            Mamba(d_model=self.n_features, d_state=d_state, d_conv=4, expand=2)
            for _ in range(self.n_electrode_groups)
        ])
        self.to(device=ref.device, dtype=ref.dtype)

    def forward(self, input: Tensor) -> Tensor:
        if input.ndim != 4 or input.shape[1:3] != (1, self.n_chans):
            raise ValueError("Expected (B,1,C,T) with configured C")
        batch, _, _, times = input.shape
        results = []
        for index, block in enumerate(self.groups):
            first = index * self.chans_per_group
            sequence = input[:, 0, first:first+self.chans_per_group, :]
            sequence = sequence.reshape(batch*self.chans_per_group, times, 1)
            output = block(self.input_projection[index](sequence))
            results.append(output.reshape(
                batch, self.chans_per_group, times, self.n_features
            ))
        return torch.cat(results, dim=1).permute(0, 3, 1, 2).contiguous()


class _OffsetWarp(nn.Module):
    """Same gated offset and grid sampling as the reference LSTM warp."""

    def __init__(self, original: nn.Module, kind: str, d_state: int = 16):
        super().__init__()
        self.n_feature_maps = original.n_feature_maps
        self.hidden_size = original.hidden_size
        self.kind = kind
        ref = next(original.parameters())
        if kind == "mamba":
            Mamba = _mamba_class()
            self.projection = nn.Linear(
                1, self.hidden_size, device=ref.device, dtype=ref.dtype
            )
            self.backbone = Mamba(
                d_model=self.hidden_size, d_state=d_state, d_conv=4, expand=2
            )
            self.backbone.to(device=ref.device, dtype=ref.dtype)
        else:
            recurrent = {"gru": nn.GRU, "rnn": nn.RNN}[kind]
            self.backbone = recurrent(
                1, self.hidden_size, batch_first=True,
                device=ref.device, dtype=ref.dtype,
            )
        self.offset_head = nn.Linear(
            self.hidden_size, 1, device=ref.device, dtype=ref.dtype
        )
        self.gate_head = nn.Linear(
            self.hidden_size, 1, device=ref.device, dtype=ref.dtype
        )
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)
        self.padding_mode = original.padding_mode
        self.align_corners = original.align_corners

    def predict_offsets(self, input: Tensor) -> Tensor:
        batch, maps, spatial, times = input.shape
        if maps != self.n_feature_maps or spatial != 1:
            raise ValueError("Expected spatial features (B,M,1,T)")
        sequence = input[:, :, 0, :].reshape(batch*maps, times, 1)
        if self.kind == "mamba":
            hidden = self.backbone(self.projection(sequence))
        else:
            hidden, _ = self.backbone(sequence)
        return (
            torch.sigmoid(self.gate_head(hidden)) * self.offset_head(hidden)
        ).squeeze(-1).reshape(batch, maps, times)

    def forward(self, input: Tensor) -> Tensor:
        batch, maps, _, times = input.shape
        offset = self.predict_offsets(input).reshape(batch*maps, times)
        signal = input.reshape(batch*maps, 1, 1, times)
        base = torch.arange(times, device=input.device, dtype=input.dtype)[None]
        sample_time = base + offset
        normalized = (
            torch.zeros_like(sample_time) if times == 1
            else 2 * sample_time / (times-1) - 1
        )
        grid = torch.stack((
            normalized[:, None], torch.zeros_like(normalized)[:, None]
        ), dim=-1)
        output = F.grid_sample(
            signal, grid, mode="bilinear", padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )
        return output.reshape(batch, maps, 1, times)


class _ControlledAblation(group4_patchLstm_deformableLstm):
    patch_kind = "lstm"
    offset_kind = "lstm"

    def __init__(self, *args, mamba_d_state: int = 16, **kwargs):
        if self.patch_kind == "mamba":
            # One 1153-step sequence per electrode; no windowing or reset.
            n_times = kwargs.get("n_times", 1153)
            kwargs["patch_size"] = n_times
            kwargs["patch_stride"] = n_times
        super().__init__(*args, **kwargs)
        if self.patch_kind in {"rnn", "gru"}:
            self.conv_temporal = _GroupPatchRecurrent(
                self.conv_temporal, self.patch_kind
            )
        elif self.patch_kind == "mamba":
            self.conv_temporal = _GroupFullMamba(
                self.conv_temporal, d_state=mamba_d_state
            )
        if self.offset_kind != "lstm":
            self.pool_1.deformation = _OffsetWarp(
                self.pool_1.deformation, self.offset_kind,
                d_state=mamba_d_state,
            )


class group4_patchGRU_deformableGRU(_ControlledAblation):
    patch_kind = "gru"
    offset_kind = "gru"


class group4_patchRNN_deformableRNN(_ControlledAblation):
    patch_kind = "rnn"
    offset_kind = "rnn"


class group4_fullMamba_deformableLstm(_ControlledAblation):
    patch_kind = "mamba"


class group4_patchLstm_deformableMamba(_ControlledAblation):
    offset_kind = "mamba"


def build_group4_patchgru_deformablegru(n_classes, *, n_channels=128,
                                        n_times=1153, **kwargs):
    return group4_patchGRU_deformableGRU(
        n_chans=n_channels, n_outputs=n_classes, n_times=n_times, **kwargs
    )


def build_group4_patchrnn_deformablernn(n_classes, *, n_channels=128,
                                        n_times=1153, **kwargs):
    return group4_patchRNN_deformableRNN(
        n_chans=n_channels, n_outputs=n_classes, n_times=n_times, **kwargs
    )


def build_group4_fullmamba_deformablelstm(n_classes, *, n_channels=128,
                                          n_times=1153, **kwargs):
    return group4_fullMamba_deformableLstm(
        n_chans=n_channels, n_outputs=n_classes, n_times=n_times, **kwargs
    )


def build_group4_patchlstm_deformablemamba(n_classes, *, n_channels=128,
                                           n_times=1153, **kwargs):
    return group4_patchLstm_deformableMamba(
        n_chans=n_channels, n_outputs=n_classes, n_times=n_times, **kwargs
    )
