"""Deformable Patch-CfC with a grouped pointwise FFN after CfC1."""

from __future__ import annotations

from torch import Tensor, nn

from eeg_pipeline.models.causal1_group8PatchLSTM_EEGNet import (
    _positive_integer,
)
from eeg_pipeline.models.deformable_patchCfC import (
    _dropout_probability,
    deformable_patchCfC,
)


__all__ = [
    "deformable_patchCfC_pointFFN",
    "build_deformable_patchcfc_pointffn",
]


class _CfCPointFFNTemporal(nn.Module):
    """Apply dropout and a feature-grouped pointwise FFN after Patch-CfC1."""

    def __init__(
        self,
        cfc_temporal: nn.Module,
        *,
        n_features: int,
        hidden_features: int,
        conv_groups: int,
        norm_groups: int,
        dropout: float,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        n_features = _positive_integer("F1", n_features)
        hidden_features = _positive_integer(
            "point_ffn_hidden_features", hidden_features
        )
        conv_groups = _positive_integer("point_ffn_groups", conv_groups)
        norm_groups = _positive_integer(
            "point_ffn_norm_groups", norm_groups
        )
        dropout = _dropout_probability("cfc_output_dropout", dropout)

        if n_features % conv_groups != 0:
            raise ValueError(
                "point_ffn_groups must divide the CfC feature count; "
                f"got F1={n_features} and groups={conv_groups}."
            )
        if hidden_features % conv_groups != 0:
            raise ValueError(
                "point_ffn_groups must divide point_ffn_hidden_features; "
                f"got hidden_features={hidden_features} and "
                f"groups={conv_groups}."
            )
        if hidden_features % norm_groups != 0:
            raise ValueError(
                "point_ffn_norm_groups must divide point_ffn_hidden_features; "
                f"got hidden_features={hidden_features} and "
                f"norm_groups={norm_groups}."
            )

        self.cfc_temporal = cfc_temporal
        self.dropout = nn.Dropout(p=dropout)
        self.expand = nn.Conv2d(
            n_features,
            hidden_features,
            kernel_size=1,
            groups=conv_groups,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.norm = nn.GroupNorm(
            num_groups=norm_groups,
            num_channels=hidden_features,
            device=device,
            dtype=dtype,
        )
        self.activation = nn.ELU()
        self.contract = nn.Conv2d(
            hidden_features,
            n_features,
            kernel_size=1,
            groups=conv_groups,
            bias=True,
            device=device,
            dtype=dtype,
        )

        nn.init.xavier_uniform_(self.expand.weight)
        nn.init.xavier_uniform_(self.contract.weight)
        nn.init.zeros_(self.contract.bias)

        self.n_features = n_features
        self.hidden_features = hidden_features
        self.conv_groups = conv_groups
        self.norm_groups = norm_groups
        self.dropout_probability = dropout

    @property
    def group_cfcs(self) -> nn.ModuleList:
        """Expose the unchanged group CfCs for inspection and compatibility."""
        return self.cfc_temporal.group_cfcs

    def forward(self, input: Tensor) -> Tensor:
        features = self.cfc_temporal(input)
        features = self.dropout(features)
        features = self.expand(features)
        features = self.norm(features)
        features = self.activation(features)
        return self.contract(features)


class deformable_patchCfC_pointFFN(deformable_patchCfC):
    """Add a grouped 8-to-16-to-8 pointwise FFN after Patch-CfC1."""

    def __init__(
        self,
        *args,
        cfc_output_dropout: float = 0.1,
        point_ffn_hidden_features: int = 16,
        point_ffn_groups: int = 8,
        point_ffn_norm_groups: int = 8,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        cfc_temporal = self.conv_temporal
        reference_parameter = next(cfc_temporal.parameters())
        self.conv_temporal = _CfCPointFFNTemporal(
            cfc_temporal,
            n_features=self.F1,
            hidden_features=point_ffn_hidden_features,
            conv_groups=point_ffn_groups,
            norm_groups=point_ffn_norm_groups,
            dropout=cfc_output_dropout,
            device=reference_parameter.device,
            dtype=reference_parameter.dtype,
        )

        self.cfc_output_dropout = self.conv_temporal.dropout_probability
        self.point_ffn_hidden_features = self.conv_temporal.hidden_features
        self.point_ffn_groups = self.conv_temporal.conv_groups
        self.point_ffn_norm_groups = self.conv_temporal.norm_groups


def build_deformable_patchcfc_pointffn(
    n_classes: int,
    *,
    n_channels: int = 128,
    n_times: int = 1153,
    **model_kwargs,
) -> deformable_patchCfC_pointFFN:
    """Build the grouped-point-FFN deformable Patch-CfC model."""
    return deformable_patchCfC_pointFFN(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        **model_kwargs,
    )
