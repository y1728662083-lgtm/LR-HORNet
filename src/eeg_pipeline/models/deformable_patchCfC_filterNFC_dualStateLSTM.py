"""FilterNFC backbone with dual-state recurrent feature fusion."""

from __future__ import annotations

from torch import nn

from eeg_pipeline.models._recurrent_feature_fusion import (
    DualStateFeatureFusionLSTM,
    replace_backbone_output_width,
)
from eeg_pipeline.models.deformable_patchCfC_filterNFC import (
    deformable_patchCfC_filterNFC,
)


__all__ = [
    "deformable_patchCfC_filterNFC_dualStateLSTM",
    "build_deformable_patchcfc_filternfc_dualstatelstm",
]


class deformable_patchCfC_filterNFC_dualStateLSTM(
    deformable_patchCfC_filterNFC
):
    """Replace depthwise and point convolutions by score/value LSTMs."""

    def __init__(
        self,
        *args,
        dual_state_hidden_size: int = 8,
        fusion_temperature: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        old_depth = self.conv_separable_depth
        old_point = self.conv_separable_point
        n_input_features = old_depth.in_channels
        if (
            old_depth.out_channels != n_input_features
            or old_point.in_channels != n_input_features
            or old_point.out_channels != n_input_features
        ):
            raise RuntimeError(
                "dual-state ablation expects a width-preserving separable "
                "block"
            )
        reference_parameter = next(old_depth.parameters())
        self.conv_separable_depth = nn.Identity()
        self.conv_separable_point = DualStateFeatureFusionLSTM(
            n_input_features,
            hidden_size=dual_state_hidden_size,
            temperature=fusion_temperature,
            device=reference_parameter.device,
            dtype=reference_parameter.dtype,
        )

        final_conv_length = (
            self.temporal_output_length // self.pool2_kernel_size
        )
        replace_backbone_output_width(
            self,
            n_output_features=1,
            final_conv_length=final_conv_length,
            reference_parameter=reference_parameter,
        )

        self.recurrent_fusion = "dual_state_feature_softmax_16_to_1"
        self.dual_state_input_features = n_input_features
        self.dual_state_hidden_size = (
            self.conv_separable_point.hidden_size
        )
        self.fusion_temperature = (
            self.conv_separable_point.temperature
        )


def build_deformable_patchcfc_filternfc_dualstatelstm(
    n_classes: int,
    *,
    n_channels: int = 128,
    n_times: int = 1153,
    **model_kwargs,
) -> deformable_patchCfC_filterNFC_dualStateLSTM:
    """Build the FilterNFC dual-state recurrent-fusion variant."""
    return deformable_patchCfC_filterNFC_dualStateLSTM(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        **model_kwargs,
    )
