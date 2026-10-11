"""FilterNFC backbone with a post-depthwise 16-to-1 point LSTM."""

from __future__ import annotations

from eeg_pipeline.models._recurrent_feature_fusion import (
    PointLSTMFeatureCollapse,
    replace_backbone_output_width,
)
from eeg_pipeline.models.deformable_patchCfC_filterNFC import (
    deformable_patchCfC_filterNFC,
)


__all__ = [
    "deformable_patchCfC_filterNFC_pointLSTM",
    "build_deformable_patchcfc_filternfc_pointlstm",
]


class deformable_patchCfC_filterNFC_pointLSTM(
    deformable_patchCfC_filterNFC
):
    """Replace only the 16-to-16 point convolution by a 16-to-1 LSTM."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        old_point = self.conv_separable_point
        n_input_features = old_point.in_channels
        if old_point.out_channels != n_input_features:
            raise RuntimeError(
                "point-LSTM ablation expects a square point convolution, "
                f"got {old_point.in_channels}->{old_point.out_channels}"
            )
        reference_parameter = next(old_point.parameters())
        self.conv_separable_point = PointLSTMFeatureCollapse(
            n_input_features,
            device=reference_parameter.device,
            dtype=reference_parameter.dtype,
        )
        replace_backbone_output_width(
            self,
            n_output_features=1,
            final_conv_length=self.final_conv_length,
            reference_parameter=reference_parameter,
        )

        self.recurrent_fusion = "post_depthwise_point_lstm_16_to_1"
        self.point_lstm_input_features = n_input_features
        self.point_lstm_output_features = 1


def build_deformable_patchcfc_filternfc_pointlstm(
    n_classes: int,
    *,
    n_channels: int = 128,
    n_times: int = 1153,
    **model_kwargs,
) -> deformable_patchCfC_filterNFC_pointLSTM:
    """Build the FilterNFC post-depthwise point-LSTM variant."""
    return deformable_patchCfC_filterNFC_pointLSTM(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        **model_kwargs,
    )
