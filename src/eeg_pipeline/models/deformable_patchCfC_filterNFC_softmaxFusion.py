"""FilterNFC deformable Patch-CfC with dynamic Softmax electrode fusion."""

from __future__ import annotations

from eeg_pipeline.models._feature_time_electrode_softmax_fusion import (
    FeatureTimeElectrodeSoftmaxFusion,
)
from eeg_pipeline.models.deformable_patchCfC_filterNFC import (
    deformable_patchCfC_filterNFC,
)


__all__ = [
    "deformable_patchCfC_filterNFC_softmaxFusion",
    "build_deformable_patchcfc_filternfc_softmaxfusion",
]


class deformable_patchCfC_filterNFC_softmaxFusion(
    deformable_patchCfC_filterNFC
):
    """Replace only the classification backbone's spatial convolution."""

    def __init__(
        self,
        *args,
        fusion_hidden_features_per_feature: int = 8,
        fusion_time_chunk_size: int | None = 32,
        fusion_activation_checkpoint: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        old_spatial = self.conv_spatial
        if old_spatial.in_channels != self.F1:
            raise RuntimeError(
                "unexpected spatial-convolution input width: "
                f"{old_spatial.in_channels} != F1={self.F1}."
            )
        expected_outputs = self.F1 * self.D
        if old_spatial.out_channels != expected_outputs:
            raise RuntimeError(
                "unexpected spatial-convolution output width: "
                f"{old_spatial.out_channels} != F1*D={expected_outputs}."
            )
        reference_parameter = next(old_spatial.parameters())
        self.conv_spatial = FeatureTimeElectrodeSoftmaxFusion(
            n_features=self.F1,
            n_heads=self.D,
            hidden_features_per_feature=(
                fusion_hidden_features_per_feature
            ),
            time_chunk_size=fusion_time_chunk_size,
            activation_checkpoint=fusion_activation_checkpoint,
            device=reference_parameter.device,
            dtype=reference_parameter.dtype,
        )
        if "conv_spatial" not in self._backbone_module_names:
            raise RuntimeError(
                "filterNFC backbone does not contain conv_spatial."
            )

        self.spatial_fusion = "feature_time_electrode_softmax"
        self.fusion_hidden_features_per_feature = (
            self.conv_spatial.hidden_features_per_feature
        )
        self.fusion_time_chunk_size = self.conv_spatial.time_chunk_size
        self.fusion_activation_checkpoint = (
            self.conv_spatial.activation_checkpoint
        )


def build_deformable_patchcfc_filternfc_softmaxfusion(
    n_classes: int,
    *,
    n_channels: int = 128,
    n_times: int = 1153,
    **model_kwargs,
) -> deformable_patchCfC_filterNFC_softmaxFusion:
    """Build the FilterNFC deformable Patch-CfC fusion variant."""
    return deformable_patchCfC_filterNFC_softmaxFusion(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        **model_kwargs,
    )
