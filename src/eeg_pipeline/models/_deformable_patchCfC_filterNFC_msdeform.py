"""Shared insertion logic for FilterNFC MSDeformAttn variants."""

from __future__ import annotations

from collections import OrderedDict

from eeg_pipeline.models._ms_deform_sparse_attention import (
    AttentionMode,
    FeatureWiseMSDeformSparseAttention,
)
from eeg_pipeline.models.deformable_patchCfC_filterNFC import (
    deformable_patchCfC_filterNFC,
)


class _DeformablePatchCfCFilterNFCMSDeform(
    deformable_patchCfC_filterNFC
):
    """Insert feature-independent sparse attention before spatial filtering."""

    def __init__(
        self,
        *args,
        attention_mode: AttentionMode,
        attention_dim: int = 8,
        attention_heads: int = 4,
        attention_points: int = 4,
        attention_time_chunk_size: int = 4,
        attention_checkpoint: bool = True,
        attention_backend: str = "cuda",
        attention_im2col_step: int = 64,
        attention_position_encoding: str = "orthogonal",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        reference = self.conv_spatial.weight
        sparse_attention = FeatureWiseMSDeformSparseAttention(
            n_features=self.F1,
            n_chans=self.n_chans,
            mode=attention_mode,
            d_model=attention_dim,
            n_heads=attention_heads,
            n_points=attention_points,
            time_chunk_size=attention_time_chunk_size,
            checkpoint_attention=attention_checkpoint,
            backend=attention_backend,
            im2col_step=attention_im2col_step,
            position_encoding=attention_position_encoding,
            device=reference.device,
            dtype=reference.dtype,
        )

        # FilterNFC freezes this tuple before it registers its training-only
        # modules. Rebuild both the Sequential order and that frozen execution
        # list, otherwise forward() would silently skip the new attention.
        ordered_modules = OrderedDict()
        for name, module in self._modules.items():
            if name == "conv_spatial":
                ordered_modules["sparse_attention"] = sparse_attention
            ordered_modules[name] = module
        self._modules = ordered_modules

        backbone_names = list(self._backbone_module_names)
        try:
            spatial_index = backbone_names.index("conv_spatial")
        except ValueError as error:
            raise RuntimeError(
                "conv_spatial was not found in the FilterNFC backbone"
            ) from error
        backbone_names.insert(spatial_index, "sparse_attention")
        self._backbone_module_names = tuple(backbone_names)

        self.attention_mode = attention_mode
        self.attention_dim = sparse_attention.d_model
        self.attention_heads = sparse_attention.n_heads
        self.attention_points = sparse_attention.n_points
        self.attention_time_chunk_size = sparse_attention.time_chunk_size
        self.attention_checkpoint = sparse_attention.checkpoint_attention
        self.attention_backend = sparse_attention.backend
        self.attention_position_encoding = sparse_attention.position_encoding
