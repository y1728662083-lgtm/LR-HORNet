"""Vector-token sparse-attention insertion for the pure Group4 variants.

This module deliberately inherits the Group4 PatchLSTM/DeformableLSTM model,
not any FilterNFC model.  The only new operation is sparse attention over the
complete feature vector at every electrode-time location, inserted between
temporal normalization and EEGNet's spatial convolution.
"""

from __future__ import annotations

from collections import OrderedDict

from eeg_pipeline.models._ms_deform_sparse_attention import (
    AttentionMode,
    VectorTokenMSDeformSparseAttention,
)
from eeg_pipeline.models.group4_patchLstm_deformableLstm import (
    group4_patchLstm_deformableLstm,
)


class _Group4PatchLSTMDeformableLSTMMSDeform(
    group4_patchLstm_deformableLstm
):
    """Insert SAtt or StAtt before the unchanged spatial convolution."""

    def __init__(
        self,
        *args,
        attention_mode: AttentionMode,
        attention_dim: int = 8,
        attention_heads: int = 4,
        attention_points: int = 4,
        attention_time_chunk_size: int = 64,
        attention_checkpoint: bool = False,
        attention_backend: str = "cuda",
        attention_im2col_step: int = 4,
        attention_position_encoding: str = "orthogonal",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        if "sparse_attention" in self._modules:
            raise RuntimeError("sparse_attention is already registered")

        reference = self.conv_spatial.weight
        sparse_attention = VectorTokenMSDeformSparseAttention(
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

        # The parent is an nn.Sequential model.  Rebuilding its registered
        # order makes the insertion part of the actual forward path:
        # PatchLSTM -> temporal norm -> attention -> spatial convolution.
        ordered_modules = OrderedDict()
        inserted = False
        for name, module in self._modules.items():
            if name == "conv_spatial":
                ordered_modules["sparse_attention"] = sparse_attention
                inserted = True
            ordered_modules[name] = module
        if not inserted:
            raise RuntimeError("conv_spatial was not found in the Group4 model")
        self._modules = ordered_modules

        self.attention_mode = attention_mode
        self.attention_dim = sparse_attention.d_model
        self.attention_heads = sparse_attention.n_heads
        self.attention_points = sparse_attention.n_points
        self.attention_time_chunk_size = sparse_attention.time_chunk_size
        self.attention_checkpoint = sparse_attention.checkpoint_attention
        self.attention_backend = sparse_attention.backend
        self.attention_im2col_step = sparse_attention.im2col_step
        self.attention_position_encoding = sparse_attention.position_encoding
        self.attention_placement = "after_patchlstm_before_spatial_conv"
        self.attention_token_definition = sparse_attention.token_definition
        self.attention_route_count = sparse_attention.route_count

        # A hard guard against accidentally changing this controlled ablation
        # into the older FilterNFC experiment family.
        if hasattr(self, "forward_contrastive"):
            raise RuntimeError("FilterNFC/TFC must not be active in this model")
