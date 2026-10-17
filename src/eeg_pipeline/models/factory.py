"""Create EEG models from configuration names."""

from __future__ import annotations

from torch import nn

from eeg_pipeline.models.eegnet import build_eegnet
from eeg_pipeline.models.causal1_EEGNet import build_causal1_eegnet
from eeg_pipeline.models.causal2_EEGNet import build_causal2_eegnet
from eeg_pipeline.models.causal1_spaceGroup_EEGNet import build_causal1_spacegroup_eegnet
from eeg_pipeline.models.causal1_timeMean_EEGNet import build_causal1_timemean_eegnet
from eeg_pipeline.models.causal1_timePatchMean_EEGNet import build_causal1_timepatchmean_eegnet
from eeg_pipeline.models.causal1_groupnorm_EEGNet import (
    build_causal1_groupnorm_eegnet,
)
from eeg_pipeline.models.causal1_group8PatchLSTM_EEGNet import (
    build_causal1_group8patchlstm_eegnet,
)
from eeg_pipeline.models.causal1_group16PatchLSTM_filterNCE_EEGNet import (
    build_causal1_group16patchlstm_filternce_eegnet,
)
from eeg_pipeline.models.causal1_groupSAttPatchLSTM_EEGNet import (
    build_causal1_groupsattpatchlstm_eegnet,
)
from eeg_pipeline.models.causal1_groupPatchLSTM_softmaxFusion_EEGNet import (
    build_causal1_grouppatchlstm_softmaxfusion_eegnet,
)
from eeg_pipeline.models.egroupPatchCfC_EEGNet import (
    build_egrouppatchcfc_eegnet,
)
from eeg_pipeline.models.deformable_patchCfC import (
    build_deformable_patchcfc,
)
from eeg_pipeline.models.group4_patchLstm_deformableLstm import (
    build_group4_patchlstm_deformablelstm,
)
from eeg_pipeline.models.group4_causal1_deformableLstm import (
    build_group4_causal1_deformablelstm,
)
from eeg_pipeline.models.group4_patchLstm_deformableLstm_satt import (
    build_group4_patchlstm_deformablelstm_satt,
)
from eeg_pipeline.models.group4_patchLstm_deformableLstm_statt import (
    build_group4_patchlstm_deformablelstm_statt,
)
from eeg_pipeline.models.group4_serialPatchLstmMamba_deformableLstm import (
    build_group4_serialpatchlstmmamba_deformablelstm,
)
from eeg_pipeline.models.group4_patchLstm_deformableCfC import (
    build_group4_patchlstm_deformablecfc,
)
from eeg_pipeline.models.group4_patchCfC_deformableCfC import (
    build_group4_patchcfc_deformablecfc,
)
from eeg_pipeline.models.patchLstm_deformableLstm_group_ablation import (
    build_group1_patchlstm_deformablelstm,
    build_group2_patchlstm_deformablelstm,
    build_group8_patchlstm_deformablelstm,
    build_group16_patchlstm_deformablelstm,
)
from eeg_pipeline.models.group4_patchlstm_core_ablation import (
    build_group4_patchlstm_nodeformation,
    build_group4_patchlstm_pointwisedeformation,
    build_group4_patchlstm_staticdeformation,
    build_group4_patchlstm_groupedconvdeformation,
)
from eeg_pipeline.models.group4_deformable_ablation import (
    build_group4_patchgru_deformablegru,
    build_group4_patchrnn_deformablernn,
    build_group4_fullmamba_deformablelstm,
    build_group4_patchlstm_deformablemamba,
)
from eeg_pipeline.models.DualMem_WaveClock_CfC import (
    build_dualmem_waveclock_cfc,
)
from eeg_pipeline.models.deformable_patchCfC_pointFFN import (
    build_deformable_patchcfc_pointffn,
)
from eeg_pipeline.models.deformable_patchCfC_areaMask import (
    build_deformable_patchcfc_areamask,
)
from eeg_pipeline.models.deformable_patchCfC_filterNFC import (
    build_deformable_patchcfc_filternfc,
)
from eeg_pipeline.models.deformable_patchCfC_filterNFC_softmaxFusion import (
    build_deformable_patchcfc_filternfc_softmaxfusion,
)
from eeg_pipeline.models.deformable_patchCfC_filterNFC_pointLSTM import (
    build_deformable_patchcfc_filternfc_pointlstm,
)
from eeg_pipeline.models.deformable_patchCfC_filterNFC_dualStateLSTM import (
    build_deformable_patchcfc_filternfc_dualstatelstm,
)
from eeg_pipeline.models.deformable_patchCfC_filterNFC_dualStateLSTM_ablation import (
    build_deformable_patchcfc_filternfc_dualstatelstm_ablation,
)
from eeg_pipeline.models.deformable_patchCfC_filterNFC_satt import (
    build_deformable_patchcfc_filternfc_satt,
)
from eeg_pipeline.models.deformable_patchCfC_filterNFC_statt import (
    build_deformable_patchcfc_filternfc_statt,
)
from eeg_pipeline.models.baseline_model import (
    build_braindecode_from_model_key,
    is_braindecode_model_key,
)


MODEL_BUILDERS = {
    "eegnet": build_eegnet,
    "causal1_eegnet": build_causal1_eegnet,
    "causal2_eegnet": build_causal2_eegnet,
    "causal1_spacegroup_eegnet": build_causal1_spacegroup_eegnet,
    "causal1_timemean_eegnet": build_causal1_timemean_eegnet,
    "causal1_timepatchmean_eegnet": build_causal1_timepatchmean_eegnet,
    "causal1_groupnorm_eegnet": build_causal1_groupnorm_eegnet,
    "causal1_group8patchlstm_eegnet": build_causal1_group8patchlstm_eegnet,
    "causal1_group16patchlstm_filternce_eegnet": (
        build_causal1_group16patchlstm_filternce_eegnet
    ),
    "causal1_group16PatchLSTM_filterNCE_EEGNet": (
        build_causal1_group16patchlstm_filternce_eegnet
    ),
    "causal1_groupSAttPatchLSTM_EEGNet": (
        build_causal1_groupsattpatchlstm_eegnet
    ),
    "causal1_groupsattpatchlstm_eegnet": build_causal1_groupsattpatchlstm_eegnet,
    "causal1_groupPatchLSTM_softmaxFusion_EEGNet": (
        build_causal1_grouppatchlstm_softmaxfusion_eegnet
    ),
    "causal1_grouppatchlstm_softmaxfusion_eegnet": (
        build_causal1_grouppatchlstm_softmaxfusion_eegnet
    ),
    "egrouppatchcfc_eegnet": build_egrouppatchcfc_eegnet,
    "deformable_patchCfC": build_deformable_patchcfc,
    "deformable_patchcfc": build_deformable_patchcfc,
    "group4_patchLstm_deformableLstm": (
        build_group4_patchlstm_deformablelstm
    ),
    "group4_patchlstm_deformablelstm": (
        build_group4_patchlstm_deformablelstm
    ),
    "group4_causal1_deformableLstm": (
        build_group4_causal1_deformablelstm
    ),
    "group4_causal1_deformablelstm": (
        build_group4_causal1_deformablelstm
    ),
    "group4_patchLstm_deformableLstm_satt": (
        build_group4_patchlstm_deformablelstm_satt
    ),
    "group4_patchlstm_deformablelstm_satt": (
        build_group4_patchlstm_deformablelstm_satt
    ),
    "group4_patchLstm_deformableLstm_statt": (
        build_group4_patchlstm_deformablelstm_statt
    ),
    "group4_patchlstm_deformablelstm_statt": (
        build_group4_patchlstm_deformablelstm_statt
    ),
    "group4_serialPatchLstmMamba_deformableLstm": (
        build_group4_serialpatchlstmmamba_deformablelstm
    ),
    "group4_serialpatchlstmmamba_deformablelstm": (
        build_group4_serialpatchlstmmamba_deformablelstm
    ),
    "group4_patchLstm_deformableCfC": (
        build_group4_patchlstm_deformablecfc
    ),
    "group4_patchlstm_deformablecfc": (
        build_group4_patchlstm_deformablecfc
    ),
    "group4_patchCfC_deformableCfC": (
        build_group4_patchcfc_deformablecfc
    ),
    "group4_patchcfc_deformablecfc": (
        build_group4_patchcfc_deformablecfc
    ),
    "group1_patchLstm_deformableLstm": (
        build_group1_patchlstm_deformablelstm
    ),
    "group1_patchlstm_deformablelstm": (
        build_group1_patchlstm_deformablelstm
    ),
    "group2_patchLstm_deformableLstm": (
        build_group2_patchlstm_deformablelstm
    ),
    "group2_patchlstm_deformablelstm": (
        build_group2_patchlstm_deformablelstm
    ),
    "group8_patchLstm_deformableLstm": (
        build_group8_patchlstm_deformablelstm
    ),
    "group8_patchlstm_deformablelstm": (
        build_group8_patchlstm_deformablelstm
    ),
    "group16_patchLstm_deformableLstm": (
        build_group16_patchlstm_deformablelstm
    ),
    "group16_patchlstm_deformablelstm": (
        build_group16_patchlstm_deformablelstm
    ),
    "group4_patchLstm_noDeformation": (
        build_group4_patchlstm_nodeformation
    ),
    "group4_patchlstm_nodeformation": (
        build_group4_patchlstm_nodeformation
    ),
    "group4_patchLstm_pointwiseDeformation": (
        build_group4_patchlstm_pointwisedeformation
    ),
    "group4_patchlstm_pointwisedeformation": (
        build_group4_patchlstm_pointwisedeformation
    ),
    "group4_patchLstm_staticDeformation": (
        build_group4_patchlstm_staticdeformation
    ),
    "group4_patchlstm_staticdeformation": (
        build_group4_patchlstm_staticdeformation
    ),
    "group4_patchLstm_groupedConvDeformation": (
        build_group4_patchlstm_groupedconvdeformation
    ),
    "group4_patchlstm_groupedconvdeformation": (
        build_group4_patchlstm_groupedconvdeformation
    ),
    "group4_patchGRU_deformableGRU": build_group4_patchgru_deformablegru,
    "group4_patchRNN_deformableRNN": build_group4_patchrnn_deformablernn,
    "group4_fullMamba_deformableLstm": build_group4_fullmamba_deformablelstm,
    "group4_patchLstm_deformableMamba": build_group4_patchlstm_deformablemamba,
    "DualMem_WaveClock_CfC": build_dualmem_waveclock_cfc,
    "dualmem_waveclock_cfc": build_dualmem_waveclock_cfc,
    "deformable_patchCfC_pointFFN": build_deformable_patchcfc_pointffn,
    "deformable_patchcfc_pointffn": build_deformable_patchcfc_pointffn,
    "deformable_patchCfC_areaMask": build_deformable_patchcfc_areamask,
    "deformable_patchcfc_areamask": build_deformable_patchcfc_areamask,
    "deformable_patchCfC_filterNFC": build_deformable_patchcfc_filternfc,
    "deformable_patchcfc_filternfc": build_deformable_patchcfc_filternfc,
    "deformable_patchCfC_filterNFC_softmaxFusion": (
        build_deformable_patchcfc_filternfc_softmaxfusion
    ),
    "deformable_patchcfc_filternfc_softmaxfusion": (
        build_deformable_patchcfc_filternfc_softmaxfusion
    ),
    "deformable_patchCfC_filterNFC_pointLSTM": (
        build_deformable_patchcfc_filternfc_pointlstm
    ),
    "deformable_patchcfc_filternfc_pointlstm": (
        build_deformable_patchcfc_filternfc_pointlstm
    ),
    "deformable_patchCfC_filterNFC_dualStateLSTM": (
        build_deformable_patchcfc_filternfc_dualstatelstm
    ),
    "deformable_patchcfc_filternfc_dualstatelstm": (
        build_deformable_patchcfc_filternfc_dualstatelstm
    ),
    "deformable_patchCfC_filterNFC_dualStateLSTM_ablation": (
        build_deformable_patchcfc_filternfc_dualstatelstm_ablation
    ),
    "deformable_patchcfc_filternfc_dualstatelstm_ablation": (
        build_deformable_patchcfc_filternfc_dualstatelstm_ablation
    ),
    "deformable_patchCfC_filterNFC_satt": (
        build_deformable_patchcfc_filternfc_satt
    ),
    "deformable_patchcfc_filternfc_satt": (
        build_deformable_patchcfc_filternfc_satt
    ),
    "deformable_patchCfC_filterNFC_statt": (
        build_deformable_patchcfc_filternfc_statt
    ),
    "deformable_patchcfc_filternfc_statt": (
        build_deformable_patchcfc_filternfc_statt
    ),
}


def build_model(
    name: str,
    n_classes: int,
    **model_kwargs,
) -> nn.Module:
    """Build a registered model with the requested output classes."""
    if is_braindecode_model_key(name):
        return build_braindecode_from_model_key(
            name=name,
            n_classes=n_classes,
            **model_kwargs,
        )
    try:
        builder = MODEL_BUILDERS[name]
    except KeyError:
        available_models = ", ".join(sorted(MODEL_BUILDERS))
        raise ValueError(
            f"Unknown model {name!r}; available models: {available_models}"
        ) from None
    return builder(n_classes=n_classes, **model_kwargs)


if __name__ == "__main__":
    import torch

    for n_classes in (3, 4):
        model = build_model("eegnet", n_classes=n_classes)
        model.eval()
        with torch.inference_mode():
            logits = model(torch.zeros(2, 128, 1153))
        if logits.shape != (2, n_classes):
            raise RuntimeError(f"Unexpected output shape: {logits.shape}")

    try:
        build_model("unknown", n_classes=4)
    except ValueError:
        pass
    else:
        raise RuntimeError("Unknown model name was not rejected")

    print("model factory test passed")
