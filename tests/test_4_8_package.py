"""Regression checks for the integrated Section 4.8 package."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eeg_pipeline.models.factory import build_model  # noqa: E402
from eeg_pipeline.models._ms_deform_sparse_attention import (  # noqa: E402
    FeatureWiseMSDeformSparseAttention,
)


def _yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def test_4_8_1_templates_and_results() -> None:
    expected = {
        "00_patchlstm_deformablelstm.yaml": "group4_patchLstm_deformableLstm",
        "01_patchgru_deformablegru.yaml": "group4_patchGRU_deformableGRU",
        "02_patchrnn_deformablernn.yaml": "group4_patchRNN_deformableRNN",
        "03_patchlstm_deformablecfc.yaml": "group4_patchLstm_deformableCfC",
        "04_patchcfc_deformablecfc.yaml": "group4_patchCfC_deformableCfC",
    }
    config_root = ROOT / "experiments" / "4.8.1" / "configs"
    for filename, model_name in expected.items():
        config = _yaml(config_root / filename)
        assert config["seed"] == 42
        assert config["split"]["fold_indices"] == [0]
        assert config["model"]["name"] == model_name

    summaries = _csv(ROOT / "results" / "4.8.1" / "stored_results_summary.csv")
    assert len(summaries) == 5
    assert {int(row["n"]) for row in summaries} == {5}
    assert len(_csv(ROOT / "results" / "4.8.1" / "checkpoint_manifest.csv")) == 25


def test_4_8_2_protocol_and_result_counts() -> None:
    config_root = ROOT / "experiments" / "4.8.2" / "configs"
    for filename, model_name in {
        "group4_patchLstm_deformableLstm_satt.yaml": (
            "group4_patchLstm_deformableLstm_satt"
        ),
        "group4_patchLstm_deformableLstm_statt.yaml": (
            "group4_patchLstm_deformableLstm_statt"
        ),
    }.items():
        config = _yaml(config_root / "attention" / filename)
        assert config["split"]["fold_indices"] == list(range(10))
        assert config["model"]["name"] == model_name
        assert config["model"]["kwargs"]["attention_backend"] == "cuda"

    assert _yaml(config_root / "mamba" / "full_mamba_patch_replaced.yaml")[
        "model"
    ]["name"] == "group4_fullMamba_deformableLstm"
    assert _yaml(config_root / "mamba" / "mamba_offset_only.yaml")["model"][
        "name"
    ] == "group4_patchLstm_deformableMamba"

    result_root = ROOT / "results" / "4.8.2"
    assert len(_csv(result_root / "stored_results_by_run.csv")) == 48
    primary = _csv(result_root / "stored_primary_10fold_summary.csv")
    repeats = _csv(result_root / "stored_mamba_fold1_repeat5_summary.csv")
    assert len(primary) == 4 and {int(row["n"]) for row in primary} == {10}
    assert len(repeats) == 2 and {int(row["n"]) for row in repeats} == {5}
    assert len(_csv(result_root / "checkpoint_manifest.csv")) == 48

    for filename, model_name in {
        "filterNFC_satt.yaml": "deformable_patchCfC_filterNFC_satt",
        "filterNFC_statt.yaml": "deformable_patchCfC_filterNFC_statt",
    }.items():
        legacy = _yaml(config_root / "legacy" / filename)
        assert legacy["model"]["name"] == model_name
        assert legacy["data"]["root"] == "data/ds003626"


def test_vector_token_source_and_vendored_cuda_operator() -> None:
    model_root = ROOT / "src" / "eeg_pipeline" / "models"
    wrapper = (
        model_root / "_group4_patchlstm_deformablelstm_msdeform.py"
    ).read_text(encoding="utf-8")
    operator = (model_root / "_ms_deform_sparse_attention.py").read_text(
        encoding="utf-8"
    )
    assert "VectorTokenMSDeformSparseAttention" in wrapper
    assert "class VectorTokenMSDeformSparseAttention" in operator
    assert "self.route = _MSDeformVectorRoute" in operator

    vendor_root = ROOT / "src" / "Deformable-DETR-main"
    assert len([path for path in vendor_root.rglob("*") if path.is_file()]) == 15
    assert (vendor_root / "LICENSE").is_file()
    assert (vendor_root / "models" / "ops" / "setup.py").is_file()
    assert (
        vendor_root
        / "models"
        / "ops"
        / "src"
        / "cuda"
        / "ms_deform_attn_cuda.cu"
    ).is_file()


def test_attention_state_dict_uses_formal_vector_token_contract() -> None:
    expected_parameters = {
        "group4_patchLstm_deformableLstm_satt": 7374,
        "group4_patchLstm_deformableLstm_statt": 7518,
    }
    config_root = ROOT / "experiments" / "4.8.2" / "configs" / "attention"
    for filename, model_name in (
        (
            "group4_patchLstm_deformableLstm_satt.yaml",
            "group4_patchLstm_deformableLstm_satt",
        ),
        (
            "group4_patchLstm_deformableLstm_statt.yaml",
            "group4_patchLstm_deformableLstm_statt",
        ),
    ):
        config = _yaml(config_root / filename)
        kwargs = dict(config["model"]["kwargs"])
        kwargs["attention_backend"] = "pytorch"
        model = build_model(
            model_name,
            n_classes=4,
            n_channels=128,
            n_times=1153,
            **kwargs,
        )
        keys = tuple(model.state_dict())
        assert any(key.startswith("sparse_attention.route.") for key in keys)
        assert not any(key.startswith("sparse_attention.routes.") for key in keys)
        assert sum(parameter.numel() for parameter in model.parameters()) == (
            expected_parameters[model_name]
        )

        # The reference backend exercises the same graph without requiring a
        # compiled CUDA extension. A reduced geometry keeps the unit test fast.
        smoke_kwargs = dict(kwargs)
        smoke_kwargs["patch_size"] = 8
        smoke_kwargs["patch_stride"] = 4
        smoke_kwargs["attention_time_chunk_size"] = 4
        smoke_model = build_model(
            model_name,
            n_classes=4,
            n_channels=4,
            n_times=16,
            **smoke_kwargs,
        ).eval()
        with torch.inference_mode():
            logits = smoke_model(torch.zeros(1, 4, 16))
        assert tuple(logits.shape) == (1, 4)
        assert bool(torch.isfinite(logits).all())


def test_legacy_featurewise_checkpoint_contract_is_preserved() -> None:
    """The vector-token addition must not rename legacy FilterNFC weights."""
    module = FeatureWiseMSDeformSparseAttention(
        n_features=1,
        n_chans=2,
        mode="space_1d",
        d_model=4,
        n_heads=1,
        n_points=1,
        time_chunk_size=4,
        checkpoint_attention=False,
        backend="pytorch",
    ).eval()
    keys = tuple(module.state_dict())
    assert any(key.startswith("routes.0.scalar_readout.") for key in keys)
    assert not any("feature_readout" in key for key in keys)

    with torch.inference_mode():
        output = module(torch.zeros(1, 1, 2, 4))
    assert tuple(output.shape) == (1, 1, 2, 4)
    assert bool(torch.isfinite(output).all())
