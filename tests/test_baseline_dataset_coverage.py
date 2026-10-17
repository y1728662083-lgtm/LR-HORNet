"""Ensure every comparison model has a configuration for all five datasets."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "ds003626": ("ds003626", 10),
    "bimodal_inner_speech": ("bimodal_inner_speech", 4),
    "directional_words_russian": ("directional_words", 12),
    "directional_words_spanish": ("directional_words", 10),
    "bci2020": ("bci2020", 15),
}
CONFIG_PATHS = {
    "eegnet": {
        dataset: f"baselines/eegnet/configs/{dataset}.yaml"
        for dataset in DATASETS
    },
    "atcnet": {
        dataset: f"baselines/atcnet/configs/{dataset}.yaml"
        for dataset in DATASETS
    },
    "eegmamba": {
        dataset: (
            "baselines/eegmamba/addition/configs/"
            f"{dataset}_official_baselines.yaml"
        )
        for dataset in DATASETS
    },
    "labram": {
        dataset: (
            "baselines/labram/addition/configs/"
            f"{dataset}_official_baselines.yaml"
        )
        for dataset in DATASETS
    },
    "fast": {
        "ds003626": "baselines/fast/addition/configs/fast_native_ds003626.yaml",
        "bimodal_inner_speech": (
            "baselines/fast/addition/configs/fast_native_bimodal.yaml"
        ),
        "directional_words_russian": (
            "baselines/fast/addition/configs/fast_native_russian.yaml"
        ),
        "directional_words_spanish": (
            "baselines/fast/addition/configs/fast_native_spanish.yaml"
        ),
        "bci2020": "baselines/fast/addition/configs/fast_native_bci2020.yaml",
    },
}


CASES = tuple(
    (model, dataset, relative)
    for model, paths in CONFIG_PATHS.items()
    for dataset, relative in paths.items()
)


@pytest.mark.parametrize(
    ("model", "dataset", "relative"),
    CASES,
    ids=lambda value: str(value),
)
def test_baseline_dataset_configuration(
    model: str,
    dataset: str,
    relative: str,
) -> None:
    path = ROOT / relative
    assert path.is_file(), path
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    canonical_dataset, folds = DATASETS[dataset]
    assert config["data"].get("dataset", "ds003626") == canonical_dataset
    assert len(config["data"]["subjects"]) == folds
    assert config["split"]["fold_indices"] == list(range(folds))
    assert config["split"].get("adapt_held_out_subjects", True) is True

    if model in {"eegnet", "atcnet"}:
        assert config["model"]["name"] == f"braindecode_{model}"
    else:
        assert config["models"][model]["enabled"] is True
