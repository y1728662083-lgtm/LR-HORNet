#!/usr/bin/env python3
"""Validate repository structure and portability without loading EEG data."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".pyc", ".log", ".vsdx"}
WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")


def validate_configs() -> None:
    paths = sorted([*ROOT.rglob("*.yaml"), *ROOT.rglob("*.yml")])
    if not paths:
        raise RuntimeError("No YAML configurations found")
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        if not isinstance(config, dict):
            raise ValueError(f"Configuration is not a mapping: {path}")
        data = config.get("data")
        if isinstance(data, dict) and isinstance(data.get("root"), str):
            value = data["root"]
            if value.startswith("/") or WINDOWS_ABSOLUTE.match(value):
                raise ValueError(f"Absolute data.root in {path}: {value}")


def validate_files() -> None:
    forbidden = [
        path.relative_to(ROOT)
        for path in ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES
    ]
    if forbidden:
        raise ValueError(f"Forbidden generated artifacts: {forbidden}")


def validate_main_results() -> None:
    path = ROOT / "results" / "4.4.1" / "model_summary_all_datasets.csv"
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    selected = [row for row in rows if row.get("model") == "LR-HORNet"]
    expected = {
        "ds003626": 10,
        "Bimodal Inner Speech": 4,
        "Directional Words–Russian": 12,
        "Directional Words–Spanish": 10,
    }
    observed = {row["dataset"]: int(row["folds_completed"]) for row in selected}
    if observed != expected:
        raise ValueError(f"Unexpected LR-HORNet result coverage: {observed}")


def validate_required_paths() -> None:
    required = [
        ROOT / "main.py",
        ROOT / "src" / "eeg_pipeline" / "models" / "group4_patchLstm_deformableLstm.py",
        ROOT / "configs" / "main" / "ds003626.yaml",
        ROOT / "docs" / "RESULTS_AUDIT.md",
    ]
    missing = [path.relative_to(ROOT) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")


def main() -> None:
    validate_required_paths()
    validate_configs()
    validate_files()
    validate_main_results()
    print("Repository validation passed.")


if __name__ == "__main__":
    main()
