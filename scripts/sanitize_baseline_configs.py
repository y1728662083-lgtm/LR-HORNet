#!/usr/bin/env python3
"""Replace archived machine-specific roots with portable repository paths."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINES = ROOT / "baselines"
CONFIG_GLOBS = (
    "eegnet/configs/*.yaml",
    "eegnet/historical_configs/*.yaml",
    "atcnet/configs/*.yaml",
    "atcnet/historical_configs/*.yaml",
    "eegmamba/addition/configs/*.yaml",
    "labram/addition/configs/*.yaml",
    "fast/addition/configs/*.yaml",
)
FIELD = re.compile(r"^(?P<indent>\s*)(?P<key>root|output_root):\s*(?P<value>.+?)\s*$")


def dataset_key(path: Path) -> str:
    name = path.stem.lower()
    if "bimodal" in name:
        return "bimodal_inner_speech"
    if "russian" in name:
        return "directional_words_russian"
    if "spanish" in name:
        return "directional_words_spanish"
    if "ds003626" in name:
        return "ds003626"
    raise ValueError(f"Cannot identify dataset from {path}")


def data_root(dataset: str) -> str:
    if dataset.startswith("directional_words_"):
        return "data/directional_words"
    return f"data/{dataset}"


def is_absolute_yaml_value(value: str) -> bool:
    normalized = value.strip("'\"")
    return normalized.startswith("/") or bool(
        re.match(r"^[A-Za-z]:[\\/]", normalized)
    )


def sanitize(path: Path) -> bool:
    dataset = dataset_key(path)
    baseline = path.relative_to(BASELINES).parts[0]
    original = path.read_text(encoding="utf-8")
    lines: list[str] = []
    changed = False
    for line in original.splitlines():
        match = FIELD.match(line)
        if match is None or not is_absolute_yaml_value(match.group("value")):
            lines.append(line)
            continue
        key = match.group("key")
        if key == "root":
            value = data_root(dataset)
        else:
            value = f"outputs/baselines/{baseline}/{dataset}"
        lines.append(f"{match.group('indent')}{key}: {value}")
        changed = True
    if changed:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return changed


def replace_exact(path: Path, replacements: dict[str, str]) -> bool:
    original = path.read_text(encoding="utf-8")
    updated = original
    for old, new in replacements.items():
        updated = updated.replace(old, new)
    if updated == original:
        return False
    path.write_text(updated, encoding="utf-8", newline="\n")
    return True


def sanitize_parameter_sources() -> bool:
    path = ROOT / "results" / "4.4.1" / "parameter_counts_fold_level.csv"
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fieldnames = reader.fieldnames
    if not fieldnames:
        raise ValueError(f"Missing CSV header: {path}")
    changed = False
    marker = "dataset_result\\"
    for row in rows:
        source = row["source_file"]
        if marker not in source:
            continue
        suffix = source.split(marker, 1)[1].replace("\\", "/")
        row["source_file"] = f"dataset_result/{suffix}"
        changed = True
    if changed:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return changed


def sanitize_visualization_paths() -> int:
    paths = (
        ROOT
        / "experiments"
        / "4.7.2"
        / "reference_results"
        / "group_04_visualization"
        / "offset_trend_diagnostics.csv",
        ROOT
        / "experiments"
        / "4.7.2"
        / "summary"
        / "group04_offset_trend_diagnostics.csv",
        ROOT
        / "results"
        / "4.7.2"
        / "summary"
        / "group04_offset_trend_diagnostics.csv",
    )
    changed_files = 0
    marker = "/deformable_study/output/group_02/"
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"Missing CSV header: {path}")
        changed = False
        for row in rows:
            source = row["checkpoint"]
            if marker not in source:
                continue
            suffix = source.split(marker, 1)[1]
            row["checkpoint"] = (
                f"external_checkpoints/group_04_visualization/{suffix}"
            )
            changed = True
        if changed:
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            changed_files += 1
    return changed_files


def sanitize_mechanism_metadata() -> int:
    root = (
        ROOT
        / "experiments"
        / "4.7.2"
        / "reference_results"
        / "group_06_offset_mechanism_rank01_04"
    )
    changed = 0
    for path in sorted(root.glob("rank*/run_metadata.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        name = path.parent.name
        rank = int(name[4:6])
        expected = {
            "project_root": ".",
            "config": (
                "experiments/4.7.2/configs/group06_offset_mechanism/"
                f"{name}.yaml"
            ),
            "checkpoint": (
                "external_checkpoints/group_06_offset_mechanism/"
                f"{name}/best_model.pt"
            ),
        }
        if all(payload.get(key) == value for key, value in expected.items()):
            continue
        payload.update(expected)
        path.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        changed += 1
    return changed


def main() -> None:
    paths = sorted(
        path
        for pattern in CONFIG_GLOBS
        for path in BASELINES.glob(pattern)
    )
    if len(paths) != 28:
        raise ValueError(f"Expected 28 baseline configs, found {len(paths)}")
    changed = sum(sanitize(path) for path in paths)
    legacy_autodl_root = "/root/" + "autodl-tmp/paper4"
    text_replacements = {
        BASELINES / "eegnet" / "scripts" / "run_all_datasets.sh": {
            f'PAPER4_ROOT="${{PAPER4_ROOT:-{legacy_autodl_root}}}"':
                'PAPER4_ROOT="${PAPER4_ROOT:-$(cd "$ARCHIVE_ROOT/../.." && pwd)}"'
        },
        BASELINES / "atcnet" / "scripts" / "run_all_datasets.sh": {
            f'PAPER4_ROOT="${{PAPER4_ROOT:-{legacy_autodl_root}}}"':
                'PAPER4_ROOT="${PAPER4_ROOT:-$(cd "$ARCHIVE_ROOT/../.." && pwd)}"'
        },
        BASELINES / "eegnet" / "README.md": {
            legacy_autodl_root: "/path/to/LR-HORNet"
        },
        BASELINES / "atcnet" / "README.md": {
            legacy_autodl_root: "/path/to/LR-HORNet"
        },
        BASELINES / "eegmamba" / "addition" / "data_cache.py": {
            "little free space on " + legacy_autodl_root.rsplit("/", 1)[0]:
                "limited temporary disk space"
        },
        BASELINES / "labram" / "addition" / "data_cache.py": {
            "little free space on " + legacy_autodl_root.rsplit("/", 1)[0]:
                "limited temporary disk space"
        },
    }
    changed += sum(
        replace_exact(path, replacements)
        for path, replacements in text_replacements.items()
    )
    changed += int(sanitize_parameter_sources())
    changed += sanitize_visualization_paths()
    changed += sanitize_mechanism_metadata()
    print(
        f"Checked {len(paths)} baseline configs and archived path fields; "
        f"changed {changed} files."
    )


if __name__ == "__main__":
    main()
