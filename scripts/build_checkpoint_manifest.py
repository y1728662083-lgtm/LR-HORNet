#!/usr/bin/env python3
"""Build the deterministic manifest for released LR-HORNet checkpoints."""

from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_ROOT = ROOT / "checkpoints" / "main"
OUTPUT_PATH = CHECKPOINT_ROOT / "manifest.csv"
RESULT_TABLE = ROOT / "results" / "4.4.1" / "fold_level_results_with_val_acc.csv"
EXPECTED_FOLDS = {
    "ds003626": 10,
    "bimodal_inner_speech": 4,
    "directional_words_russian": 12,
    "directional_words_spanish": 10,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    with RESULT_TABLE.open(newline="", encoding="utf-8-sig") as handle:
        result_rows = [
            row
            for row in csv.DictReader(handle)
            if row["model"] == "LR-HORNet"
        ]
    result_index = {
        (row["dataset_key"], int(row["fold"])): row
        for row in result_rows
    }
    if len(result_index) != sum(EXPECTED_FOLDS.values()):
        raise ValueError(
            f"Expected {sum(EXPECTED_FOLDS.values())} LR-HORNet result rows, "
            f"found {len(result_index)}"
        )

    rows: list[dict[str, str | int]] = []
    for dataset, expected_count in EXPECTED_FOLDS.items():
        dataset_dir = CHECKPOINT_ROOT / dataset
        paths = sorted(dataset_dir.glob("fold_*.pt"))
        if len(paths) != expected_count:
            raise ValueError(
                f"{dataset}: expected {expected_count} checkpoints, "
                f"found {len(paths)}"
            )
        observed_folds: list[int] = []
        for path in paths:
            match = re.fullmatch(r"fold_(\d{2})\.pt", path.name)
            if match is None:
                raise ValueError(f"Unexpected checkpoint name: {path}")
            fold = int(match.group(1))
            observed_folds.append(fold)
            result = result_index[(dataset, fold)]
            rows.append(
                {
                    "dataset": dataset,
                    "fold": fold,
                    "selected_epoch": int(result["selected_epoch"]),
                    "validation_accuracy": result["val_accuracy"],
                    "test_accuracy": result["accuracy"],
                    "source_metrics_path": result["metrics_path"],
                    "source_metrics_sha256": result["metrics_sha256"],
                    "path": path.relative_to(ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        if observed_folds != list(range(1, expected_count + 1)):
            raise ValueError(f"{dataset}: invalid fold set {observed_folds}")

    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "dataset",
                "fold",
                "selected_epoch",
                "validation_accuracy",
                "test_accuracy",
                "source_metrics_path",
                "source_metrics_sha256",
                "path",
                "bytes",
                "sha256",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} checkpoint records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
