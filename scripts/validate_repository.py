#!/usr/bin/env python3
"""Validate repository structure and portability without loading EEG data."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import re
import tokenize
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".pyc",
    ".log",
    ".vsdx",
}
WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")
RELEASED_CHECKPOINT_ROOT = ROOT / "checkpoints" / "main"
MACHINE_PATH_MARKERS = (
    "E:" + "\\model",
    "C:" + "\\Users\\86159",
    "/root/" + "autodl-tmp",
)
BCI2020_ARCHIVED_CONFIG_SHA256 = {
    "configs/main/bci2020.yaml": (
        "8e1b695a3a0c10a848b73db3b9c38673f7985e087c29c6dec737093d1e9c75a9"
    ),
    "baselines/eegnet/configs/bci2020.yaml": (
        "9501b8c1a59c857c49a244f248b95c11a6f3ed6c97435bcd618ddb103f79dd5f"
    ),
    "baselines/atcnet/configs/bci2020.yaml": (
        "7c3b708210b7d76f02794b91a964ed5463faadf67f01a713e5e5f20a0806c062"
    ),
    "baselines/eegmamba/addition/configs/bci2020_official_baselines.yaml": (
        "f7e96d24d8e0c41c778c4d9cdfcd9ca3f6e23e6d10dc8ea04835637ce8594b4f"
    ),
    "baselines/fast/addition/configs/fast_native_bci2020.yaml": (
        "2b46d308cd395e3e5ad0afcff3e411f9aa783d75051558b171d272968f87a76c"
    ),
    "baselines/labram/addition/configs/bci2020_official_baselines.yaml": (
        "dcf30d612be7fbaf454f9d87e19fb7f176f0bf1f454787101a0e14456819edf7"
    ),
    "addition/bci2020_preprocessing/configs/bci2020_standard.yaml": (
        "62544f03aa8da0d5fc967ffd913b97ad425e1d87f53d51d29fe365b9f4fbc19b"
    ),
}
BCI2020_ARCHIVED_PATH_SOURCES = set(BCI2020_ARCHIVED_CONFIG_SHA256) | {
    "addition/bci2020_suite/preflight_all_models.py",
}


def validate_configs() -> None:
    config_roots = [ROOT / "configs", ROOT / "experiments"]
    config_roots.extend((ROOT / "baselines").glob("*/configs"))
    config_roots.extend((ROOT / "baselines").glob("*/addition/configs"))
    paths = sorted(
        path
        for config_root in config_roots
        for pattern in ("*.yaml", "*.yml")
        for path in config_root.rglob(pattern)
    )
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
                relative = path.relative_to(ROOT).as_posix()
                if relative not in BCI2020_ARCHIVED_CONFIG_SHA256:
                    raise ValueError(f"Absolute data.root in {path}: {value}")


def validate_files() -> None:
    forbidden = [
        path.relative_to(ROOT)
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.suffix.lower() in FORBIDDEN_SUFFIXES
        and not (
            path.suffix.lower() == ".pt"
            and path.is_relative_to(RELEASED_CHECKPOINT_ROOT)
        )
    ]
    if forbidden:
        raise ValueError(f"Forbidden generated artifacts: {forbidden}")


def validate_portable_text_paths() -> None:
    text_suffixes = {
        ".csv",
        ".json",
        ".md",
        ".py",
        ".sh",
        ".txt",
        ".yaml",
        ".yml",
    }
    offenders: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        relative_path = path.relative_to(ROOT)
        if relative_path.is_relative_to(
            Path("addition") / "bci2020_suite" / "runtime_configs"
        ):
            continue
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        relative = relative_path.as_posix()
        matched = [marker for marker in MACHINE_PATH_MARKERS if marker in text]
        if matched and not (
            relative in BCI2020_ARCHIVED_PATH_SOURCES
            and matched == ["/root/" + "autodl-tmp"]
        ):
            offenders.append(path.relative_to(ROOT))
    if offenders:
        raise ValueError(f"Machine-specific paths remain in: {offenders}")


def validate_baseline_manifests() -> None:
    targets = {
        ROOT / "baselines" / "eegnet": "MANIFEST_SHA256.txt",
        ROOT / "baselines" / "atcnet": "MANIFEST_SHA256.txt",
        ROOT / "baselines" / "fast": "MANIFEST_SHA256_FAST_ARCHIVE.txt",
    }
    for root, manifest_name in targets.items():
        manifest_path = root / manifest_name
        expected: dict[str, str] = {}
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            digest, relative = line.split("  ", 1)
            expected[relative] = digest
        paths = sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path != manifest_path
            and "__pycache__" not in path.parts
        )
        observed = {
            path.relative_to(root).as_posix(): sha256_file(path)
            for path in paths
        }
        if observed != expected:
            raise ValueError(f"Stale baseline manifest: {manifest_path}")


def validate_root_manifest() -> None:
    manifest_path = ROOT / "MANIFEST.sha256"
    expected: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        expected[relative] = digest
    excluded_anywhere = {
        ".git",
        ".venv",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "__pycache__",
    }
    excluded_top_level = {"outputs", "data", "derivatives", "logs"}
    excluded_relative_roots = {
        Path("addition") / "bci2020_suite" / "runtime_configs",
    }
    paths = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path != manifest_path
        and not any(
            part in excluded_anywhere
            for part in path.relative_to(ROOT).parts
        )
        and path.relative_to(ROOT).parts[0] not in excluded_top_level
        and not any(
            path.relative_to(ROOT).is_relative_to(root)
            for root in excluded_relative_roots
        )
    )
    observed = {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in paths
    }
    if observed != expected:
        raise ValueError("Root MANIFEST.sha256 is stale")


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_bci2020_archived_configs() -> None:
    """Keep the uploaded BCI2020 source configs byte-identical to the runs."""
    for relative, expected in BCI2020_ARCHIVED_CONFIG_SHA256.items():
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing archived BCI2020 config: {relative}")
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(
                f"BCI2020 source config drifted: {relative}: "
                f"expected {expected}, observed {observed}"
            )


def validate_bci2020_results() -> None:
    """Enforce the paper's complete-repeat BCI2020 aggregation contract."""
    root = ROOT / "results" / "4.4.1" / "bci2020"
    overall = json.loads(
        (root / "overall_summary.json").read_text(encoding="utf-8")
    )
    expected_fields = {
        "model": "lr_hornet",
        "dataset": "bci2020",
        "protocol": "calibrated_loso",
        "fixed_seed": 42,
        "included_repeats": list(range(1, 10)),
        "folds": 15,
        "repeats_per_fold": 9,
        "runs_complete": 135,
        "parameters_total": 5479,
    }
    observed_fields = {key: overall.get(key) for key in expected_fields}
    if observed_fields != expected_fields:
        raise ValueError(f"Unexpected BCI2020 LR-HORNet coverage: {observed_fields}")
    if not math.isclose(
        float(overall["test_accuracy_mean_across_fold_means"]),
        0.21684700992371753,
        abs_tol=1e-15,
    ):
        raise ValueError("Unexpected BCI2020 LR-HORNet test mean")

    with (root / "summary_by_fold.csv").open(
        newline="", encoding="utf-8-sig"
    ) as stream:
        fold_rows = list(csv.DictReader(stream))
    if len(fold_rows) != 15:
        raise ValueError(f"Unexpected BCI2020 fold count: {len(fold_rows)}")
    if {int(row["fold"]) for row in fold_rows} != set(range(1, 16)):
        raise ValueError("BCI2020 fold summary must cover folds 1–15 exactly")
    if any(int(row["repeats_complete"]) != 9 for row in fold_rows):
        raise ValueError("BCI2020 fold summary must use complete repeats 1–9 only")

    with (root / "model_summary.csv").open(
        newline="", encoding="utf-8-sig"
    ) as stream:
        baseline_rows = list(csv.DictReader(stream))
    expected_models = {"eegnet", "atcnet", "eegmamba", "fast", "labram"}
    if {row["model"] for row in baseline_rows} != expected_models:
        raise ValueError("Unexpected BCI2020 baseline model coverage")
    if any(int(row["folds_complete"]) != 15 for row in baseline_rows):
        raise ValueError("Every BCI2020 baseline must contain all 15 folds")


def validate_checkpoint_release() -> None:
    manifest_path = RELEASED_CHECKPOINT_ROOT / "manifest.csv"
    with manifest_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    result_table_path = (
        ROOT / "results" / "4.4.1" / "fold_level_results_with_val_acc.csv"
    )
    with result_table_path.open(newline="", encoding="utf-8-sig") as stream:
        result_rows = [
            row
            for row in csv.DictReader(stream)
            if row.get("model") == "LR-HORNet"
        ]
    result_index = {
        (row["dataset_key"], int(row["fold"])): row
        for row in result_rows
    }
    expected = {
        "ds003626": 10,
        "bimodal_inner_speech": 4,
        "directional_words_russian": 12,
        "directional_words_spanish": 10,
    }
    observed: dict[str, int] = {}
    for row in rows:
        path = ROOT / row["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing released checkpoint: {path}")
        if int(row["bytes"]) != path.stat().st_size:
            raise ValueError(f"Checkpoint size mismatch: {path}")
        if row["sha256"] != sha256_file(path):
            raise ValueError(f"Checkpoint hash mismatch: {path}")
        dataset = row["dataset"]
        fold = int(row["fold"])
        result = result_index.get((dataset, fold))
        if result is None:
            raise ValueError(
                f"Missing LR-HORNet result provenance for {dataset} fold {fold}"
            )
        expected_provenance = {
            "selected_epoch": result["selected_epoch"],
            "validation_accuracy": result["val_accuracy"],
            "test_accuracy": result["accuracy"],
            "source_metrics_path": result["metrics_path"],
            "source_metrics_sha256": result["metrics_sha256"],
        }
        observed_provenance = {
            field: row[field] for field in expected_provenance
        }
        if observed_provenance != expected_provenance:
            raise ValueError(
                f"Checkpoint provenance mismatch for {dataset} fold {fold}: "
                f"{observed_provenance}"
            )
        observed[dataset] = observed.get(dataset, 0) + 1
    if observed != expected:
        raise ValueError(f"Unexpected checkpoint coverage: {observed}")


def validate_reconciled_results() -> None:
    with (ROOT / "results" / "4.5" / "reported_results_summary.csv").open(
        newline="", encoding="utf-8-sig"
    ) as stream:
        rows_45 = list(csv.DictReader(stream))
    if len(rows_45) != 5:
        raise ValueError(f"Unexpected Section 4.5 model count: {len(rows_45)}")
    full_45 = next(
        (
            row
            for row in rows_45
            if row["model"] == "group4_patchLstm_deformableLstm"
        ),
        None,
    )
    with (
        ROOT / "results" / "4.4.1" / "model_summary_all_datasets.csv"
    ).open(newline="", encoding="utf-8-sig") as stream:
        main_rows = list(csv.DictReader(stream))
    full_main = next(
        (
            row
            for row in main_rows
            if row["dataset"] == "ds003626" and row["model"] == "LR-HORNet"
        ),
        None,
    )
    if full_45 is None or full_main is None or not math.isclose(
        float(full_45["accuracy_mean"]),
        float(full_main["accuracy_mean"]),
        abs_tol=1e-12,
    ):
        raise ValueError("Section 4.5 does not reuse the 4.4.1 main result")
    reference_root_45 = ROOT / "experiments" / "4.5" / "reference_results"
    if len(list(reference_root_45.rglob("test_metrics.json"))) != 50:
        raise ValueError("Section 4.5 must contain 50 source metric files")
    if len(list(reference_root_45.rglob("training_log.csv"))) != 50:
        raise ValueError("Section 4.5 must contain 50 source training logs")

    with (ROOT / "results" / "4.6.3" / "run_level_results.csv").open(
        newline="", encoding="utf-8-sig"
    ) as stream:
        rows_463 = list(csv.DictReader(stream))
    coverage_463: dict[str, int] = {}
    for row in rows_463:
        coverage_463[row["condition_key"]] = (
            coverage_463.get(row["condition_key"], 0) + 1
        )
    if coverage_463 != {f"group_{count}": 5 for count in (1, 2, 4, 8, 16)}:
        raise ValueError(f"Unexpected Section 4.6.3 coverage: {coverage_463}")

    with (ROOT / "results" / "4.7.1" / "run_level_results.csv").open(
        newline="", encoding="utf-8-sig"
    ) as stream:
        rows_471 = list(csv.DictReader(stream))
    coverage_471: dict[str, int] = {}
    for row in rows_471:
        coverage_471[row["model_key"]] = (
            coverage_471.get(row["model_key"], 0) + 1
        )
    if len(coverage_471) != 4 or set(coverage_471.values()) != {10}:
        raise ValueError(f"Unexpected Section 4.7.1 coverage: {coverage_471}")

    with (ROOT / "results" / "4.7.2" / "summary" / "completion_status.csv").open(
        newline="", encoding="utf-8-sig"
    ) as stream:
        rows_472 = list(csv.DictReader(stream))
    observed_ranks = [int(row["rank"]) for row in rows_472]
    if observed_ranks != [1, 2, 3, 4]:
        raise ValueError(f"Unexpected Section 4.7.2 ranks: {observed_ranks}")
    if any(
        row["causal_complete"] != "1"
        or row["artificial_shift_complete"] != "1"
        or row["included_in_statistics"] != "1"
        for row in rows_472
    ):
        raise ValueError("Section 4.7.2 includes an incomplete ranked result")
    rank5_paths = [
        path.relative_to(ROOT)
        for base in (ROOT / "experiments" / "4.7.2", ROOT / "results" / "4.7.2")
        for path in base.rglob("*")
        if "rank05" in path.name.lower()
    ]
    if rank5_paths:
        raise ValueError(f"Unexpected Section 4.7.2 rank-5 artifacts: {rank5_paths}")


def validate_flop_boundary_counts() -> None:
    path = ROOT / "figures" / "4.4.1" / "cross_dataset_figure_data.csv"
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    observed = {
        row["dataset_key"]: int(row["flops"])
        for row in rows
        if row["model"] == "LR-HORNet"
    }
    expected = {
        "ds003626": 176_805_376,
        "bimodal_inner_speech": 23_547_904,
        "directional_words_russian": 10_156_800,
        "directional_words_spanish": 35_860_928,
    }
    if observed != expected:
        raise ValueError(f"Unexpected LR-HORNet FLOP counts: {observed}")


def validate_patch_boundary_contract() -> None:
    model_path = (
        ROOT
        / "src"
        / "eeg_pipeline"
        / "models"
        / "causal1_group8PatchLSTM_EEGNet.py"
    )
    tree = ast.parse(model_path.read_text(encoding="utf-8"))
    required_classes = {
        "_ElectrodeGroupPatchLSTMTemporal",
        "causal1_group8PatchLSTM_EEGNet",
    }
    discovered: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name not in required_classes:
            continue
        init = next(
            (
                item
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                and item.name == "__init__"
            ),
            None,
        )
        if init is None:
            raise ValueError(f"Missing __init__ in {node.name}")
        arguments = {
            argument.arg
            for argument in (*init.args.args, *init.args.kwonlyargs)
        }
        if "pad_incomplete_patch" not in arguments:
            raise ValueError(
                f"{node.name} does not expose pad_incomplete_patch"
            )
        discovered.add(node.name)
    if discovered != required_classes:
        raise ValueError(
            f"Missing LR boundary implementation classes: "
            f"{sorted(required_classes - discovered)}"
        )

    expected = {
        "ds003626.yaml": False,
        "bimodal_inner_speech.yaml": True,
        "directional_words_russian.yaml": True,
        "directional_words_spanish.yaml": True,
        "bci2020.yaml": True,
    }
    for filename, expected_value in expected.items():
        path = ROOT / "configs" / "main" / filename
        with path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        observed = config["model"]["kwargs"].get("pad_incomplete_patch")
        if observed is not expected_value:
            raise ValueError(
                f"Unexpected pad_incomplete_patch in {filename}: {observed!r}"
            )


def validate_required_paths() -> None:
    required = [
        ROOT / "main.py",
        ROOT / "pytest.ini",
        ROOT / "src" / "eeg_pipeline" / "models" / "group4_patchLstm_deformableLstm.py",
        ROOT / "configs" / "main" / "ds003626.yaml",
        ROOT / "configs" / "main" / "bci2020.yaml",
        ROOT / "src" / "eeg_pipeline" / "data" / "bci2020.py",
        ROOT / "addition" / "bci2020_suite" / "preflight_all_models.py",
        ROOT / "addition" / "bci2020_suite" / "dispatch_all_models_4gpu.py",
        ROOT / "addition" / "bci2020_preprocessing" / "preprocess_bci2020.py",
        ROOT / "addition" / "bci2020_preprocessing" / "configs" / "bci2020_standard.yaml",
        ROOT / "run_bci2020_all_models_4gpu.sh",
        ROOT / "run_bci2020_lr_hornet_repeat10_4gpu.sh",
        ROOT / "run_bci2020_other_models_then_lr_hornet_repeat10_4gpu.sh",
        ROOT / "scripts" / "smoke_test_bci2020_checkpoints.py",
        ROOT / "tests" / "test_bci2020_adapter.py",
        ROOT / "tests" / "test_bci2020_model.py",
        ROOT / "baselines" / "eegnet" / "configs" / "bci2020.yaml",
        ROOT / "baselines" / "atcnet" / "configs" / "bci2020.yaml",
        ROOT / "baselines" / "eegmamba" / "addition" / "configs" / "bci2020_official_baselines.yaml",
        ROOT / "baselines" / "fast" / "addition" / "configs" / "fast_native_bci2020.yaml",
        ROOT / "baselines" / "labram" / "addition" / "configs" / "bci2020_official_baselines.yaml",
        ROOT / "baselines" / "eegmamba" / "src" / "eeg_pipeline" / "data" / "bci2020.py",
        ROOT / "baselines" / "fast" / "src" / "eeg_pipeline" / "data" / "bci2020.py",
        ROOT / "baselines" / "labram" / "src" / "eeg_pipeline" / "data" / "bci2020.py",
        ROOT / "docs" / "RESULTS_AUDIT.md",
        ROOT / "baselines" / "eegmamba" / "source_manifest.json",
        ROOT / "baselines" / "labram" / "source_manifest.json",
        ROOT / "baselines" / "fast" / "addition" / "source_manifest.json",
        RELEASED_CHECKPOINT_ROOT / "manifest.csv",
    ]
    missing = [path.relative_to(ROOT) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")


def validate_python_syntax() -> None:
    paths = sorted(
        path
        for path in ROOT.rglob("*.py")
        if ".git" not in path.relative_to(ROOT).parts
    )
    for path in paths:
        with tokenize.open(path) as stream:
            source = stream.read()
        compile(source, str(path), "exec")


def main() -> None:
    validate_required_paths()
    validate_bci2020_archived_configs()
    validate_configs()
    validate_files()
    validate_portable_text_paths()
    validate_baseline_manifests()
    validate_root_manifest()
    validate_main_results()
    validate_bci2020_results()
    validate_checkpoint_release()
    validate_reconciled_results()
    validate_flop_boundary_counts()
    validate_patch_boundary_contract()
    validate_python_syntax()
    print("Repository validation passed.")


if __name__ == "__main__":
    main()
