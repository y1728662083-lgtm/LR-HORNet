#!/usr/bin/env python3
"""Static smoke test for the four offset-generation builders and YAML files."""

from __future__ import annotations

from pathlib import Path

import yaml


PACKAGE = Path(__file__).resolve().parent.parent
BUILDERS = {
    "group4_patchLstm_deformableLstm": ("group4_patchLstm_deformableLstm.py", "build_group4_patchlstm_deformablelstm"),
    "group4_patchLstm_pointwiseDeformation": ("group4_patchlstm_core_ablation.py", "build_group4_patchlstm_pointwisedeformation"),
    "group4_patchLstm_staticDeformation": ("group4_patchlstm_core_ablation.py", "build_group4_patchlstm_staticdeformation"),
    "group4_patchLstm_groupedConvDeformation": ("group4_patchlstm_core_ablation.py", "build_group4_patchlstm_groupedconvdeformation"),
}


def main() -> None:
    configs = sorted((PACKAGE / "configs").glob("*.yaml"))
    if len(configs) != 4:
        raise RuntimeError(f"Expected four YAML files, found {configs}")
    configured = []
    for path in configs:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        name = config["model"]["name"]
        configured.append(name)
        if config["seed"] != 42 or config["split"]["fold_indices"] != list(range(10)):
            raise RuntimeError(f"Unexpected seed/folds in {path}")
        if config["data"]["n_electrode_groups"] != 4:
            raise RuntimeError(f"Expected four electrode groups in {path}")
    if set(configured) != set(BUILDERS):
        raise RuntimeError(f"YAML/model mismatch: {configured}")

    model_dir = PACKAGE / "src/eeg_pipeline/models"
    for name, (filename, builder) in BUILDERS.items():
        source = (model_dir / filename).read_text(encoding="utf-8")
        if f"def {builder}(" not in source:
            raise RuntimeError(f"{name}: missing builder {builder}")
    factory = (PACKAGE / "src/eeg_pipeline/models/factory.py").read_text(encoding="utf-8")
    missing = [name for name in BUILDERS if name not in factory]
    if missing:
        raise RuntimeError(f"Factory overlay is missing registrations: {missing}")
    grouped = (PACKAGE / "src/eeg_pipeline/models/group4_patchlstm_core_ablation.py").read_text(encoding="utf-8")
    for fragment in ("nn.Conv1d", '"groups": self.n_groups', '"kernel_size": self.kernel_size'):
        if fragment not in grouped:
            raise RuntimeError(f"Grouped-convolution implementation is missing {fragment!r}")
    print("4.7.1 builder/YAML/factory smoke test passed")


if __name__ == "__main__":
    main()
