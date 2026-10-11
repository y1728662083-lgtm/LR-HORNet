#!/usr/bin/env python3
"""Regenerate the checked-in paper figures from local source tables."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIGURE_ROOT = ROOT / "figures"
SKIP_SECTIONS = {
    "4.5": "plot-level LR-HORNet row conflicts with the formal 4.5 summary",
    "4.6.3": "legacy 1/2-group rows conflict with the formal incomplete-run record",
    "4.7.1": "legacy LSTM table counts a duplicated fold as ten folds",
}


def commands_for(section: Path) -> list[Path]:
    return sorted(section.glob("plot*.py"))


def main() -> None:
    commands: list[Path] = []
    for section in sorted(path for path in FIGURE_ROOT.iterdir() if path.is_dir()):
        if section.name in SKIP_SECTIONS:
            print(f"[skip] {section.name}: {SKIP_SECTIONS[section.name]}")
            continue
        commands.extend(commands_for(section))

    if not commands:
        raise RuntimeError(f"No figure scripts found under {FIGURE_ROOT}")

    for script in commands:
        relative = script.relative_to(ROOT)
        print(f"[figure] {relative}", flush=True)
        subprocess.run(
            [sys.executable, script.name],
            cwd=script.parent,
            check=True,
        )

    print(f"Regenerated figures with {len(commands)} scripts.")


if __name__ == "__main__":
    main()
