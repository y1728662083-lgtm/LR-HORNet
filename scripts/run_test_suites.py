#!/usr/bin/env python3
"""Run repository and standalone baseline tests in isolated interpreters."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUITES = (
    ("integrated repository", ROOT / "tests"),
    ("EEGNet baseline", ROOT / "baselines" / "eegnet" / "tests"),
    ("ATCNet baseline", ROOT / "baselines" / "atcnet" / "tests"),
    (
        "FAST baseline",
        ROOT / "baselines" / "fast" / "addition" / "tests",
    ),
)


def main() -> int:
    failed: list[str] = []
    for name, path in SUITES:
        if not path.is_dir():
            print(f"[FAIL] {name}: missing test directory {path}", flush=True)
            failed.append(name)
            continue
        print(f"[RUN] {name}: {path.relative_to(ROOT)}", flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", str(path)],
            cwd=ROOT,
            check=False,
        )
        if result.returncode:
            print(
                f"[FAIL] {name}: pytest exit code {result.returncode}",
                flush=True,
            )
            failed.append(name)
        else:
            print(f"[PASS] {name}", flush=True)

    if failed:
        print(f"[FAIL] Test suites failed: {', '.join(failed)}", flush=True)
        return 1
    print(f"[PASS] All {len(SUITES)} isolated test suites passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
