#!/usr/bin/env python3
"""Write deterministic SHA-256 checksums for repository files."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "MANIFEST.sha256"
EXCLUDED_ANYWHERE = {".git", ".venv", "__pycache__"}
EXCLUDED_TOP_LEVEL = {"outputs", "data"}


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return (
        path.is_file()
        and path != OUTPUT
        and not any(part in EXCLUDED_ANYWHERE for part in relative.parts)
        and relative.parts[0] not in EXCLUDED_TOP_LEVEL
    )


def main() -> None:
    lines = []
    for path in sorted(path for path in ROOT.rglob("*") if included(path)):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        relative = path.relative_to(ROOT).as_posix()
        lines.append(f"{digest}  {relative}")
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {len(lines)} checksums to {OUTPUT.name}")


if __name__ == "__main__":
    main()
