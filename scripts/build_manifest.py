#!/usr/bin/env python3
"""Write deterministic SHA-256 checksums for repository files."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "MANIFEST.sha256"
EXCLUDED_ANYWHERE = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
}
EXCLUDED_TOP_LEVEL = {"outputs", "data", "derivatives", "logs"}
EXCLUDED_RELATIVE_ROOTS = {
    Path("addition") / "bci2020_suite" / "runtime_configs",
}
TEXT_SUFFIXES = {
    ".bib",
    ".csv",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".sha256",
    ".svg",
    ".txt",
    ".yaml",
    ".yml",
}
TEXT_FILENAMES = {".gitattributes", ".gitignore", "LICENSE", "CBCR License"}


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return (
        path.is_file()
        and path != OUTPUT
        and not any(part in EXCLUDED_ANYWHERE for part in relative.parts)
        and relative.parts[0] not in EXCLUDED_TOP_LEVEL
        and not any(
            relative.is_relative_to(root)
            for root in EXCLUDED_RELATIVE_ROOTS
        )
    )


def sha256_file(path: Path) -> str:
    payload = path.read_bytes()
    if path.suffix.casefold() in TEXT_SUFFIXES or path.name in TEXT_FILENAMES:
        payload = payload.replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    lines = []
    for path in sorted(path for path in ROOT.rglob("*") if included(path)):
        digest = sha256_file(path)
        relative = path.relative_to(ROOT).as_posix()
        lines.append(f"{digest}  {relative}")
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {len(lines)} checksums to {OUTPUT.name}")


if __name__ == "__main__":
    main()
