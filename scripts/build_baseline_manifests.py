#!/usr/bin/env python3
"""Rebuild deterministic checksum manifests for bundled baseline archives."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGETS = {
    ROOT / "baselines" / "eegnet": "MANIFEST_SHA256.txt",
    ROOT / "baselines" / "atcnet": "MANIFEST_SHA256.txt",
    ROOT / "baselines" / "fast": "MANIFEST_SHA256_FAST_ARCHIVE.txt",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    for root, manifest_name in TARGETS.items():
        manifest_path = root / manifest_name
        paths = sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path != manifest_path
            and "__pycache__" not in path.parts
        )
        lines = [
            f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
            for path in paths
        ]
        manifest_path.write_text(
            "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
        )
        print(f"Wrote {len(lines)} checksums to {manifest_path}")


if __name__ == "__main__":
    main()
