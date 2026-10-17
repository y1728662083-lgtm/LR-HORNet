"""Repository-relative path normalization shared by experiment runners.

The archived baseline bundles execute from different working directories.  A
relative ``data.root`` must therefore be anchored explicitly instead of being
left to the caller's current directory.  Absolute paths are retained for
historical configurations and portable launchers that replace them at runtime.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parent


def _is_cross_platform_absolute(value: str | Path) -> bool:
    raw = str(value)
    return (
        Path(raw).is_absolute()
        or PurePosixPath(raw).is_absolute()
        or PureWindowsPath(raw).is_absolute()
    )


def resolve_repository_path(
    value: str | Path,
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> Path:
    """Resolve one path against the repository root when it is relative."""
    path = Path(value).expanduser()
    if _is_cross_platform_absolute(value):
        return path
    return Path(repository_root).expanduser().resolve() / path


def resolve_config_data_root(
    config: MutableMapping[str, Any],
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> MutableMapping[str, Any]:
    """Normalize ``config['data']['root']`` in place and return ``config``."""
    data = config.get("data")
    if not isinstance(data, MutableMapping) or "root" not in data:
        raise ValueError("Configuration must define data.root")
    configured = data["root"]
    if _is_cross_platform_absolute(configured):
        # Preserve foreign-platform absolute strings byte-for-byte.  For
        # example, ``/root/...`` is not considered absolute by WindowsPath.
        data["root"] = str(configured)
    else:
        data["root"] = str(
            resolve_repository_path(
                configured,
                repository_root=repository_root,
            )
        )
    return config
