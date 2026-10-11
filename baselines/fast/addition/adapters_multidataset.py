"""Dataset-generic electrode mapping for the existing official adapters.

The original ``addition/adapters.py`` was intentionally audited for the
BioSemi-128 ds003626 montage.  This module leaves that file untouched and
only widens its coordinate lookup to the published montages used by the two
additional datasets.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np
from torch import nn

import adapters as _base


MONTAGE_CANDIDATES = (
    "biosemi128",
    "biosemi64",
    "standard_1020",
    "standard_1005",
)


def _multidataset_positions(
    channel_names: Iterable[str],
) -> dict[str, np.ndarray]:
    """Resolve all channels against one authoritative MNE montage.

    A single montage must cover the complete input channel set.  Mixing
    coordinates from unrelated montages would silently change geometry, so a
    partial match is rejected.
    """

    names = tuple(str(name) for name in channel_names)
    if not names:
        raise ValueError("channel_names must not be empty")
    attempts: list[str] = []
    for montage_name in MONTAGE_CANDIDATES:
        available = _base._montage_positions(montage_name)
        missing = [name for name in names if name.casefold() not in available]
        if not missing:
            return {
                name: available[name.casefold()][1]
                for name in names
            }
        attempts.append(
            f"{montage_name}: missing {', '.join(missing[:8])}"
            + (" ..." if len(missing) > 8 else "")
        )
    raise ValueError(
        "No supported MNE montage covers every EEG channel. "
        + " | ".join(attempts)
    )


# FASTAdapter and LaBraMAdapter resolve this helper from the adapters module
# when they are instantiated.  The replacement is process-local and the
# audited ds003626 source file is not edited.
_base._biosemi_positions = _multidataset_positions

FASTAdapter = _base.FASTAdapter
EEGMambaAdapter = _base.EEGMambaAdapter
LaBraMAdapter = _base.LaBraMAdapter
ResampleTo200 = _base.ResampleTo200
patchify_200 = _base.patchify_200


def build_official_model(
    name: str,
    config: Mapping[str, Any],
    *,
    channel_names: Iterable[str],
    n_times: int,
    n_classes: int,
    input_config: Mapping[str, Any],
) -> nn.Module:
    return _base.build_official_model(
        name,
        config,
        channel_names=channel_names,
        n_times=n_times,
        n_classes=n_classes,
        input_config=input_config,
    )

