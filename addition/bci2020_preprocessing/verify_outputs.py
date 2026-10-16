"""Verify all standard BCI2020 preprocessing outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd


EVENT_ID = {"Hello": 1, "Help me": 2, "Stop": 3, "Thank you": 4, "Yes": 5}


def verify(root: str | Path, expected_records: int = 45) -> None:
    root = Path(root).expanduser().resolve()
    summary_path = root / "preprocessing_summary.csv"
    manifest_path = root / "preprocessing_manifest.json"
    if not summary_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Missing preprocessing summary/manifest in {root}")
    summary = pd.read_csv(summary_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(summary) != expected_records:
        raise RuntimeError(f"Expected {expected_records} records, found {len(summary)}")
    if (summary["trials_kept"] <= 0).any():
        raise RuntimeError("At least one record retained no trials")
    files = sorted(root.glob("sub-*/eeg/*_desc-preproc_epo.fif"))
    if len(files) != expected_records:
        raise RuntimeError(f"Expected {expected_records} FIF files, found {len(files)}")
    reference_channels = None
    reference_times = None
    per_subject: dict[str, np.ndarray] = {}
    total = 0
    for path in files:
        epochs = mne.read_epochs(path, preload=False, verbose="ERROR")
        if any(name not in EVENT_ID or EVENT_ID[name] != code for name, code in epochs.event_id.items()):
            raise RuntimeError(f"Invalid event map in {path}")
        if len(epochs.ch_names) != 64 or not np.isclose(epochs.info["sfreq"], 256.0):
            raise RuntimeError(f"Invalid input dimensions in {path}")
        if len(epochs.times) != 795:
            raise RuntimeError(f"Expected 795 samples in {path}")
        if not all(np.linalg.norm(ch["loc"][:3]) > 0 for ch in epochs.info["chs"]):
            raise RuntimeError(f"Missing channel positions in {path}")
        channels = tuple(epochs.ch_names)
        if reference_channels is None:
            reference_channels = channels
            reference_times = epochs.times.copy()
        elif channels != reference_channels or not np.array_equal(epochs.times, reference_times):
            raise RuntimeError(f"Channel/time mismatch in {path}")
        subject = path.name.split("_", 1)[0]
        labels = epochs.events[:, 2] - 1
        per_subject.setdefault(subject, np.zeros(5, dtype=np.int64))
        per_subject[subject] += np.bincount(labels, minlength=5)
        total += len(epochs)
    insufficient = {key: value.tolist() for key, value in per_subject.items() if (value < 2).any()}
    if insufficient:
        raise RuntimeError(f"Insufficient subject/class trials: {insufficient}")
    print("BCI2020 standard preprocessing verification passed")
    print(f"records={len(files)} epochs={total} shape=(64, 795) sfreq=256")
    print(f"operations={manifest['operations']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-records", type=int, default=45)
    args = parser.parse_args()
    verify(args.root, args.expected_records)


if __name__ == "__main__":
    main()

