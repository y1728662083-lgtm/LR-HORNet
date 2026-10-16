"""Read one subject through Paper4's shared model-input wrapper."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_pipeline.data.dataset import EEGTaskDataset
from eeg_pipeline.data.registry import load_task_sessions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--subject", type=int, default=1)
    args = parser.parse_args()
    sessions = load_task_sessions(
        "bci2020",
        args.dataset_root,
        "imagined_speech",
        subjects=(args.subject,),
        sessions=(0, 1, 2),
        preload=False,
        options={"derivatives_root": "derivatives/paper4/bci2020_standard"},
    )
    dataset = EEGTaskDataset(sessions, standardize=True, n_electrode_groups=4)
    eeg, label = dataset[0]
    if tuple(eeg.shape) != (64, 795):
        raise RuntimeError(f"Unexpected sample shape: {tuple(eeg.shape)}")
    print("BCI2020 model-input smoke test passed")
    print(
        f"trials={len(dataset)} sample={tuple(eeg.shape)} label={label.item()} "
        f"sfreq={dataset.sampling_rate_hz} groups={dataset.electrode_group_sizes}"
    )


if __name__ == "__main__":
    main()

