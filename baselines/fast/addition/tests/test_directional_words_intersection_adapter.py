from __future__ import annotations

import sys
from pathlib import Path

import mne
import numpy as np


PAPER4_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAPER4_ROOT / "src"))

from eeg_pipeline.data.directional_words import _LazyDirectionalView


def _epochs(ch_names: list[str], *, sfreq: float, tmin: float, samples: int):
    info = mne.create_info(ch_names, sfreq=sfreq, ch_types="eeg")
    data = np.stack(
        [
            np.tile(np.arange(samples, dtype=np.float64), (len(ch_names), 1)),
            np.tile(np.arange(samples, dtype=np.float64) + 1, (len(ch_names), 1)),
        ]
    )
    events = np.asarray([[0, 0, 1], [samples + 1, 0, 1]], dtype=np.int64)
    return mne.EpochsArray(
        data,
        info,
        events=events,
        event_id={"UP2": 1},
        tmin=tmin,
        baseline=None,
        verbose="ERROR",
    )


def test_case_only_channel_names_are_renamed_without_preloading_a_copy():
    epochs = _epochs(["CP3", "Cz"], sfreq=500.0, tmin=-0.5, samples=750)
    view = _LazyDirectionalView(
        epochs,
        ["CP3", "Cz"],
        ["Cp3", "Cz"],
        target_sampling_rate_hz=500.0,
        time_window_seconds=(-0.5, 1.0),
    )
    assert view.ch_names == ["Cp3", "Cz"]
    assert view.get_data(item=slice(0, 1)).shape == (1, 2, 750)


def test_russian_grid_crops_to_zero_one_second_and_downsamples_to_250_hz():
    epochs = _epochs(["C3", "C4"], sfreq=500.0, tmin=-0.5, samples=750)
    view = _LazyDirectionalView(
        epochs,
        ["C3", "C4"],
        ["C3", "C4"],
        target_sampling_rate_hz=250.0,
        time_window_seconds=(0.0, 1.0),
    )
    output = view.get_data(item=slice(0, 1))
    assert output.shape == (1, 2, 250)
    assert view.info["sfreq"] == 250.0
    assert np.isclose(view.times[0], 0.0)
    assert np.isclose(view.times[-1], 0.996)


def test_native_250_hz_subject_uses_the_same_russian_output_grid():
    epochs = _epochs(["C3", "C4"], sfreq=250.0, tmin=-0.5, samples=375)
    view = _LazyDirectionalView(
        epochs,
        ["C3", "C4"],
        ["C3", "C4"],
        target_sampling_rate_hz=250.0,
        time_window_seconds=(0.0, 1.0),
    )
    assert view.get_data(item=slice(0, 1)).shape == (1, 2, 250)
    assert np.array_equal(view.times, np.arange(250) / 250.0)
