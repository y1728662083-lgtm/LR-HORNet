"""PyTorch dataset adapter for task-selected EEG recordings."""

from __future__ import annotations

import operator
import sys
from bisect import bisect_right
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypeAlias

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from eeg_pipeline.data.ds003626 import (
        DS003626TaskData,
        build_biosemi128_electrode_layout,
        build_spatial_electrode_layout,
        load_task_sessions,
    )
else:
    from .ds003626 import (
        DS003626TaskData,
        build_biosemi128_electrode_layout,
        build_spatial_electrode_layout,
        load_task_sessions,
    )


EEGTransform: TypeAlias = Callable[[Tensor], Tensor]


def _task_channel_names(task_data: DS003626TaskData) -> tuple[str, ...]:
    if task_data.channel_names is not None:
        return tuple(task_data.channel_names)
    if task_data.channel_indices is not None:
        return tuple(
            task_data.epochs.ch_names[index]
            for index in task_data.channel_indices
        )
    return tuple(task_data.epochs.ch_names)


def _task_times(task_data: DS003626TaskData) -> np.ndarray:
    if task_data.time_slice is None:
        return task_data.epochs.times
    start, stop, step = task_data.time_slice
    return task_data.epochs.times[start:stop:step]


def _task_sfreq(task_data: DS003626TaskData) -> float:
    if task_data.sampling_rate_hz is not None:
        return float(task_data.sampling_rate_hz)
    return float(task_data.epochs.info["sfreq"])


def _build_electrode_layout(
    task_data: DS003626TaskData,
    n_electrode_groups: int,
):
    """Build a spatial grouping from the montage stored in an MNE object.

    The original ds003626 path keeps its BioSemi-128 lookup so existing
    experiments retain exactly the same channel permutation. Other datasets
    use the channel locations carried by their preprocessed MNE objects.
    """
    epochs = task_data.epochs
    channel_names = _task_channel_names(task_data)
    source_indices = (
        tuple(range(len(epochs.ch_names)))
        if task_data.channel_indices is None
        else task_data.channel_indices
    )
    if len(source_indices) != len(channel_names):
        raise ValueError(
            "Effective channel names and indices must have equal length"
        )
    if len(channel_names) == 128:
        try:
            return build_biosemi128_electrode_layout(
                channel_names,
                n_electrode_groups,
            )
        except ValueError:
            pass

    positions = np.asarray(
        [
            epochs.info["chs"][index]["loc"][:3]
            for index in source_indices
        ],
        dtype=np.float64,
    )
    invalid = ~np.isfinite(positions).all(axis=1) | (
        np.linalg.norm(positions, axis=1) <= np.finfo(float).eps
    )
    if invalid.any():
        missing = [
            channel_names[index]
            for index in np.flatnonzero(invalid).tolist()
        ]
        raise ValueError(
            "EEG channel positions are required for spatial grouping; "
            f"missing positions for {missing}. Ensure preprocessing attaches "
            "the dataset montage before saving epochs."
        )
    return build_spatial_electrode_layout(
        channel_names,
        positions,
        n_electrode_groups,
    )


class EEGTaskDataset(Dataset[tuple[Tensor, Tensor]]):
    """Expose one or more preprocessed task sessions as PyTorch samples."""

    def __init__(
        self,
        task_sessions: Sequence[DS003626TaskData],
        *,
        standardize: bool = False,
        n_electrode_groups: int = 1,
        transform: EEGTransform | None = None,
        standardize_eps: float = 1e-8,
    ) -> None:
        sessions = tuple(task_sessions)
        if not sessions:
            raise ValueError("task_sessions must contain at least one session")
        if not all(isinstance(item, DS003626TaskData) for item in sessions):
            raise TypeError(
                "Every item in task_sessions must be DS003626TaskData"
            )
        if not isinstance(standardize, bool):
            raise TypeError("standardize must be a bool")
        if transform is not None and not callable(transform):
            raise TypeError("transform must be callable or None")
        if not np.isfinite(standardize_eps) or standardize_eps <= 0:
            raise ValueError("standardize_eps must be a finite positive number")

        reference = sessions[0]
        reference_channels = _task_channel_names(reference)
        reference_times = _task_times(reference)
        reference_sfreq = _task_sfreq(reference)

        cumulative_sizes: list[int] = []#用于存放session的trail数
        total_trials = 0
        #逐一检查任务、类别、通道、采样率、时间轴和标签长度是否一致、session是否为空
        for item in sessions:
            if item.task != reference.task:
                raise ValueError(
                    "All task sessions must have the same task; got "
                    f"{reference.task!r} and {item.task!r}"
                )
            if item.class_names != reference.class_names:
                raise ValueError("All task sessions must have the same classes")
            if _task_channel_names(item) != reference_channels:
                raise ValueError(
                    f"Channel mismatch in {item.subject}/{item.session}"
                )
            if not np.isclose(
                _task_sfreq(item), reference_sfreq
            ):
                raise ValueError(
                    f"Sampling-rate mismatch in {item.subject}/{item.session}"
                )
            if not np.array_equal(_task_times(item), reference_times):
                raise ValueError(
                    f"Time-axis mismatch in {item.subject}/{item.session}"
                )
            if len(item.epochs) != len(item.labels):
                raise ValueError(
                    f"EEG/label length mismatch in "
                    f"{item.subject}/{item.session}"
                )
            if len(item.labels) == 0:
                raise ValueError(
                    f"Empty task session: {item.subject}/{item.session}"
                )

            total_trials += len(item.labels)
            cumulative_sizes.append(total_trials)

        electrode_layout = _build_electrode_layout(
            reference,
            n_electrode_groups,
        )

        self._task_sessions = sessions
        self._cumulative_sizes = tuple(cumulative_sizes)
        self._channel_permutation = np.asarray(
            electrode_layout.permutation,
            dtype=np.intp,
        )
        self.labels = tuple(
            int(label) for item in sessions for label in item.labels
        )
        self.standardize = standardize
        self.transform = transform
        self.standardize_eps = float(standardize_eps)
        self.task = reference.task
        self.class_names = reference.class_names
        self.num_channels = len(reference_channels)
        self.num_times = len(reference_times)
        self.sampling_rate_hz = reference_sfreq
        self.tmin = float(reference_times[0])
        self.tmax = float(reference_times[-1])
        self.n_electrode_groups = electrode_layout.n_electrode_groups
        self.original_channel_names = electrode_layout.original_channel_names
        self.channel_names = electrode_layout.reordered_channel_names
        self.electrode_groups = electrode_layout.group_channel_names
        self.electrode_group_sizes = electrode_layout.group_sizes
        self.electrode_group_centroids = electrode_layout.group_centroids
        self.channel_permutation = electrode_layout.permutation

    def __len__(self) -> int:
        """Return the total number of trials across all task sessions."""
        return self._cumulative_sizes[-1]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Load, transform and return one EEG trial and its label."""
        try:
            normalized_index = operator.index(index)
        except TypeError as error:
            raise TypeError("Dataset index must be an integer") from error

        if normalized_index < 0:
            normalized_index += len(self)
        if not 0 <= normalized_index < len(self):
            raise IndexError(
                f"Dataset index {index} is outside [0, {len(self) - 1}]"
            )
        #全局索引映射到 session
        session_index = bisect_right(
            self._cumulative_sizes, normalized_index
        )
        session_start = (
            0
            if session_index == 0
            else self._cumulative_sizes[session_index - 1]
        )
        local_index = normalized_index - session_start
        task_data = self._task_sessions[session_index]

        eeg_array = task_data.epochs.get_data(
            item=slice(local_index, local_index + 1),
            copy=True,
            verbose="ERROR",
        )[0]
        if task_data.channel_indices is not None:
            eeg_array = np.take(
                eeg_array,
                task_data.channel_indices,
                axis=0,
            )
        eeg_array = np.take(
            eeg_array,
            self._channel_permutation,
            axis=0,
        )
        if task_data.time_slice is not None:
            start, stop, step = task_data.time_slice
            eeg_array = eeg_array[:, start:stop:step]

        #将 MNE 的 float64 数据转换成连续的 torch.float32 Tensor，减少一半内存，并适配神经网络训练。
        eeg = torch.from_numpy(
            np.ascontiguousarray(eeg_array, dtype=np.float32)
        )

        if self.transform is not None:
            eeg = self.transform(eeg)
            if not isinstance(eeg, Tensor):
                raise TypeError("transform must return a torch.Tensor")

        eeg = eeg.to(dtype=torch.float32)
        if eeg.ndim != 2:
            raise ValueError(
                f"EEG transform must return [channels, time], got "
                f"shape {tuple(eeg.shape)}"
            )
        if eeg.shape[0] == 0 or eeg.shape[1] == 0:
            raise ValueError("EEG transform returned an empty dimension")

        if self.standardize:
            channel_mean = eeg.mean(dim=-1, keepdim=True)
            channel_std = eeg.std(dim=-1, keepdim=True, correction=0)
            eeg = (eeg - channel_mean) / channel_std.clamp_min(
                self.standardize_eps
            )

        if not torch.isfinite(eeg).all().item():
            raise ValueError(
                f"EEG contains NaN or infinite values at dataset index "
                f"{normalized_index}"
            )

        label = torch.tensor(
            self.labels[normalized_index], dtype=torch.long
        )
        return eeg.contiguous(), label


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset_root = Path("data/ds003626")
    task_sessions = load_task_sessions(
        dataset_root,
        task="imagine",
        subjects=(1, 2),
        sessions=(1, 2),
        preload=False,
    )

    dataset = EEGTaskDataset(
        task_sessions,
        n_electrode_groups=8,
    )
    eeg, label = dataset[0]
    if eeg.shape != (128, 1153) or label.ndim != 0:
        raise RuntimeError(
            f"Unexpected sample shapes: eeg={eeg.shape}, label={label.shape}"
        )
    if dataset.electrode_group_sizes != (16,) * 8:
        raise RuntimeError(
            f"Unexpected electrode group sizes: "
            f"{dataset.electrode_group_sizes}"
        )
    if set(dataset.channel_names) != set(dataset.original_channel_names):
        raise RuntimeError("Spatial channel order lost or duplicated channels")

    original_eeg = task_sessions[0].epochs.get_data(
        item=slice(0, 1),
        copy=True,
        verbose="ERROR",
    )[0]
    expected_eeg = torch.from_numpy(
        np.ascontiguousarray(
            np.take(
                original_eeg,
                dataset.channel_permutation,
                axis=0,
            ),
            dtype=np.float32,
        )
    )
    if not torch.equal(eeg, expected_eeg):
        raise RuntimeError("Dataset sample does not follow channel_permutation")

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )
    batch_eeg, batch_labels = next(iter(loader))
    if batch_eeg.shape != (8, 128, 1153):
        raise RuntimeError(f"Unexpected EEG batch shape: {batch_eeg.shape}")
    if batch_labels.shape != (8,):
        raise RuntimeError(
            f"Unexpected label batch shape: {batch_labels.shape}"
        )

    def keep_first_second(item: Tensor) -> Tensor:
        return item[:, :256]

    standardized_dataset = EEGTaskDataset(
        task_sessions,
        standardize=True,
        n_electrode_groups=8,
        transform=keep_first_second,
    )
    standardized_eeg, _ = standardized_dataset[0]
    channel_means = standardized_eeg.mean(dim=-1)
    channel_stds = standardized_eeg.std(dim=-1, correction=0)
    if not torch.allclose(
        channel_means, torch.zeros_like(channel_means), atol=1e-5
    ):
        raise RuntimeError("Standardized channel means are not close to zero")
    if not torch.allclose(
        channel_stds, torch.ones_like(channel_stds), atol=1e-4
    ):
        raise RuntimeError("Standardized channel stds are not close to one")

    print("EEGTaskDataset read test passed")
    print(
        f"task={dataset.task}, recordings={len(task_sessions)}, "
        f"trials={len(dataset)}, classes={dataset.class_names}"
    )
    print(
        f"electrode_groups={dataset.n_electrode_groups}, "
        f"group_sizes={dataset.electrode_group_sizes}"
    )
    print(f"reordered_channels={dataset.channel_names}")
    print(
        f"sample: eeg={tuple(eeg.shape)} {eeg.dtype}, "
        f"label={tuple(label.shape)} {label.dtype}"
    )
    print(
        f"batch: eeg={tuple(batch_eeg.shape)}, "
        f"labels={tuple(batch_labels.shape)}"
    )
    print(
        "transform + standardize: "
        f"eeg={tuple(standardized_eeg.shape)}, "
        f"max_abs_mean={channel_means.abs().max().item():.3e}, "
        f"max_abs_std_error={(channel_stds - 1).abs().max().item():.3e}"
    )
