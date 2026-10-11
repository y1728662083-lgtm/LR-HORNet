"""In-process RAM cache for already split and standardized EEG trials.

The source datasets use lazy MNE reads and per-trial standardization.  Repeating
that work in every epoch leaves the GPU idle.  This wrapper materializes each
split exactly once without changing trial order, values, labels, or split
membership.  It intentionally uses RAM rather than disk because the target
server has ample memory but limited temporary disk space.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset


class CachedTrialDataset(Dataset):
    """A tensor-backed, order-preserving snapshot of an EEG dataset."""

    def __init__(self, eeg: Tensor, labels: Tensor, *, source_type: str) -> None:
        if eeg.ndim < 2:
            raise ValueError(f"Expected batched EEG tensor, got shape={tuple(eeg.shape)}")
        if labels.ndim != 1 or len(eeg) != len(labels):
            raise ValueError("Cached EEG and label counts do not match")
        self.eeg = eeg.contiguous()
        self.labels = labels.to(dtype=torch.long).contiguous()
        self.source_type = source_type

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.eeg[index], self.labels[index]


def _as_cpu_tensor(value: Any) -> Tensor:
    if isinstance(value, Tensor):
        return value.detach().cpu()
    return torch.as_tensor(value)


def materialize_dataset(
    dataset: Dataset,
    *,
    split_name: str,
    progress_every: int = 256,
) -> CachedTrialDataset:
    """Read and standardize every trial once, retaining float32 values in RAM."""

    count = len(dataset)
    if count <= 0:
        raise RuntimeError(f"Cannot cache empty {split_name} dataset")
    started = time.perf_counter()
    first_eeg, first_label = dataset[0]
    first_eeg = _as_cpu_tensor(first_eeg).to(dtype=torch.float32).contiguous()
    eeg = torch.empty((count, *first_eeg.shape), dtype=torch.float32)
    labels = torch.empty(count, dtype=torch.long)
    eeg[0].copy_(first_eeg)
    labels[0] = int(_as_cpu_tensor(first_label).item())

    for index in range(1, count):
        trial, label = dataset[index]
        trial = _as_cpu_tensor(trial).to(dtype=torch.float32).contiguous()
        if tuple(trial.shape) != tuple(first_eeg.shape):
            raise ValueError(
                f"Variable EEG shape in {split_name}: index=0 {tuple(first_eeg.shape)} "
                f"index={index} {tuple(trial.shape)}"
            )
        eeg[index].copy_(trial)
        labels[index] = int(_as_cpu_tensor(label).item())
        if progress_every > 0 and (index + 1) % progress_every == 0:
            print(
                f"CACHE split={split_name} loaded={index + 1}/{count}",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    size_mib = (eeg.numel() * eeg.element_size() + labels.numel() * labels.element_size()) / 2**20
    print(
        f"CACHE READY split={split_name} trials={count} shape={tuple(eeg.shape)} "
        f"dtype={eeg.dtype} ram_mib={size_mib:.1f} seconds={elapsed:.1f}",
        flush=True,
    )
    return CachedTrialDataset(eeg, labels, source_type=type(dataset).__name__)


def materialize_splits(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
) -> tuple[CachedTrialDataset, CachedTrialDataset, CachedTrialDataset]:
    return (
        materialize_dataset(train_dataset, split_name="train"),
        materialize_dataset(val_dataset, split_name="validation"),
        materialize_dataset(test_dataset, split_name="test"),
    )

