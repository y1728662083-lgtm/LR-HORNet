"""DataLoader construction for EEG training, validation and testing."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset


def build_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    *,
    batch_size: int = 32,
    seed: int = 42,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build reproducible train, validation and test data loaders."""
    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from torch.utils.data import RandomSampler, SequentialSampler

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from eeg_pipeline.data.splits import (
        build_adapted_fold_datasets,
        build_loso_subject_folds,
    )

    dataset_root = Path("data/ds003626")
    fold = build_loso_subject_folds(tuple(range(1, 11)))[0]
    train_dataset, val_dataset, test_dataset = build_adapted_fold_datasets(
        dataset_root,
        task="imagine",
        fold=fold,
        seed=42,
    )
    train_loader, val_loader, test_loader = build_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=32,
        seed=42,
    )

    if not isinstance(train_loader.sampler, RandomSampler):
        raise RuntimeError("Train DataLoader must shuffle trials")
    if not isinstance(val_loader.sampler, SequentialSampler):
        raise RuntimeError("Validation DataLoader must preserve trial order")
    if not isinstance(test_loader.sampler, SequentialSampler):
        raise RuntimeError("Test DataLoader must preserve trial order")

    train_eeg, train_labels = next(iter(train_loader))
    val_eeg, val_labels = next(iter(val_loader))
    test_eeg, test_labels = next(iter(test_loader))
    for name, loader, eeg, labels in (
        ("train", train_loader, train_eeg, train_labels),
        ("val", val_loader, val_eeg, val_labels),
        ("test", test_loader, test_eeg, test_labels),
    ):
        expected_size = min(loader.batch_size, len(loader.dataset))
        if eeg.shape != (expected_size, 128, 1153):
            raise RuntimeError(f"Unexpected {name} EEG shape: {eeg.shape}")
        if labels.shape != (expected_size,):
            raise RuntimeError(
                f"Unexpected {name} label shape: {labels.shape}"
            )

    repeated_train_loader, _, _ = build_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=32,
        seed=42,
    )
    repeated_eeg, repeated_labels = next(iter(repeated_train_loader))
    if not torch.equal(train_eeg, repeated_eeg) or not torch.equal(
        train_labels, repeated_labels
    ):
        raise RuntimeError("Train shuffle is not reproducible for the same seed")

    for loader, dataset in (
        (train_loader, train_dataset),
        (val_loader, val_dataset),
        (test_loader, test_dataset),
    ):
        expected_batches = (len(dataset) + loader.batch_size - 1) // loader.batch_size
        if len(loader) != expected_batches:
            raise RuntimeError("Unexpected number of DataLoader batches")

    print("DataLoader test passed")
    print(
        f"datasets: train={len(train_dataset)}, val={len(val_dataset)}, "
        f"test={len(test_dataset)}"
    )
    print(
        f"batches: train={len(train_loader)}, val={len(val_loader)}, "
        f"test={len(test_loader)}"
    )
    print(
        f"batch shapes: eeg={tuple(train_eeg.shape)}, "
        f"labels={tuple(train_labels.shape)}"
    )
    print("train shuffle reproducibility passed")
