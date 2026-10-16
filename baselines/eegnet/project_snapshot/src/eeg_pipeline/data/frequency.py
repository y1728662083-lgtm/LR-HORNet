"""Training-only cached frequency-band targets for EEG contrastive learning."""

from __future__ import annotations

import math
import operator
from collections.abc import Sequence

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset


class CachedFilterBankDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Add deterministic, patch-aligned EEG band-power targets to a dataset.

    The wrapped dataset must return ``(eeg, label)`` with EEG shaped ``(C, T)``.
    A zero-phase Butterworth filter bank is evaluated only on the first access
    to each sample. The resulting log-power tensor is kept in CPU memory and
    reused by every later epoch. This wrapper is intended for ``num_workers=0``.

    Returned samples are ``(eeg, label, frequency)`` where ``frequency`` has
    shape ``(C, K, P)`` and
    ``P = floor((T - patch_size) / patch_stride) + 1``.
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        sampling_rate_hz: float,
        bands_hz: Sequence[Sequence[float]],
        patch_size: int,
        patch_stride: int,
        filter_order: int = 4,
        cache_dtype: str = "float16",
    ) -> None:
        super().__init__()
        if len(dataset) < 1:
            raise ValueError("frequency-cache dataset must not be empty")

        sampling_rate_hz = float(sampling_rate_hz)
        if not math.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0.0:
            raise ValueError("sampling_rate_hz must be finite and positive")
        patch_size = operator.index(patch_size)
        patch_stride = operator.index(patch_stride)
        filter_order = operator.index(filter_order)
        if patch_size < 1 or patch_stride < 1:
            raise ValueError("patch_size and patch_stride must be positive")
        if patch_stride > patch_size:
            raise ValueError("patch_stride must not exceed patch_size")
        if filter_order < 1:
            raise ValueError("filter_order must be positive")

        parsed_bands: list[tuple[float, float]] = []
        nyquist = sampling_rate_hz / 2.0
        for band_index, band in enumerate(bands_hz):
            if len(band) != 2:
                raise ValueError(
                    f"bands_hz[{band_index}] must contain [low, high]"
                )
            low_hz, high_hz = float(band[0]), float(band[1])
            if not (
                math.isfinite(low_hz)
                and math.isfinite(high_hz)
                and 0.0 < low_hz < high_hz < nyquist
            ):
                raise ValueError(
                    f"invalid frequency band {band!r}; expected "
                    f"0 < low < high < {nyquist:g} Hz"
                )
            parsed_bands.append((low_hz, high_hz))
        if not parsed_bands:
            raise ValueError("bands_hz must contain at least one band")

        dtype_by_name = {
            "float16": torch.float16,
            "float32": torch.float32,
        }
        try:
            parsed_cache_dtype = dtype_by_name[cache_dtype]
        except KeyError:
            raise ValueError(
                "cache_dtype must be 'float16' or 'float32'"
            ) from None

        self.dataset = dataset
        self.sampling_rate_hz = sampling_rate_hz
        self.bands_hz = tuple(parsed_bands)
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.filter_order = filter_order
        self.cache_dtype = parsed_cache_dtype
        self._filters = tuple(
            butter(
                filter_order,
                (low_hz, high_hz),
                btype="bandpass",
                fs=sampling_rate_hz,
                output="sos",
            )
            for low_hz, high_hz in self.bands_hz
        )
        self._frequency_cache: list[Tensor | None] = [None] * len(dataset)
        self.cache_misses = 0

    def __len__(self) -> int:
        return len(self.dataset)

    def _compute_frequency_target(self, eeg: Tensor) -> Tensor:
        if eeg.ndim != 2:
            raise ValueError(
                f"expected EEG shaped (C,T), got {tuple(eeg.shape)}"
            )
        n_chans, n_times = eeg.shape
        if n_times < self.patch_size:
            raise ValueError(
                f"EEG time length {n_times} is shorter than "
                f"patch_size={self.patch_size}"
            )

        eeg_array = np.asarray(
            eeg.detach().to(device="cpu", dtype=torch.float32).contiguous(),
            dtype=np.float32,
        )
        n_patches = (
            (n_times - self.patch_size) // self.patch_stride + 1
        )
        frequency = torch.empty(
            n_chans,
            len(self._filters),
            n_patches,
            dtype=torch.float32,
        )

        for band_index, second_order_sections in enumerate(self._filters):
            filtered_array = sosfiltfilt(
                second_order_sections,
                eeg_array,
                axis=-1,
            )
            filtered = torch.from_numpy(
                np.ascontiguousarray(filtered_array, dtype=np.float32)
            )
            mean_power = F.avg_pool1d(
                filtered.square().unsqueeze(1),
                kernel_size=self.patch_size,
                stride=self.patch_stride,
            ).squeeze(1)
            frequency[:, band_index, :] = torch.log1p(mean_power)

        if not torch.isfinite(frequency).all().item():
            raise ValueError("frequency filter bank produced non-finite values")
        return frequency.to(dtype=self.cache_dtype).contiguous()

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        normalized_index = operator.index(index)
        if normalized_index < 0:
            normalized_index += len(self)
        if not 0 <= normalized_index < len(self):
            raise IndexError(normalized_index)

        sample = self.dataset[normalized_index]
        if not isinstance(sample, (tuple, list)) or len(sample) != 2:
            raise TypeError("wrapped dataset must return (eeg, label)")
        eeg, label = sample
        cached = self._frequency_cache[normalized_index]
        if cached is None:
            cached = self._compute_frequency_target(eeg)
            self._frequency_cache[normalized_index] = cached
            self.cache_misses += 1
        return eeg, label, cached


if __name__ == "__main__":
    from torch.utils.data import TensorDataset

    generator = torch.Generator().manual_seed(42)
    eeg = torch.randn(3, 4, 129, generator=generator)
    labels = torch.tensor([0, 1, 0])
    cached_dataset = CachedFilterBankDataset(
        TensorDataset(eeg, labels),
        sampling_rate_hz=256.0,
        bands_hz=((4.0, 8.0), (8.0, 13.0)),
        patch_size=32,
        patch_stride=4,
    )
    first_eeg, first_label, first_frequency = cached_dataset[0]
    _, _, repeated_frequency = cached_dataset[0]
    if first_eeg.shape != (4, 129) or first_label.ndim != 0:
        raise RuntimeError("unexpected wrapped sample shape")
    if first_frequency.shape != (4, 2, 25):
        raise RuntimeError(
            f"unexpected frequency shape: {first_frequency.shape}"
        )
    if cached_dataset.cache_misses != 1:
        raise RuntimeError("frequency target was recomputed")
    if first_frequency.data_ptr() != repeated_frequency.data_ptr():
        raise RuntimeError("cached tensor was not reused")
    print("CachedFilterBankDataset self-test passed")
