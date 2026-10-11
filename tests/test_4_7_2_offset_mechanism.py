from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "experiments"
    / "4.7.2"
    / "scripts"
    / "offset_mechanism.py"
)
SPEC = importlib.util.spec_from_file_location("offset_mechanism", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DummyWarp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(1, 3, batch_first=True)
        self.offset_head = nn.Linear(3, 1)
        self.gate_head = nn.Linear(3, 1)
        self.padding_mode = "border"
        self.align_corners = True

    def forward(self, input):
        batch, maps, _, times = input.shape
        seq = input[:, :, 0].reshape(batch * maps, times, 1)
        hidden, _ = self.lstm(seq)
        offset = torch.sigmoid(self.gate_head(hidden)).squeeze(-1) * self.offset_head(hidden).squeeze(-1)
        signal = input.reshape(batch * maps, 1, 1, times)
        base = torch.arange(times, dtype=input.dtype)[None]
        normalized = 2 * (base + offset) / (times - 1) - 1
        grid = torch.stack((normalized.unsqueeze(1), torch.zeros_like(normalized).unsqueeze(1)), dim=-1)
        return F.grid_sample(signal, grid, mode="bilinear", padding_mode="border", align_corners=True).reshape_as(input)


def main() -> None:
    torch.manual_seed(7)
    base = DummyWarp().eval()
    x = torch.randn(2, 4, 1, 23)
    controller = MODULE.OffsetInterventionWarp(base).eval()
    controller.set_intervention("learned")
    torch.testing.assert_close(controller(x), base(x), atol=1e-6, rtol=1e-6)
    controller.set_intervention("zero")
    torch.testing.assert_close(controller(x), x, atol=2e-6, rtol=1e-6)
    offsets = torch.randn(5, 4, 65).cumsum(dim=-1) * 0.01 + 0.2
    randomized = MODULE.phase_randomized_matched(offsets, seed=11)
    torch.testing.assert_close(torch.fft.rfft(randomized, dim=-1).abs(), torch.fft.rfft(offsets, dim=-1).abs(), atol=2e-5, rtol=2e-4)
    eeg = torch.tensor([[[0.0, 1.0, 2.0, 3.0, 4.0]]])
    torch.testing.assert_close(MODULE.fractional_delay(eeg, 1.0), torch.tensor([[[0.0, 0.0, 1.0, 2.0, 3.0]]]))
    generator = torch.Generator().manual_seed(42)
    permutation = MODULE.derangement(100, generator)
    assert not torch.any(permutation == torch.arange(100))
    print("4.7.2 offset mechanism CPU tests passed")


if __name__ == "__main__":
    main()
