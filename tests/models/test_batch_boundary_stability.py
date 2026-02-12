from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rcwx.pipeline.inference import apply_output_edge_fade, stabilize_f0_boundaries


def test_stabilize_f0_boundaries_reduces_edge_jitter() -> None:
    base = torch.full((1, 40), 200.0, dtype=torch.float32)
    base[0, 0:6] = torch.tensor([330.0, 90.0, 290.0, 120.0, 250.0, 170.0])
    base[0, -6:] = torch.tensor([175.0, 255.0, 130.0, 300.0, 95.0, 320.0])

    out = stabilize_f0_boundaries(base, edge_frames=8)

    target = torch.median(base[0, 8:28]).item()
    before_head = torch.mean(torch.abs(base[0, 0:6] - target)).item()
    after_head = torch.mean(torch.abs(out[0, 0:6] - target)).item()
    before_tail = torch.mean(torch.abs(base[0, -6:] - target)).item()
    after_tail = torch.mean(torch.abs(out[0, -6:] - target)).item()

    assert after_head < before_head * 0.75, (before_head, after_head)
    assert after_tail < before_tail * 0.75, (before_tail, after_tail)

    # Keep center contour untouched.
    assert torch.max(torch.abs(out[0, 12:28] - base[0, 12:28])).item() < 1e-6


def test_stabilize_f0_boundaries_keeps_unvoiced_silence() -> None:
    f0 = torch.zeros((1, 30), dtype=torch.float32)
    f0[0, 12:24] = 210.0

    out = stabilize_f0_boundaries(f0, edge_frames=8)

    assert torch.all(out[0, :12] == 0.0)
    assert torch.all(out[0, 24:] == 0.0)


def test_apply_output_edge_fade() -> None:
    x = np.ones(1000, dtype=np.float32)
    y = apply_output_edge_fade(x, sample_rate=1000, fade_ms=10.0)

    assert abs(float(y[0])) < 1e-7
    assert abs(float(y[-1])) < 1e-7
    assert float(np.min(y[20:-20])) > 0.99


def test_apply_output_edge_fade_short_signal_noop() -> None:
    x = np.ones(12, dtype=np.float32)
    y = apply_output_edge_fade(x, sample_rate=1000, fade_ms=10.0)
    assert np.array_equal(x, y)