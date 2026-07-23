"""Parity tests for the fused F0 filter chain (single host transfer)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rcwx.pipeline.inference import (
    apply_f0_filter_chain,
    fill_short_unvoiced_gaps,
    limit_f0_slew,
    lowpass_f0,
    smooth_f0_spikes,
    suppress_octave_flips,
)


def _legacy_chain(
    f0: torch.Tensor,
    *,
    f0_lowpass_cutoff_hz: float,
    enable_octave_flip_suppress: bool,
    enable_f0_slew_limit: bool,
    f0_slew_max_step_st: float,
    f0_hole_fill_ms: float,
) -> torch.Tensor:
    """Pre-fusion staged composition (for parity)."""
    out = fill_short_unvoiced_gaps(
        f0, max_gap_frames=int(round(f0_hole_fill_ms / 10.0))
    )
    out = smooth_f0_spikes(out, window=3)
    out = lowpass_f0(out, cutoff_hz=f0_lowpass_cutoff_hz, sample_rate=100.0)
    if enable_octave_flip_suppress:
        out = suppress_octave_flips(out)
    if enable_f0_slew_limit:
        out = limit_f0_slew(out, max_step_st=f0_slew_max_step_st)
    return out


def _sample_tracks() -> list[torch.Tensor]:
    rng = np.random.default_rng(0)
    tracks: list[torch.Tensor] = []

    # Steady voiced contour with mild vibrato (streaming-like length)
    t = np.arange(80, dtype=np.float32)
    steady = 200.0 + 15.0 * np.sin(2 * np.pi * t / 20.0)
    tracks.append(torch.from_numpy(steady.astype(np.float32)).unsqueeze(0))

    # Short unvoiced holes inside a vowel
    holed = steady.copy()
    holed[20] = 0.0
    holed[21] = 0.0
    holed[45] = 0.0
    tracks.append(torch.from_numpy(holed.astype(np.float32)).unsqueeze(0))

    # Octave flip glitch
    oct_glitch = steady.copy()
    oct_glitch[30] = steady[30] * 2.0
    tracks.append(torch.from_numpy(oct_glitch.astype(np.float32)).unsqueeze(0))

    # Large slew jump
    slew = steady.copy()
    slew[50] = 420.0
    tracks.append(torch.from_numpy(slew.astype(np.float32)).unsqueeze(0))

    # Leading/trailing unvoiced + interior speech
    mixed = np.zeros(100, dtype=np.float32)
    mixed[15:70] = 180.0 + 10.0 * rng.standard_normal(55).astype(np.float32)
    mixed[mixed < 50] = 0.0
    tracks.append(torch.from_numpy(mixed).unsqueeze(0))

    # Fully unvoiced / fully voiced short
    tracks.append(torch.zeros(1, 40))
    tracks.append(torch.full((1, 40), 220.0))

    return tracks


def test_fused_chain_matches_legacy_composition() -> None:
    kwargs = dict(
        f0_lowpass_cutoff_hz=16.0,
        enable_octave_flip_suppress=True,
        enable_f0_slew_limit=True,
        f0_slew_max_step_st=3.6,
        f0_hole_fill_ms=30.0,
    )
    for i, f0 in enumerate(_sample_tracks()):
        fused = apply_f0_filter_chain(f0, **kwargs)
        legacy = _legacy_chain(f0, **kwargs)
        assert fused.shape == legacy.shape
        assert fused.dtype == legacy.dtype
        max_err = float((fused - legacy).abs().max().item())
        assert max_err < 1e-4, f"track {i}: max abs err {max_err}"


def test_fused_chain_respects_feature_flags() -> None:
    row = [220.0, 224.0, 446.0, 228.0, 232.0] + [0.0] * 20
    f0 = torch.tensor([row], dtype=torch.float32)

    no_oct = apply_f0_filter_chain(
        f0,
        f0_lowpass_cutoff_hz=16.0,
        enable_octave_flip_suppress=False,
        enable_f0_slew_limit=True,
        f0_slew_max_step_st=3.6,
        f0_hole_fill_ms=30.0,
    )
    with_oct = apply_f0_filter_chain(
        f0,
        f0_lowpass_cutoff_hz=16.0,
        enable_octave_flip_suppress=True,
        enable_f0_slew_limit=False,
        f0_slew_max_step_st=3.6,
        f0_hole_fill_ms=30.0,
    )
    # With octave suppress the glitch frame should move closer to ~223Hz
    assert float(with_oct[0, 2].item()) < 300.0
    # Without octave suppress but with slew, still not free to keep 446
    assert float(no_oct[0, 2].item()) != 446.0


if __name__ == "__main__":
    test_fused_chain_matches_legacy_composition()
    test_fused_chain_respects_feature_flags()
    print("all ok")
