"""TrebleBoost chunk-boundary continuity tests.

Regression: the streaming path used to reset the IIR filter state on every
chunk (``zi = self.zi * audio[0]`` for every chunk instead of only the first
one after reset), which threw away the resonator's phase memory and injected
a click at every chunk boundary on sustained tones. See:

    <postprocess.py>::TrebleBoost.process

Test strategy: feed a sustained sine wave chunk-by-chunk and compare against
a batch reference. If filter state carries across chunks correctly, streamed
output is bit-exact against batch. If the bug returns, the boundary delta
spikes to many times the natural signal delta (5-6x in the reproduction we
ran) and the output range overshoots the input's ±0.3 by ~34%.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rcwx.audio.postprocess import PostprocessConfig, TrebleBoost


def _stream(audio: np.ndarray, cfg: PostprocessConfig, sr: int, chunk: int) -> np.ndarray:
    tb = TrebleBoost(sr, cfg)
    out = np.empty_like(audio)
    for start in range(0, len(audio), chunk):
        end = min(start + chunk, len(audio))
        out[start:end] = tb.process(audio[start:end])
    return out


def _batch(audio: np.ndarray, cfg: PostprocessConfig, sr: int) -> np.ndarray:
    return TrebleBoost(sr, cfg).process(audio)


def _worst_case_boundary_sine(sr: int, freq: float, chunk_samples: int, n: int) -> np.ndarray:
    """Sine whose first chunk boundary lands on a positive peak (worst case
    for the state-reset bug: audio[0] at the boundary is at maximum amplitude,
    so ``zi_template * audio[0]`` jumps the filter far from its real state)."""
    t = np.arange(n, dtype=np.float32) / sr
    phase = np.pi / 2 - 2 * np.pi * freq * (chunk_samples / sr)
    return 0.3 * np.sin(2 * np.pi * freq * t + phase).astype(np.float32)


def test_streaming_matches_batch_on_sustained_sine() -> None:
    """Streamed output should match batch bit-exactly for sustained input."""
    sr = 48000
    chunk_samples = int(sr * 0.15)
    n = sr  # 1 second
    cfg = PostprocessConfig(enabled=True, treble_boost_db=4.0, treble_cutoff_hz=2800.0)
    sine = _worst_case_boundary_sine(sr, 440.0, chunk_samples, n)

    batch = _batch(sine, cfg, sr)
    streamed = _stream(sine, cfg, sr, chunk_samples)

    max_diff = float(np.max(np.abs(streamed - batch)))
    assert max_diff < 1e-6, (
        f"Streaming diverges from batch (MAX |diff| = {max_diff:.6f}). "
        "TrebleBoost is likely resetting filter state per chunk again."
    )


def test_boundary_delta_matches_signal_delta() -> None:
    """Sample-to-sample delta at chunk boundaries should look like any other
    sample delta.  A large boundary delta relative to the signal's p99 delta
    is the click signature."""
    sr = 48000
    chunk_samples = int(sr * 0.15)
    n = sr
    cfg = PostprocessConfig(enabled=True, treble_boost_db=4.0, treble_cutoff_hz=2800.0)
    sine = _worst_case_boundary_sine(sr, 440.0, chunk_samples, n)

    streamed = _stream(sine, cfg, sr, chunk_samples)
    delta = np.abs(np.diff(streamed))
    p99 = float(np.percentile(delta, 99))

    boundaries = list(range(chunk_samples, n, chunk_samples))
    boundary_max = float(max(delta[b - 1] for b in boundaries))

    # Sine wave neighbouring-sample delta is deterministic; boundaries must
    # not exceed the signal-wide p99 by any meaningful margin.
    assert boundary_max <= p99 * 1.1, (
        f"Chunk boundaries have larger delta than the signal p99 "
        f"({boundary_max:.6f} vs p99 {p99:.6f}) — chunk-rate click regression."
    )


def test_no_amplitude_overshoot_at_boundaries() -> None:
    """Streamed output must not exceed the batch output's peak range.

    The bug produced +0.4032 peaks on a signal whose batch output peaked at
    +0.3005 — a 34% overshoot every chunk.
    """
    sr = 48000
    chunk_samples = int(sr * 0.15)
    n = sr
    cfg = PostprocessConfig(enabled=True, treble_boost_db=4.0, treble_cutoff_hz=2800.0)
    sine = _worst_case_boundary_sine(sr, 440.0, chunk_samples, n)

    batch = _batch(sine, cfg, sr)
    streamed = _stream(sine, cfg, sr, chunk_samples)

    batch_peak = float(np.max(np.abs(batch)))
    stream_peak = float(np.max(np.abs(streamed)))
    assert stream_peak <= batch_peak * 1.001, (
        f"Streamed peak {stream_peak:.4f} exceeds batch peak {batch_peak:.4f} "
        "— boundary overshoot regression."
    )


def test_reset_rearms_warm_start() -> None:
    """After reset(), the next chunk must be warm-started against its own
    DC level (so a fresh stream doesn't leak the previous stream's tail
    state), and the streamed=batch property must hold again."""
    sr = 48000
    chunk_samples = int(sr * 0.15)
    n = sr
    cfg = PostprocessConfig(enabled=True, treble_boost_db=4.0, treble_cutoff_hz=2800.0)
    sine = _worst_case_boundary_sine(sr, 440.0, chunk_samples, n)

    tb = TrebleBoost(sr, cfg)
    # Run one full stream, then reset and run again — second run must match
    # a fresh batch call.
    for start in range(0, n, chunk_samples):
        tb.process(sine[start : start + chunk_samples])
    tb.reset()

    out = np.empty_like(sine)
    for start in range(0, n, chunk_samples):
        end = start + chunk_samples
        out[start:end] = tb.process(sine[start:end])

    batch = _batch(sine, cfg, sr)
    max_diff = float(np.max(np.abs(out - batch)))
    assert max_diff < 1e-6, (
        f"After reset(), streamed diverges from batch (MAX |diff| = {max_diff:.6f})"
    )


if __name__ == "__main__":
    test_streaming_matches_batch_on_sustained_sine()
    print("OK: streaming == batch")
    test_boundary_delta_matches_signal_delta()
    print("OK: boundary delta ~= signal p99")
    test_no_amplitude_overshoot_at_boundaries()
    print("OK: no boundary overshoot")
    test_reset_rearms_warm_start()
    print("OK: reset re-arms warm start")
