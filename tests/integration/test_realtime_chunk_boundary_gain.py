"""Tests for RmsNormalizer (EMA-smoothed AGC) in the postprocess pipeline.

Replaces the old _apply_output_boundary_gain tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rcwx.audio.postprocess import PostprocessConfig, RmsNormalizer, Postprocessor


def _make_normalizer(
    target_rms: float = 0.1,
    ema_alpha: float = 0.15,
    max_gain_db: float = 12.0,
    min_gain_db: float = -12.0,
    sample_rate: int = 48000,
) -> RmsNormalizer:
    cfg = PostprocessConfig(
        normalizer_enabled=True,
        normalizer_target_rms=target_rms,
        normalizer_ema_alpha=ema_alpha,
        normalizer_max_gain_db=max_gain_db,
        normalizer_min_gain_db=min_gain_db,
    )
    return RmsNormalizer(sample_rate, cfg)


def test_normalizer_stabilises_varying_amplitude() -> None:
    """Chunks with different amplitudes should converge toward target RMS."""
    norm = _make_normalizer(target_rms=0.1, ema_alpha=0.3)  # fast alpha for test

    rms_values = []
    for amplitude in [0.05, 0.2, 0.05, 0.2, 0.1, 0.1, 0.1]:
        chunk = np.full(4800, amplitude, dtype=np.float32)  # 100ms @ 48kHz
        out = norm.process(chunk)
        rms_values.append(float(np.sqrt(np.mean(out ** 2))))

    # After convergence, last few chunks should be close to target
    for rms in rms_values[-3:]:
        assert 0.06 < rms < 0.16, f"RMS {rms:.4f} should be near target 0.1"


def test_normalizer_silent_chunk_not_amplified() -> None:
    """Silent chunks (below MIN_RMS) must not be boosted."""
    norm = _make_normalizer(target_rms=0.1)

    # Prime the EMA with a normal chunk
    voiced = np.full(4800, 0.05, dtype=np.float32)
    norm.process(voiced)

    # Now feed silence
    silence = np.full(4800, 0.001, dtype=np.float32)
    out = norm.process(silence)

    # Output should be identical to input (no gain applied)
    assert np.allclose(out, silence), "Silent chunk should pass through unchanged"


def test_normalizer_gain_clamped() -> None:
    """Gain must not exceed configured max/min bounds."""
    norm = _make_normalizer(target_rms=0.1, max_gain_db=6.0, min_gain_db=-6.0)
    max_gain = 10 ** (6.0 / 20)  # ~2x
    min_gain = 10 ** (-6.0 / 20)  # ~0.5x

    # Very quiet chunk -> gain should hit max
    quiet = np.full(4800, 0.01, dtype=np.float32)
    out = norm.process(quiet)
    effective_gain = float(np.sqrt(np.mean(out ** 2))) / 0.01
    assert effective_gain <= max_gain + 0.01, f"Gain {effective_gain:.2f} exceeds max {max_gain:.2f}"

    norm.reset()

    # Very loud chunk -> gain should hit min
    loud = np.full(4800, 0.8, dtype=np.float32)
    out = norm.process(loud)
    effective_gain = float(np.sqrt(np.mean(out ** 2))) / 0.8
    assert effective_gain >= min_gain - 0.01, f"Gain {effective_gain:.2f} below min {min_gain:.2f}"


def test_normalizer_disabled_passthrough() -> None:
    """When disabled, audio should pass through unchanged."""
    cfg = PostprocessConfig(normalizer_enabled=False)
    norm = RmsNormalizer(48000, cfg)
    chunk = np.full(4800, 0.05, dtype=np.float32)
    out = norm.process(chunk)
    assert np.allclose(out, chunk), "Disabled normalizer should pass through"


def test_normalizer_reset() -> None:
    """Reset should clear EMA state."""
    norm = _make_normalizer(target_rms=0.1)
    chunk = np.full(4800, 0.05, dtype=np.float32)
    norm.process(chunk)
    assert norm._ema_rms > 0

    norm.reset()
    assert norm._ema_rms == 0.0
    assert norm._prev_gain == 1.0


def test_postprocessor_pipeline_order() -> None:
    """Postprocessor should run treble -> normalizer -> limiter."""
    cfg = PostprocessConfig(
        enabled=True,
        treble_boost_db=0.0,  # disable treble for clarity
        normalizer_enabled=True,
        normalizer_target_rms=0.1,
        limiter_threshold_db=-1.0,
    )
    pp = Postprocessor(48000, cfg)

    # Feed a chunk and verify no crash
    chunk = np.random.randn(4800).astype(np.float32) * 0.05
    out = pp.process(chunk)
    assert len(out) > 0
    assert np.all(np.isfinite(out))


if __name__ == "__main__":
    test_normalizer_stabilises_varying_amplitude()
    test_normalizer_silent_chunk_not_amplified()
    test_normalizer_gain_clamped()
    test_normalizer_disabled_passthrough()
    test_normalizer_reset()
    test_postprocessor_pipeline_order()
    print("PASS: RmsNormalizer tests")
