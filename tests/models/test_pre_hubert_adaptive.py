from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rcwx.pipeline.inference import (
    compute_adaptive_pre_hubert_shift,
    compute_pre_hubert_shift,
    estimate_median_f0_autocorr,
)


def _sine(f0_hz: float, duration_sec: float = 0.42, sample_rate: int = 16000) -> np.ndarray:
    t = np.arange(int(duration_sec * sample_rate), dtype=np.float32) / float(sample_rate)
    sig = 0.22 * np.sin(2.0 * np.pi * f0_hz * t)
    sig += 0.04 * np.sin(2.0 * np.pi * 2.0 * f0_hz * t)
    sig += 0.01 * np.sin(2.0 * np.pi * 3.1 * f0_hz * t)
    return sig.astype(np.float32)


def test_estimate_median_f0_autocorr_tracks_simple_sine() -> None:
    audio = _sine(120.0)
    median_hz, voiced_ratio, periodicity = estimate_median_f0_autocorr(audio)

    assert median_hz is not None
    assert abs(median_hz - 120.0) < 10.0, f"Expected ~120Hz, got {median_hz:.2f}Hz"
    assert voiced_ratio > 0.45, f"Expected voiced_ratio > 0.45, got {voiced_ratio:.2f}"
    assert periodicity > 0.45, f"Expected periodicity > 0.45, got {periodicity:.2f}"


def test_estimate_median_f0_autocorr_silence_returns_none() -> None:
    audio = np.zeros(4096, dtype=np.float32)
    median_hz, voiced_ratio, periodicity = estimate_median_f0_autocorr(audio)

    assert median_hz is None
    assert voiced_ratio == 0.0
    assert periodicity == 0.0


def test_adaptive_shift_raises_low_register_more_than_manual() -> None:
    base = compute_pre_hubert_shift(pitch_shift=12, ratio=0.08)
    adaptive = compute_adaptive_pre_hubert_shift(
        pitch_shift=12,
        ratio=0.08,
        source_median_f0_hz=108.0,
        moe_boost=0.9,
        voiced_ratio=0.90,
        periodicity=0.86,
    )

    assert adaptive > base + 1.0, f"Expected adaptive>{base + 1.0:.2f}, got {adaptive:.2f}"
    assert adaptive <= 12.0 + 1e-6


def test_adaptive_shift_preserves_manual_for_high_register() -> None:
    base = compute_pre_hubert_shift(pitch_shift=12, ratio=0.08)
    adaptive = compute_adaptive_pre_hubert_shift(
        pitch_shift=12,
        ratio=0.08,
        source_median_f0_hz=260.0,
        moe_boost=0.9,
        voiced_ratio=0.90,
        periodicity=0.86,
    )

    assert adaptive <= base + 0.25, f"High register should stay near base. base={base:.2f}, got={adaptive:.2f}"


def test_adaptive_shift_low_confidence_stays_near_manual() -> None:
    base = compute_pre_hubert_shift(pitch_shift=12, ratio=0.08)
    adaptive = compute_adaptive_pre_hubert_shift(
        pitch_shift=12,
        ratio=0.08,
        source_median_f0_hz=100.0,
        moe_boost=0.9,
        voiced_ratio=0.18,
        periodicity=0.33,
    )

    assert abs(adaptive - base) < 0.35, f"Low confidence should not over-adapt. base={base:.2f}, got={adaptive:.2f}"


def test_adaptive_shift_disabled_for_downward_pitch() -> None:
    base = compute_pre_hubert_shift(pitch_shift=-8, ratio=0.4)
    adaptive = compute_adaptive_pre_hubert_shift(
        pitch_shift=-8,
        ratio=0.4,
        source_median_f0_hz=110.0,
        moe_boost=1.0,
        voiced_ratio=1.0,
        periodicity=1.0,
    )

    assert adaptive == base