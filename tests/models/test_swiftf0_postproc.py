"""Algorithmic tests for SwiftF0 harmonic/octave resolution."""

from __future__ import annotations

import numpy as np

from rcwx.models.swiftf0 import _resolve_octave_from_waveform


def _synthesize_voice_like_audio(
    f0_track: np.ndarray,
    *,
    sample_rate: int = 16000,
    hop_length: int = 160,
) -> np.ndarray:
    """Generate a harmonic voice-like waveform from an F0 track."""
    n_frames = int(f0_track.shape[0])
    n_samples = n_frames * hop_length
    audio = np.zeros(n_samples, dtype=np.float32)

    p1 = 0.0
    p2 = 0.0
    p3 = 0.0
    two_pi = 2.0 * np.pi

    for i in range(n_frames):
        f0 = float(f0_track[i])
        if f0 <= 0.0:
            continue

        s = i * hop_length
        e = s + hop_length
        t = np.arange(hop_length, dtype=np.float32) / float(sample_rate)

        seg = (
            0.72 * np.sin(p1 + two_pi * f0 * t)
            + 0.18 * np.sin(p2 + two_pi * (2.0 * f0) * t + 0.2)
            + 0.10 * np.sin(p3 + two_pi * (3.0 * f0) * t + 0.4)
        )
        audio[s:e] = seg.astype(np.float32, copy=False)

        p1 += two_pi * f0 * hop_length / float(sample_rate)
        p2 += two_pi * (2.0 * f0) * hop_length / float(sample_rate)
        p3 += two_pi * (3.0 * f0) * hop_length / float(sample_rate)

    return audio


def test_waveform_viterbi_resolves_harmonic_mistrack() -> None:
    """Viterbi resolver should recover from a forced high-harmonic mistrack."""
    n = 220
    idx = np.arange(n, dtype=np.float32)
    true_f0 = (176.0 + 14.0 * np.sin(2.0 * np.pi * idx / 68.0)).astype(np.float32)
    true_f0[92:98] = 0.0
    true_f0[151:156] = 0.0

    audio = _synthesize_voice_like_audio(true_f0)

    raw_f0 = true_f0.copy()
    bad = np.zeros(n, dtype=bool)
    bad[40:84] = True
    bad &= true_f0 > 0
    raw_f0[bad] *= 4.0  # emulate harmonic lock at 4th multiple

    voiced = raw_f0 > 0
    conf = np.full(n, 0.9, dtype=np.float32)
    fixed = _resolve_octave_from_waveform(
        raw_f0,
        voiced,
        audio,
        sample_rate=16000,
        hop_length=160,
        f0_min=50.0,
        f0_max=1100.0,
        confidence=conf,
    )

    raw_mae = float(np.median(np.abs(raw_f0[bad] - true_f0[bad])))
    fixed_mae = float(np.median(np.abs(fixed[bad] - true_f0[bad])))
    assert fixed_mae < raw_mae * 0.35, f"Expected large error drop: raw={raw_mae:.2f}, fixed={fixed_mae:.2f}"
    assert float(np.median(fixed[bad])) < 260.0, "Harmonic mistrack should not remain in very high band"


def test_waveform_viterbi_preserves_clean_track() -> None:
    """Resolver should avoid distorting an already-correct contour."""
    n = 180
    idx = np.arange(n, dtype=np.float32)
    true_f0 = (165.0 + 10.0 * np.sin(2.0 * np.pi * idx / 53.0)).astype(np.float32)
    true_f0[70:74] = 0.0

    audio = _synthesize_voice_like_audio(true_f0)
    raw_f0 = true_f0.copy()
    voiced = raw_f0 > 0
    conf = np.full(n, 0.9, dtype=np.float32)

    fixed = _resolve_octave_from_waveform(
        raw_f0,
        voiced,
        audio,
        sample_rate=16000,
        hop_length=160,
        f0_min=50.0,
        f0_max=1100.0,
        confidence=conf,
    )

    voiced_err = np.abs(fixed[voiced] - true_f0[voiced])
    assert float(np.median(voiced_err)) < 5.0, f"Unexpected median drift: {np.median(voiced_err):.2f}Hz"
    assert float(np.percentile(voiced_err, 95)) < 18.0, f"Unexpected tail drift: {np.percentile(voiced_err, 95):.2f}Hz"


def test_waveform_viterbi_keeps_unvoiced_zero() -> None:
    """Unvoiced frames must remain zero after resolution."""
    f0 = np.array([0.0, 170.0, 0.0, 680.0, 0.0, 175.0, 0.0], dtype=np.float32)
    voiced = f0 > 0
    true_track = np.array([0.0, 170.0, 0.0, 170.0, 0.0, 175.0, 0.0], dtype=np.float32)
    audio = _synthesize_voice_like_audio(true_track, hop_length=160)
    conf = np.full(len(f0), 0.8, dtype=np.float32)

    out = _resolve_octave_from_waveform(
        f0,
        voiced,
        audio,
        sample_rate=16000,
        hop_length=160,
        f0_min=50.0,
        f0_max=1100.0,
        confidence=conf,
    )
    assert np.all(out[~voiced] == 0.0), "Unvoiced frames should remain exactly 0"


if __name__ == "__main__":
    tests = [
        test_waveform_viterbi_resolves_harmonic_mistrack,
        test_waveform_viterbi_preserves_clean_track,
        test_waveform_viterbi_keeps_unvoiced_zero,
    ]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
