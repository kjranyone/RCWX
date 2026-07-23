"""LookaheadLimiter parity and hot-path shape checks."""

from __future__ import annotations

import numpy as np

from rcwx.audio.postprocess import LookaheadLimiter, PostprocessConfig


def _reference_limiter(
    audio: np.ndarray,
    *,
    sample_rate: int,
    threshold_db: float,
    release_ms: float,
    lookahead_ms: float,
) -> np.ndarray:
    """Scalar reference matching the pre-vectorization algorithm."""
    lookahead = max(1, min(int(lookahead_ms * sample_rate / 1000), 256))
    threshold = 10 ** (threshold_db / 20)
    release_samples = release_ms * sample_rate / 1000
    release_coeff = float(np.exp(-1.0 / max(release_samples, 1.0)))

    delay_buf = np.zeros(lookahead, dtype=np.float32)
    delay_idx = 0
    envelope = 1.0
    output = np.zeros_like(audio, dtype=np.float32)

    for i in range(len(audio)):
        in_sample = float(audio[i])
        out_sample = float(delay_buf[delay_idx])
        delay_buf[delay_idx] = in_sample
        delay_idx = (delay_idx + 1) % lookahead

        peak = abs(in_sample)
        if peak > envelope:
            envelope = peak
        else:
            envelope = release_coeff * envelope + (1.0 - release_coeff) * peak

        gain = threshold / envelope if envelope > threshold else 1.0
        output[i] = out_sample * gain

    return output


def test_lookahead_limiter_matches_scalar_reference() -> None:
    rng = np.random.default_rng(0)
    # Mix quiet and overshoot peaks so both gain paths are exercised.
    audio = (0.2 * rng.standard_normal(960)).astype(np.float32)
    audio[100:120] = 1.5
    audio[400:430] = -1.2

    cfg = PostprocessConfig(
        limiter_threshold_db=-1.0,
        limiter_release_ms=80.0,
        limiter_lookahead_ms=5.0,
    )
    lim = LookaheadLimiter(48000, cfg)
    got = lim.process(audio.copy())
    ref = _reference_limiter(
        audio,
        sample_rate=48000,
        threshold_db=cfg.limiter_threshold_db,
        release_ms=cfg.limiter_release_ms,
        lookahead_ms=cfg.limiter_lookahead_ms,
    )

    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)


def test_lookahead_limiter_state_continues_across_chunks() -> None:
    rng = np.random.default_rng(1)
    audio = (0.3 * rng.standard_normal(1920)).astype(np.float32)
    audio[900:980] = 1.4

    cfg = PostprocessConfig()
    streamed = LookaheadLimiter(48000, cfg)
    batch = LookaheadLimiter(48000, cfg)

    out_stream = np.concatenate(
        [
            streamed.process(audio[:960].copy()),
            streamed.process(audio[960:].copy()),
        ]
    )
    out_batch = batch.process(audio.copy())

    np.testing.assert_allclose(out_stream, out_batch, rtol=1e-5, atol=1e-5)
