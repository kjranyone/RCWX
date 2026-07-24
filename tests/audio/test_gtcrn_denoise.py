"""Tests for the GTCRN streaming denoiser wrapper.

The plumbing test replaces the ONNX session with an identity mask, which
turns the wrapper into a pure sqrt-Hann STFT -> ISTFT chain: if windowing,
overlap-add, carry, and FIFO adaptation are correct, the output must equal
the input delayed by exactly 512 samples (256 OLA + 256 FIFO priming),
regardless of the caller's hop size.
"""

from __future__ import annotations

import numpy as np

from rcwx.audio.gtcrn import HOP, GTCRNDenoiser, get_gtcrn_path, is_gtcrn_available


class _IdentitySession:
    def run(self, _outputs, feeds):
        return (
            feeds["mix"],
            feeds["conv_cache"],
            feeds["tra_cache"],
            feeds["inter_cache"],
        )


def _make_identity_denoiser(zero_latency: bool = False) -> GTCRNDenoiser:
    dn = GTCRNDenoiser.__new__(GTCRNDenoiser)
    dn._zero_latency = zero_latency
    dn.reset()
    dn._session = _IdentitySession()
    dn._load = lambda: None
    return dn


def test_identity_reconstruction_with_320_sample_hops() -> None:
    rng = np.random.RandomState(7)
    t = np.arange(16000)
    signal = (0.5 * np.sin(2 * np.pi * 220 * t / 16000)).astype(np.float32)
    signal += 0.05 * rng.randn(len(signal)).astype(np.float32)

    dn = _make_identity_denoiser()
    hop = 320  # 20ms — deliberately misaligned with the 256-sample STFT hop
    outs = []
    for pos in range(0, len(signal) - hop + 1, hop):
        chunk = signal[pos : pos + hop]
        out = dn.process(chunk)
        assert len(out) == len(chunk)
        outs.append(out)
    out_all = np.concatenate(outs)

    delay = 2 * HOP  # 512 samples
    aligned_out = out_all[delay:]
    aligned_in = signal[: len(aligned_out)]
    err = np.max(np.abs(aligned_out - aligned_in))
    assert err < 1e-4, f"reconstruction error {err}"


def test_identity_reconstruction_with_irregular_hops() -> None:
    rng = np.random.RandomState(3)
    signal = rng.randn(8000).astype(np.float32) * 0.1

    dn = _make_identity_denoiser()
    outs = []
    pos = 0
    for hop in (64, 512, 320, 100, 999, 256, 777, 1, 63):
        if pos + hop > len(signal):
            break
        out = dn.process(signal[pos : pos + hop])
        assert len(out) == hop
        outs.append(out)
        pos += hop
    out_all = np.concatenate(outs)

    delay = 2 * HOP
    aligned_out = out_all[delay:]
    aligned_in = signal[: len(aligned_out)]
    err = np.max(np.abs(aligned_out - aligned_in))
    assert err < 1e-4, f"reconstruction error {err}"


def test_strength_blend_uses_latency_matched_dry() -> None:
    rng = np.random.RandomState(11)
    signal = rng.randn(4000).astype(np.float32) * 0.1

    dn = _make_identity_denoiser()
    outs = []
    for pos in range(0, len(signal) - 320 + 1, 320):
        outs.append(dn.process(signal[pos : pos + 320], strength=0.5))
    out_all = np.concatenate(outs)

    # With an identity "model", wet == dry (both latency-matched), so a 0.5
    # blend must still reconstruct the delayed input exactly.
    delay = 2 * HOP
    aligned_out = out_all[delay:]
    aligned_in = signal[: len(aligned_out)]
    err = np.max(np.abs(aligned_out - aligned_in))
    assert err < 1e-4, f"blend misalignment, error {err}"


def test_zero_latency_identity_is_exact_with_zero_delay() -> None:
    """With an identity model, the speculative-edge path must reconstruct the
    input EXACTLY at zero delay: every emitted region's overlap-add sum only
    depends on real samples (the fabricated future lands in window halves
    that belong to later regions)."""
    rng = np.random.RandomState(5)
    signal = rng.randn(16000).astype(np.float32) * 0.1

    dn = _make_identity_denoiser(zero_latency=True)
    outs = []
    for pos in range(0, len(signal) - 320 + 1, 320):
        out = dn.process(signal[pos : pos + 320])
        assert len(out) == 320
        outs.append(out)
    out_all = np.concatenate(outs)

    err = np.max(np.abs(out_all - signal[: len(out_all)]))
    assert err < 1e-4, f"zero-latency reconstruction error {err}"


def test_zero_latency_identity_with_irregular_hops() -> None:
    rng = np.random.RandomState(9)
    signal = rng.randn(8000).astype(np.float32) * 0.1

    dn = _make_identity_denoiser(zero_latency=True)
    outs = []
    pos = 0
    for hop in (64, 512, 320, 100, 999, 256, 777, 1, 63, 320):
        if pos + hop > len(signal):
            break
        out = dn.process(signal[pos : pos + hop])
        assert len(out) == hop
        outs.append(out)
        pos += hop
    out_all = np.concatenate(outs)

    err = np.max(np.abs(out_all - signal[: len(out_all)]))
    assert err < 1e-4, f"zero-latency reconstruction error {err}"


def test_real_model_reduces_noise_if_available() -> None:
    from pathlib import Path

    model_path = get_gtcrn_path(Path.home() / ".cache" / "rcwx" / "models")
    if not (is_gtcrn_available() and model_path.exists()):
        print("SKIP: GTCRN model not downloaded (run: uv run rcwx download)")
        return

    rng = np.random.RandomState(0)
    sr = 16000
    t = np.arange(sr * 2) / sr
    # Speech-like: vibrato F0, rich harmonics, syllabic amplitude modulation.
    # (A steady pure tone is correctly treated as noise by DNS-trained models.)
    f0 = 140 + 15 * np.sin(2 * np.pi * 5.5 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    speechish = sum((0.6 / k) * np.sin(k * phase) for k in range(1, 12))
    syllable = np.clip(np.sin(2 * np.pi * 2.5 * t), 0, 1) ** 0.5
    speechish = (speechish * syllable * 0.25).astype(np.float32)
    noise = 0.08 * rng.randn(len(t)).astype(np.float32)
    noisy = speechish + noise

    dn = GTCRNDenoiser(model_path)
    outs = []
    for pos in range(0, len(noisy) - 320 + 1, 320):
        outs.append(dn.process(noisy[pos : pos + 320]))
    out = np.concatenate(outs)

    # Compare against noise-only processing: the denoiser should attenuate
    # pure noise substantially more than the tonal signal.
    dn2 = GTCRNDenoiser(model_path)
    outs2 = []
    for pos in range(0, len(noise) - 320 + 1, 320):
        outs2.append(dn2.process(noise[pos : pos + 320]))
    out_noise = np.concatenate(outs2)

    tail = slice(sr // 2, None)  # skip warmup
    noise_att = np.sqrt(np.mean(out_noise[tail] ** 2)) / np.sqrt(np.mean(noise[tail] ** 2))
    sig_att = np.sqrt(np.mean(out[tail] ** 2)) / np.sqrt(np.mean(noisy[tail] ** 2))
    print(f"noise attenuation: {noise_att:.3f}, signal retention: {sig_att:.3f}")
    assert noise_att < 0.2, f"noise not attenuated (ratio {noise_att:.3f})"
    assert sig_att > 0.4, f"signal destroyed (ratio {sig_att:.3f})"
