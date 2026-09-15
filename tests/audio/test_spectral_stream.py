"""Streaming spectral gate (persistent STFT denoise for the realtime path).

The stateless ``SpectralGateDenoiser`` rebuilt per hop never finished
auto-learning (10 frames = 416ms), so realtime chunks below ~416ms got a
windowed passthrough — spectral "denoise" did nothing.  ``StreamingSpectralGate``
carries STFT state across hops:

- every call returns exactly ``len(audio)`` samples (zero added latency;
  the newest <= n_fft - hop samples are crossfaded toward dry)
- steady noise is suppressed once the profile settles, and the profile
  persists across chunks (no per-chunk re-learning)
- a tone well above the noise floor passes essentially unchanged
- no chunk-edge amplification (COLA-exact sqrt-Hann pairs, no per-sample
  normalization of finalized samples)
- ``denoise(streaming_state=...)`` routes through the persistent gate even
  for hops shorter than the offline n_fft
- the realtime preprocess stage uses it for method="spectral"
"""

from __future__ import annotations

import numpy as np

import rcwx.pipeline.realtime_unified as ru
from rcwx.audio.denoise import StreamingSpectralGate, denoise
from rcwx.pipeline.realtime_config import RealtimeConfig

SR = 16000


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def _noise(rng: np.random.RandomState, n: int, amp: float = 0.01) -> np.ndarray:
    return (rng.randn(n) * amp).astype(np.float32)


def _sine(freq: float, n: int, amp: float) -> np.ndarray:
    t = np.arange(n) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_same_length_across_chunk_sizes() -> None:
    gate = StreamingSpectralGate(SR)
    rng = np.random.RandomState(0)
    signal = _noise(rng, SR * 2)
    pos = 0
    for size in [4800, 320, 641, 2048, 16000, 137, 100]:
        if pos >= len(signal):
            break
        chunk = signal[pos : pos + size]
        out = gate.process(chunk)
        assert len(out) == len(chunk), (size, len(out), len(chunk))
        pos += len(out)
    if pos < len(signal):
        out = gate.process(signal[pos:])
        assert len(out) == len(signal) - pos


def test_steady_noise_suppressed() -> None:
    # Regression for the per-hop rebuild: with the stateless class every
    # 300ms chunk spent all frames "learning" and passed through.
    gate = StreamingSpectralGate(SR)
    rng = np.random.RandomState(1)
    outs, ins = [], []
    for _ in range(8):  # 8 x 300ms = 2.4s
        chunk = _noise(rng, 4800)
        outs.append(gate.process(chunk))
        ins.append(chunk)
    # The profile needs ~0.5s to descend to the floor; later chunks must
    # be clearly attenuated.
    assert _rms(np.concatenate(outs[4:])) < 0.6 * _rms(np.concatenate(ins[4:]))


def test_profile_persists_across_chunks() -> None:
    gate = StreamingSpectralGate(SR)
    rng = np.random.RandomState(2)
    for _ in range(4):
        gate.process(_noise(rng, 4800))
    # A fresh chunk of the same noise must be attenuated immediately —
    # no learning-passthrough restart at the chunk boundary.
    chunk = _noise(rng, 4800)
    out = gate.process(chunk)
    assert _rms(out) < 0.6 * _rms(chunk)


def test_tone_above_floor_passes() -> None:
    gate = StreamingSpectralGate(SR)
    rng = np.random.RandomState(3)
    for _ in range(6):
        gate.process(_noise(rng, 4800))
    # Loud tone on top of the same noise: must pass essentially unchanged
    # while the noise underneath stays attenuated.
    chunk = _sine(220.0, 3 * 4800, 0.3) + _noise(rng, 3 * 4800)
    out = gate.process(chunk)
    clean = _sine(220.0, 3 * 4800, 0.3)
    assert _rms(out) > 0.9 * _rms(clean)
    assert _rms(out - clean) < 0.7 * _rms(chunk - clean)


def test_no_edge_amplification() -> None:
    # The offline overlap-add divided by a window-sum floored at 1e-8, which
    # could amplify chunk edges.  Streaming output must never exceed the
    # input envelope meaningfully.
    gate = StreamingSpectralGate(SR)
    rng = np.random.RandomState(4)
    max_in, max_out = 0.0, 0.0
    for _ in range(6):
        chunk = _sine(220.0, 4800, 0.3) + _noise(rng, 4800, 0.02)
        out = gate.process(chunk)
        max_in = max(max_in, float(np.max(np.abs(chunk))))
        max_out = max(max_out, float(np.max(np.abs(out))))
    assert max_out < max_in * 1.2


def test_reset_clears_state() -> None:
    gate = StreamingSpectralGate(SR)
    rng = np.random.RandomState(5)
    for _ in range(6):
        gate.process(_noise(rng, 4800))
    assert gate.noise_profile is not None
    gate.reset()
    assert gate.noise_profile is None
    chunk = _noise(rng, 4800)
    out = gate.process(chunk)
    # Fresh gate has no profile -> first frames pass through at unity.
    assert _rms(out) > 0.8 * _rms(chunk)


def test_denoise_streaming_short_hop() -> None:
    # 320 samples < offline n_fft: the stateless path passes short hops
    # through; the streaming path denoises them.
    rng = np.random.RandomState(6)
    gate = StreamingSpectralGate(SR)
    outs, ins = [], []
    for _ in range(40):  # 40 x 20ms = 0.8s
        hop = _noise(rng, 320)
        outs.append(denoise(hop, method="spectral", streaming_state=gate))
        ins.append(hop)
    assert _rms(np.concatenate(outs[20:])) < 0.6 * _rms(np.concatenate(ins[20:]))


def test_realtime_preprocess_uses_streaming_spectral() -> None:
    cfg = RealtimeConfig(
        mic_sample_rate=16000,
        output_sample_rate=16000,
        chunk_sec=0.3,
        latency_mode="normal",
        denoise_enabled=True,
        denoise_method="spectral",
        use_sola=False,
    )
    changer = ru.RealtimeVoiceChangerUnified(_FakePipeline(), cfg)
    rng = np.random.RandomState(7)
    outs, ins = [], []
    for _ in range(8):
        hop_in = _noise(rng, 4800)
        hop, _ = changer._preprocess_hop(hop_in)
        outs.append(hop)
        ins.append(hop_in)
    # Before the streaming gate, every 300ms hop was a learning no-op.
    assert _rms(np.concatenate(outs[4:])) < 0.6 * _rms(np.concatenate(ins[4:]))


class _FakePipeline:
    _loaded = True
    device = "cpu"
    sample_rate = 16000
    synthesizer = None
    accelerator_index = None
    stage_times: dict = {}


if __name__ == "__main__":
    test_same_length_across_chunk_sizes()
    test_steady_noise_suppressed()
    test_profile_persists_across_chunks()
    test_tone_above_floor_passes()
    test_no_edge_amplification()
    test_reset_clears_state()
    test_denoise_streaming_short_hop()
    test_realtime_preprocess_uses_streaming_spectral()
    print("OK: all spectral stream tests passed")
