from __future__ import annotations

import sys
from collections import deque
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import numpy as np

# Keep direct invocation behavior consistent with existing integration tests.
sys.path.insert(0, str(Path(__file__).parent.parent))

from rcwx.audio.buffer import RingOutputBuffer
from rcwx.pipeline.drift_control import FloorTracker
from rcwx.pipeline.realtime_unified import RealtimeStats, RealtimeVoiceChangerUnified


def _make_vc(
    hop_out: int,
    out_sr: int = 44100,
    buffer_margin: float = 0.25,
    prebuffer_chunks: int = 1,
    latency_mode: str = "normal",
) -> RealtimeVoiceChangerUnified:
    """Create a minimal RealtimeVoiceChangerUnified instance for output-callback tests."""
    vc = RealtimeVoiceChangerUnified.__new__(RealtimeVoiceChangerUnified)

    vc._running = True
    vc._output_started = True
    vc._prebuffer_chunks = prebuffer_chunks
    vc._required_prebuffer_chunks = prebuffer_chunks
    vc._chunks_ready = prebuffer_chunks

    vc._output_queue = Queue()
    vc.output_buffer = RingOutputBuffer(capacity_samples=hop_out * 16, fade_samples=0)

    # Drift-control state (floor-based, skip-only).
    vc._floor = FloorTracker()
    vc._observed_callback_sec = 0.0

    vc._runtime_output_sample_rate = out_sr
    vc._hop_samples_out = hop_out
    vc.config = SimpleNamespace(
        buffer_margin=buffer_margin,
        latency_mode=latency_mode,
    )
    vc.stats = RealtimeStats()

    return vc


def _prime_floor(vc: RealtimeVoiceChangerUnified, frames: int, floor: int) -> None:
    """Pre-fill the level window so the next callback sees ``floor`` as its minimum.

    Avoids having to run enough callbacks to fill the window organically; the
    windowed MIN is what drift control acts on, not the call count.
    """
    vc._floor.prime(frames, vc._hop_samples_out, floor)


def test_normal_mode_sheds_one_hop_floor() -> None:
    """Normal treats a persistent one-hop floor as removable lag."""
    hop_out = 4410
    frames = 64
    vc = _make_vc(
        hop_out=hop_out,
        buffer_margin=0.25,
        latency_mode="normal",
    )
    threshold, target = vc._compute_shed_threshold()

    assert threshold == int(hop_out * 0.75)
    assert target == hop_out // 4

    vc.output_buffer.add(np.zeros(hop_out * 2, dtype=np.float32))
    _prime_floor(vc, frames, floor=hop_out)
    _ = vc._on_audio_output(frames)

    assert vc.stats.buffer_trims == 1
    assert vc._floor.floor_samples == target


def test_normal_mode_caps_non_asio_callback_at_10ms() -> None:
    vc = RealtimeVoiceChangerUnified.__new__(RealtimeVoiceChangerUnified)
    vc.config = SimpleNamespace(chunk_sec=0.1, latency_mode="normal")
    assert vc._audio_callback_sec() == 0.010

    vc.config.latency_mode = "aggressive"
    assert vc._audio_callback_sec() == 0.0025


def test_aggressive_mode_uses_20ms_deadline_policy() -> None:
    hop_out = 960
    vc = _make_vc(
        hop_out=hop_out,
        out_sr=48000,
        buffer_margin=0.1,
        prebuffer_chunks=3,
        latency_mode="aggressive",
    )
    vc.config.chunk_sec = 0.02

    assert vc._audio_callback_sec() == 0.0025
    assert vc._compute_shed_threshold() == (1200, 960)
    assert vc.stats.jitter_guard_ms == 20.0

    vc._inference_times = deque([12.0] * 20, maxlen=256)
    vc.stats.inference_p50_ms = 12.0
    vc.stats.inference_p99_ms = 17.0
    # guard = max(0.5*hop, jitter 5ms + requested callback 2.5ms) = 10ms;
    # hysteresis band = max(hop/4, callback) = 240.
    assert vc._compute_shed_threshold() == (720, 480)
    assert vc.stats.jitter_guard_ms == 10.0

    # ASIO ignores the requested callback duration: the observed blocksize
    # (256 frames = 5.33ms) must widen both the guard and the band.
    vc._observed_callback_sec = 256 / 48000
    assert vc._compute_shed_threshold() == (752, 496)
    vc._observed_callback_sec = 0.0

    vc._prebuffer_chunks = 1
    vc.config.prebuffer_chunks = 1
    vc.set_latency_mode("aggressive")
    assert vc._prebuffer_chunks == 3
    assert vc.config.prebuffer_chunks == 3


def test_aggressive_underrun_rearms_reduced_prebuffer() -> None:
    """An underrun re-arms with 2 hops, not the initial 3-hop prebuffer.

    Anything beyond the shed guard is standing latency that drift control
    would immediately trim back out (discarding speech), so the re-arm
    cushion is intentionally smaller than the session-start prebuffer.
    """
    hop_out = 960
    frames = 120
    vc = _make_vc(
        hop_out=hop_out,
        out_sr=48000,
        prebuffer_chunks=3,
        latency_mode="aggressive",
    )

    first = vc._on_audio_output(frames)
    assert np.count_nonzero(first) == 0
    assert vc.stats.buffer_underruns == 1
    assert vc._output_started is False
    assert vc._required_prebuffer_chunks == 2
    assert vc._prebuffer_chunks == 3  # session-start prebuffer unchanged

    vc._output_queue.put_nowait(np.ones(hop_out, dtype=np.float32))
    waiting = vc._on_audio_output(frames)
    assert np.count_nonzero(waiting) == 0
    assert vc._output_started is False
    assert vc._chunks_ready == 1

    vc._output_queue.put_nowait(np.ones(hop_out, dtype=np.float32))
    recovered = vc._on_audio_output(frames)
    assert np.count_nonzero(recovered) == frames
    assert vc._output_started is True
    assert vc._chunks_ready == 2
    assert vc.stats.buffer_underruns == 1


def test_latency_estimate_uses_persistent_floor_not_ring_sawtooth() -> None:
    hop_out = 4800
    vc = _make_vc(hop_out=hop_out, out_sr=48000, latency_mode="normal")
    vc.config.chunk_sec = 0.1
    vc.config.crossfade_sec = 0.01
    vc.config.use_sola = True
    vc._floor.floor_samples = 1200

    # A freshly added output hop is normal burst/drain state, not another
    # 100ms of persistent latency.
    vc.output_buffer.add(np.zeros(hop_out, dtype=np.float32))
    vc._update_latency_estimate(inference_ms=35.0)

    assert vc.stats.hop_latency_ms == 100.0
    assert vc.stats.buffer_latency_ms == 25.0
    assert vc.stats.queue_latency_ms == 0.0
    assert vc.stats.sola_latency_ms == 10.0
    assert vc.stats.latency_ms == 170.0


def test_aggressive_latency_counts_full_sola_prefix() -> None:
    hop_out = 960
    vc = _make_vc(hop_out=hop_out, out_sr=48000, latency_mode="aggressive")
    vc.config.chunk_sec = 0.02
    vc.config.crossfade_sec = 0.01
    vc.config.use_sola = True
    vc.pipeline = SimpleNamespace(sample_rate=48000)
    vc._sola_extra_model = 1440
    vc._floor.floor_samples = 480

    vc._update_latency_estimate(inference_ms=10.0)

    assert vc.stats.hop_latency_ms == 20.0
    assert vc.stats.buffer_latency_ms == 10.0
    assert vc.stats.sola_latency_ms == 30.0
    assert vc.stats.latency_ms == 70.0


if __name__ == "__main__":
    test_normal_mode_sheds_one_hop_floor()
    test_normal_mode_caps_non_asio_callback_at_10ms()
    test_aggressive_mode_uses_20ms_deadline_policy()
    test_aggressive_underrun_rearms_reduced_prebuffer()
    test_latency_estimate_uses_persistent_floor_not_ring_sawtooth()
    test_aggressive_latency_counts_full_sola_prefix()
    print("PASS: realtime drift-control tests")
