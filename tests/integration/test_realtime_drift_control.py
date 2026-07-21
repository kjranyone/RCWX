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
from rcwx.pipeline.realtime_unified import RealtimeStats, RealtimeVoiceChangerUnified


def _make_vc(
    hop_out: int,
    out_sr: int = 44100,
    buffer_margin: float = 0.5,
    prebuffer_chunks: int = 1,
    latency_mode: str = "balanced",
) -> RealtimeVoiceChangerUnified:
    """Create a minimal RealtimeVoiceChangerUnified instance for output-callback tests."""
    vc = RealtimeVoiceChangerUnified.__new__(RealtimeVoiceChangerUnified)

    vc._running = True
    vc._output_started = True
    vc._prebuffer_chunks = prebuffer_chunks
    vc._chunks_ready = prebuffer_chunks

    vc._output_queue = Queue()
    vc.output_buffer = RingOutputBuffer(capacity_samples=hop_out * 16, fade_samples=0)

    # Drift-control state (floor-based, skip-only).
    vc._level_window: deque = None  # type: ignore[assignment]
    vc._level_window_frames = 0

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
    window_len = max(4, -(-2 * vc._hop_samples_out // max(1, frames)))
    vc._level_window = deque((floor,) * window_len, maxlen=window_len)
    vc._level_window_frames = frames


def test_shed_threshold_above_standing_latency() -> None:
    """Threshold must sit well above the natural standing floor (~prebuffer*hop).

    With prebuffer=1 the ring conserves ~1 hop of standing latency, so the
    steady-state floor is ~hop.  The threshold must exceed it (plus a margin)
    or drift control would fire every callback and buzz.
    """
    hop_out = 4410  # chunk=0.1s @44100, matching the logged ASIO scenario

    vc = _make_vc(hop_out=hop_out, prebuffer_chunks=1)
    threshold, target = vc._compute_shed_threshold()
    # Natural floor ≈ prebuffer_chunks*hop = 4410; threshold must be clearly above.
    assert threshold > hop_out * 2, (
        f"threshold {threshold} too close to standing floor {hop_out} (need >2x)"
    )
    # Skip target lands between the standing floor and the threshold.
    assert hop_out < target < threshold

    # Larger prebuffer raises both (more standing latency tolerated).
    vc2 = _make_vc(hop_out=hop_out, prebuffer_chunks=2)
    t2, _ = vc2._compute_shed_threshold()
    assert t2 > threshold, "larger prebuffer must raise the shed threshold"

    # Tighter margin lowers the threshold (less tolerance).
    vc.config.buffer_margin = 0.3
    t_tight, _ = vc._compute_shed_threshold()
    assert t_tight < threshold, "tighter margin should lower threshold"


def test_no_skip_in_steady_state_small_block() -> None:
    """A 64-frame ASIO block with the floor at the natural level must NOT skip.

    Reproduces the logged scenario (hop=4410, frames=64, floor≈4490) that
    previously buzzed: the natural standing floor sits just above the old
    threshold and fired a 1-sample compress every callback.
    """
    hop_out = 4410
    frames = 64  # driver-preferred ASIO buffer size in the log

    vc = _make_vc(hop_out=hop_out, prebuffer_chunks=1)
    vc.output_buffer.add(np.zeros(hop_out * 2, dtype=np.float32))
    _prime_floor(vc, frames, floor=hop_out + 80)  # natural standing + small wobble

    out = vc._on_audio_output(frames)

    assert len(out) == frames
    assert vc.stats.buffer_trims == 0, "natural standing latency must not trigger a skip"


def test_no_skip_in_steady_state_large_block() -> None:
    """Steady state must also stay quiet for larger device blocks."""
    cases = [
        (18522, 4630),  # chunk=0.42s, blocksize≈hop/4
        (15876, 3969),  # chunk=0.36s
    ]
    for hop_out, frames in cases:
        vc = _make_vc(hop_out=hop_out, prebuffer_chunks=1)
        vc.output_buffer.add(np.zeros(hop_out * 2, dtype=np.float32))
        _prime_floor(vc, frames, floor=hop_out)  # natural standing floor

        out = vc._on_audio_output(frames)

        assert len(out) == frames
        assert vc.stats.buffer_trims == 0, (
            f"steady state trimmed: hop_out={hop_out}, frames={frames}"
        )


def test_skip_when_accumulated_above_threshold() -> None:
    """Genuine accumulation (floor well above standing) must hard-skip."""
    hop_out = 4410
    frames = 64

    vc = _make_vc(hop_out=hop_out, prebuffer_chunks=1)
    backlog = hop_out * 5
    vc.output_buffer.add(np.zeros(backlog, dtype=np.float32))
    threshold, target = vc._compute_shed_threshold()
    _prime_floor(vc, frames, floor=threshold + hop_out)  # one chunk of excess

    available_before = vc.output_buffer.available
    _ = vc._on_audio_output(frames)

    assert vc.stats.buffer_trims == 1, "accumulation above threshold should skip once"
    assert vc.output_buffer.available < available_before, "skip must discard samples"
    # Window cleared after a skip so the post-skip level is re-measured.
    assert len(vc._level_window) == 1  # one fresh post-read entry appended


def test_aggressive_mode_sheds_one_hop_floor() -> None:
    """Aggressive mode treats a persistent one-hop floor as removable lag."""
    hop_out = 4410
    frames = 64
    vc = _make_vc(
        hop_out=hop_out,
        buffer_margin=0.25,
        latency_mode="aggressive",
    )
    threshold, target = vc._compute_shed_threshold()

    assert threshold == int(hop_out * 0.75)
    assert target == hop_out // 4

    vc.output_buffer.add(np.zeros(hop_out * 2, dtype=np.float32))
    _prime_floor(vc, frames, floor=hop_out)
    _ = vc._on_audio_output(frames)

    assert vc.stats.buffer_trims == 1
    assert len(vc._level_window) == 1
    assert vc._buffer_floor_samples == target


def test_aggressive_mode_caps_non_asio_callback_at_10ms() -> None:
    vc = RealtimeVoiceChangerUnified.__new__(RealtimeVoiceChangerUnified)
    vc.config = SimpleNamespace(chunk_sec=0.1, latency_mode="aggressive")
    assert vc._audio_callback_sec() == 0.010

    vc.config.latency_mode = "balanced"
    assert vc._audio_callback_sec() == 0.025


def test_sub100_mode_uses_five_ms_callback_and_jitter_guard() -> None:
    hop_out = 1920
    vc = _make_vc(
        hop_out=hop_out,
        out_sr=48000,
        buffer_margin=0.1,
        latency_mode="sub100",
    )
    vc.config.chunk_sec = 0.04

    assert vc._audio_callback_sec() == 0.005
    assert vc._compute_shed_threshold() == (hop_out * 5 // 4, hop_out)
    assert vc.stats.jitter_guard_ms == 40.0

    vc._inference_times = deque([16.0] * 20, maxlen=256)
    vc.stats.inference_p50_ms = 16.0
    vc.stats.inference_p99_ms = 30.0
    assert vc._compute_shed_threshold() == (hop_out * 5 // 8, hop_out // 2)
    assert vc.stats.jitter_guard_ms == 20.0

    vc.stats.inference_p99_ms = 45.0
    threshold, target = vc._compute_shed_threshold()
    assert threshold == 1872
    assert target == 1632
    assert vc.stats.jitter_guard_ms == 34.0

    vc._prebuffer_chunks = 1
    vc.config.prebuffer_chunks = 1
    vc.set_latency_mode("sub100")
    assert vc._prebuffer_chunks == 2
    assert vc.config.prebuffer_chunks == 2


def test_sub100_underrun_rearms_two_hop_prebuffer() -> None:
    hop_out = 1920
    frames = 480
    vc = _make_vc(
        hop_out=hop_out,
        out_sr=48000,
        prebuffer_chunks=2,
        latency_mode="sub100",
    )

    first = vc._on_audio_output(frames)
    assert np.count_nonzero(first) == 0
    assert vc.stats.buffer_underruns == 1
    assert vc._output_started is False
    assert vc._chunks_ready == 0

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


def test_frontier_mode_uses_20ms_deadline_policy() -> None:
    hop_out = 960
    vc = _make_vc(
        hop_out=hop_out,
        out_sr=48000,
        buffer_margin=0.1,
        prebuffer_chunks=3,
        latency_mode="frontier",
    )
    vc.config.chunk_sec = 0.02

    assert vc._audio_callback_sec() == 0.0025
    assert vc._compute_shed_threshold() == (1200, 960)
    assert vc.stats.jitter_guard_ms == 20.0

    vc._inference_times = deque([12.0] * 20, maxlen=256)
    vc.stats.inference_p50_ms = 12.0
    vc.stats.inference_p99_ms = 17.0
    assert vc._compute_shed_threshold() == (600, 480)
    assert vc.stats.jitter_guard_ms == 10.0

    vc._prebuffer_chunks = 1
    vc.config.prebuffer_chunks = 1
    vc.set_latency_mode("frontier")
    assert vc._prebuffer_chunks == 3
    assert vc.config.prebuffer_chunks == 3


def test_frontier_underrun_rearms_three_hop_prebuffer() -> None:
    hop_out = 960
    frames = 120
    vc = _make_vc(
        hop_out=hop_out,
        out_sr=48000,
        prebuffer_chunks=3,
        latency_mode="frontier",
    )

    first = vc._on_audio_output(frames)
    assert np.count_nonzero(first) == 0
    assert vc.stats.buffer_underruns == 1
    assert vc._output_started is False

    for chunks_ready in (1, 2):
        vc._output_queue.put_nowait(np.ones(hop_out, dtype=np.float32))
        waiting = vc._on_audio_output(frames)
        assert np.count_nonzero(waiting) == 0
        assert vc._output_started is False
        assert vc._chunks_ready == chunks_ready

    vc._output_queue.put_nowait(np.ones(hop_out, dtype=np.float32))
    recovered = vc._on_audio_output(frames)
    assert np.count_nonzero(recovered) == frames
    assert vc._output_started is True
    assert vc._chunks_ready == 3
    assert vc.stats.buffer_underruns == 1


def test_latency_estimate_uses_persistent_floor_not_ring_sawtooth() -> None:
    hop_out = 4800
    vc = _make_vc(hop_out=hop_out, out_sr=48000, latency_mode="aggressive")
    vc.config.chunk_sec = 0.1
    vc.config.crossfade_sec = 0.01
    vc.config.use_sola = True
    vc._buffer_floor_samples = 1200

    # A freshly added output hop is normal burst/drain state, not another
    # 100ms of persistent latency.
    vc.output_buffer.add(np.zeros(hop_out, dtype=np.float32))
    vc._update_latency_estimate(inference_ms=35.0)

    assert vc.stats.hop_latency_ms == 100.0
    assert vc.stats.buffer_latency_ms == 25.0
    assert vc.stats.queue_latency_ms == 0.0
    assert vc.stats.sola_latency_ms == 10.0
    assert vc.stats.latency_ms == 170.0


if __name__ == "__main__":
    test_shed_threshold_above_standing_latency()
    test_no_skip_in_steady_state_small_block()
    test_no_skip_in_steady_state_large_block()
    test_skip_when_accumulated_above_threshold()
    test_aggressive_mode_sheds_one_hop_floor()
    test_aggressive_mode_caps_non_asio_callback_at_10ms()
    test_sub100_mode_uses_five_ms_callback_and_jitter_guard()
    test_sub100_underrun_rearms_two_hop_prebuffer()
    test_frontier_mode_uses_20ms_deadline_policy()
    test_frontier_underrun_rearms_three_hop_prebuffer()
    test_latency_estimate_uses_persistent_floor_not_ring_sawtooth()
    print("PASS: realtime drift-control tests")
