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


def _make_vc(hop_out: int, out_sr: int = 44100, buffer_margin: float = 0.5, prebuffer_chunks: int = 1) -> RealtimeVoiceChangerUnified:
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
    vc.config = SimpleNamespace(buffer_margin=buffer_margin)
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


if __name__ == "__main__":
    test_shed_threshold_above_standing_latency()
    test_no_skip_in_steady_state_small_block()
    test_no_skip_in_steady_state_large_block()
    test_skip_when_accumulated_above_threshold()
    print("PASS: realtime drift-control tests")
