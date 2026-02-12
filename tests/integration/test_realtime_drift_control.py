from __future__ import annotations

import sys
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import numpy as np

# Keep direct invocation behavior consistent with existing integration tests.
sys.path.insert(0, str(Path(__file__).parent.parent))

from rcwx.audio.buffer import RingOutputBuffer
from rcwx.pipeline.realtime_unified import RealtimeStats, RealtimeVoiceChangerUnified


def _make_vc(hop_out: int, out_sr: int = 44100, buffer_margin: float = 0.5) -> RealtimeVoiceChangerUnified:
    """Create a minimal RealtimeVoiceChangerUnified instance for output-callback tests."""
    vc = RealtimeVoiceChangerUnified.__new__(RealtimeVoiceChangerUnified)

    vc._running = True
    vc._output_started = True
    vc._prebuffer_chunks = 1
    vc._chunks_ready = 1

    vc._output_queue = Queue()
    vc.output_buffer = RingOutputBuffer(capacity_samples=hop_out * 4, fade_samples=0)

    vc._interp_cache_frames = 0
    vc._interp_cache_x_base = None

    vc._runtime_output_sample_rate = out_sr
    vc._hop_samples_out = hop_out
    vc.config = SimpleNamespace(buffer_margin=buffer_margin)
    vc.stats = RealtimeStats()

    return vc


def test_no_trim_at_exact_hop_multiple_shapes() -> None:
    """Normal ring peak (exactly one hop) must not trigger drift trim."""
    # Real values observed in logs for 44.1kHz runtime.
    cases = [
        (18522, 4630),  # chunk=0.42s
        (15876, 3969),  # chunk=0.36s
        (4410, 1102),   # chunk=0.10s
    ]

    for hop_out, frames in cases:
        vc = _make_vc(hop_out=hop_out)
        vc.output_buffer.add(np.zeros(hop_out, dtype=np.float32))

        out = vc._on_audio_output(frames)

        assert len(out) == frames
        assert vc.stats.buffer_trims == 0, (
            f"Unexpected trim at normal peak: hop_out={hop_out}, frames={frames}"
        )


def test_trim_when_above_one_hop_backlog() -> None:
    """Backlog (>1 hop queued) should still trigger drift trim."""
    hop_out = 18522
    frames = 4630

    vc = _make_vc(hop_out=hop_out)
    vc.output_buffer.add(np.zeros(hop_out * 2, dtype=np.float32))

    _ = vc._on_audio_output(frames)

    assert vc.stats.buffer_trims >= 1


def test_neutral_margin_threshold_matches_one_hop() -> None:
    """With buffer_margin=0.5, threshold should match exact one-hop peak."""
    hop_out = 18522
    frames = 4630

    vc = _make_vc(hop_out=hop_out, buffer_margin=0.5)
    target = vc._compute_drain_target_for_frames(frames)
    assert target + frames == hop_out

    vc.config.buffer_margin = 0.3
    tighter = vc._compute_drain_target_for_frames(frames)
    assert tighter < target

    vc.config.buffer_margin = 1.0
    relaxed = vc._compute_drain_target_for_frames(frames)
    assert relaxed > target


if __name__ == "__main__":
    test_no_trim_at_exact_hop_multiple_shapes()
    test_trim_when_above_one_hop_backlog()
    test_neutral_margin_threshold_matches_one_hop()
    print("PASS: realtime drift-control tests")
