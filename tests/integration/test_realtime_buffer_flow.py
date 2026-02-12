from __future__ import annotations

import sys
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import numpy as np

# Keep direct invocation behavior consistent with existing integration tests.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rcwx.audio.buffer import RingOutputBuffer
from rcwx.pipeline.realtime_unified import RealtimeStats, RealtimeVoiceChangerUnified


def _make_vc(
    *,
    hop_out: int,
    frames: int,
    out_sr: int = 44100,
    buffer_margin: float = 0.5,
    output_queue_size: int = 8,
) -> RealtimeVoiceChangerUnified:
    """Create minimal runtime state for _on_audio_output buffer-flow simulation."""
    vc = RealtimeVoiceChangerUnified.__new__(RealtimeVoiceChangerUnified)

    vc._running = True
    vc._output_started = False
    vc._prebuffer_chunks = 1
    vc._chunks_ready = 0

    vc._output_queue = Queue(maxsize=output_queue_size)
    vc.output_buffer = RingOutputBuffer(capacity_samples=hop_out * 4, fade_samples=0)

    vc._interp_cache_frames = 0
    vc._interp_cache_x_base = None

    vc._runtime_output_sample_rate = out_sr
    vc._hop_samples_out = hop_out
    vc.config = SimpleNamespace(buffer_margin=buffer_margin)
    vc.stats = RealtimeStats()

    # Sanity: this simulation expects exact 4 callbacks per hop.
    assert hop_out == frames * 4, f"Expected hop_out == frames*4, got {hop_out} vs {frames}"

    return vc


def test_realtime_output_buffer_no_drop_no_overrun() -> None:
    """Pseudo-streaming output should keep sample balance without drops/overruns."""
    # Real values from 44.1kHz runtime path (chunk=0.36s).
    hop_out = 15876
    frames = 3969
    n_chunks = 600

    vc = _make_vc(hop_out=hop_out, frames=frames)

    rng = np.random.RandomState(1234)
    produced_samples = 0
    consumed_samples = 0
    queue_put_failures = 0

    for _ in range(n_chunks):
        # Simulate one inference chunk becoming available.
        chunk = (rng.randn(hop_out).astype(np.float32) * 0.03)
        try:
            vc._output_queue.put_nowait(chunk)
            produced_samples += hop_out
        except Exception:
            # Mirrors inference-thread overrun behavior ("dropping chunk").
            queue_put_failures += 1

        # Simulate four output callbacks per chunk.
        for _ in range(4):
            out = vc._on_audio_output(frames)
            assert len(out) == frames
            consumed_samples += len(out)

    # Drain any residual queued audio.
    for _ in range(16):
        if vc._output_queue.qsize() == 0 and vc.output_buffer.available == 0:
            break
        out = vc._on_audio_output(frames)
        assert len(out) == frames
        consumed_samples += len(out)

    assert queue_put_failures == 0, f"Output queue overrun: dropped_chunks={queue_put_failures}"
    assert vc.output_buffer.samples_dropped == 0, (
        f"Ring buffer overflow drop detected: dropped_samples={vc.output_buffer.samples_dropped}"
    )
    assert vc.stats.buffer_trims == 0, (
        f"Unexpected drift trim in steady-state pseudo flow: trims={vc.stats.buffer_trims}"
    )
    assert consumed_samples == produced_samples, (
        f"Sample imbalance detected: produced={produced_samples}, consumed={consumed_samples}"
    )


if __name__ == "__main__":
    test_realtime_output_buffer_no_drop_no_overrun()
    print("PASS: realtime buffer flow simulation")

