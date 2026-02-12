from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Keep direct invocation behavior consistent with existing integration tests.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rcwx.pipeline.realtime_unified import RealtimeVoiceChangerUnified


def _make_vc(out_sr: int = 48000) -> RealtimeVoiceChangerUnified:
    vc = RealtimeVoiceChangerUnified.__new__(RealtimeVoiceChangerUnified)
    vc._runtime_output_sample_rate = out_sr
    vc._prev_tail_rms = 0.0
    return vc


def test_boundary_gain_skips_quiet_head() -> None:
    """Quiet chunk heads (e.g. breaths) should not be amplified."""
    vc = _make_vc()
    vc._prev_tail_rms = 0.10

    head = np.full(240, 5e-4, dtype=np.float32)   # 5ms @ 48kHz
    body = np.full(3000, 0.05, dtype=np.float32)
    chunk = np.concatenate([head, body])

    out = vc._apply_output_boundary_gain(chunk)

    assert np.allclose(out, chunk), "Quiet head should bypass gain continuity"
    assert vc._prev_tail_rms > 0.01


def test_boundary_gain_is_tight_and_local() -> None:
    """Gain continuity should be mild and limited to the first short ramp."""
    vc = _make_vc()
    vc._prev_tail_rms = 0.20

    chunk = np.full(5000, 0.10, dtype=np.float32)
    out = vc._apply_output_boundary_gain(chunk)

    # Tight clamp: first sample gain should not exceed 1.1x.
    assert out[0] <= chunk[0] * 1.1 + 1e-6
    assert out[0] >= chunk[0] * 0.9 - 1e-6

    # Ramp is 10ms @ 48kHz -> 480 samples. After that, signal should remain unchanged.
    assert np.allclose(out[600:], chunk[600:], atol=1e-6)


def test_boundary_gain_reset_state() -> None:
    vc = _make_vc()
    vc._prev_tail_rms = 0.12
    vc._reset_boundary_continuity_state()
    assert vc._prev_tail_rms == 0.0


if __name__ == "__main__":
    test_boundary_gain_skips_quiet_head()
    test_boundary_gain_is_tight_and_local()
    test_boundary_gain_reset_state()
    print("PASS: realtime boundary gain tests")
