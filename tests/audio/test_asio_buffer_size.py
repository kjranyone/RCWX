"""ASIO buffer-size resolution + simplex ASIO open tests.

Regression target: the ASIO buffer-size control (GUI ``asio_buffer_size``,
default 0 = follow driver control panel) used to be honored ONLY on the ASIO
duplex path (both input and output on ASIO).  With a mixed config — e.g. an
ASIO output DAC paired with a WASAPI USB mic — the duplex precondition fails,
so the output was opened through the generic fallback that never sets an ASIO
buffer, leaving the driver at its control-panel value (often 2048) and
silently ignoring the user's choice.

The fix generalizes buffer control to the simplex path via
``AudioStreamBase._try_open_asio`` + the shared ``resolve_asio_buffer_size``
helper (also used by AsioDuplexStream).  These tests pin:

1. resolve_asio_buffer_size: snapping, preferred fallback, no-query behavior
2. _try_open_asio guards: device=None and non-ASIO devices never force ASIO
3. _try_open_asio ASIO path: opens with blocksize=0 and latency=size/sr
4. plumbing: AudioInput/AudioOutput accept & store asio_buffer_size

No real ASIO device or audio hardware is required — the ctypes/PortAudio
queries are monkeypatched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import rcwx.audio.stream_base as sb
from rcwx.audio.input import AudioInput
from rcwx.audio.output import AudioOutput


def _silent_cb_out(frames: int) -> np.ndarray:
    return np.zeros(frames, dtype=np.float32)


# ---------------------------------------------------------------------------
# 1. resolve_asio_buffer_size
# ---------------------------------------------------------------------------

def test_resolve_snaps_and_prefers() -> None:
    orig = sb.query_asio_buffer_sizes
    try:
        # min=64 max=2048 preferred=2048 granularity=-1 (power-of-two)
        sb.query_asio_buffer_sizes = lambda dev, st="output": (64, 2048, 2048, -1)
        assert sb.resolve_asio_buffer_size(5, 256, "output") == 256   # valid pow2
        assert sb.resolve_asio_buffer_size(5, 300, "output") == 256   # snap down
        assert sb.resolve_asio_buffer_size(5, 0, "output") == 2048     # preferred
        assert sb.resolve_asio_buffer_size(5, 10 ** 9, "output") == 2048  # clamp max
        assert sb.resolve_asio_buffer_size(5, 1, "output") == 64        # clamp min
    finally:
        sb.query_asio_buffer_sizes = orig


def test_resolve_without_query() -> None:
    """Query unavailable (driver busy): honor an explicit request as-is, and
    return 0 for auto so the caller applies a plain latency hint."""
    orig = sb.query_asio_buffer_sizes
    try:
        sb.query_asio_buffer_sizes = lambda dev, st="output": None
        assert sb.resolve_asio_buffer_size(5, 256, "output") == 256
        assert sb.resolve_asio_buffer_size(5, 0, "output") == 0
    finally:
        sb.query_asio_buffer_sizes = orig


# ---------------------------------------------------------------------------
# 2. _try_open_asio guards
# ---------------------------------------------------------------------------

def test_try_open_asio_rejects_default_device() -> None:
    """device=None (system default) must never be force-opened as ASIO."""
    out = AudioOutput(device=None, sample_rate=48000, channels=2, callback=_silent_cb_out)
    assert out._try_open_asio() is False


def test_try_open_asio_rejects_non_asio_device() -> None:
    orig = sb.is_device_on_asio
    try:
        sb.is_device_on_asio = lambda dev, st="output": False
        out = AudioOutput(device=3, sample_rate=48000, channels=2, callback=_silent_cb_out)
        assert out._try_open_asio() is False
    finally:
        sb.is_device_on_asio = orig


# ---------------------------------------------------------------------------
# 3. _try_open_asio ASIO open arguments
# ---------------------------------------------------------------------------

def test_try_open_asio_uses_blocksize0_and_latency() -> None:
    captured: dict = {}

    class _FakeStream:
        def __init__(self, **kw):
            captured.update(kw)
            self.samplerate = kw["samplerate"]
            self.latency = kw["latency"]

        def start(self):
            captured["started"] = True

    orig_is = sb.is_device_on_asio
    orig_native = sb.query_asio_native_sample_rate
    orig_resolve = sb.resolve_asio_buffer_size
    try:
        sb.is_device_on_asio = lambda dev, st="output": True
        sb.query_asio_native_sample_rate = lambda dev, st="output": 44100
        sb.resolve_asio_buffer_size = lambda dev, req, st="output": 256

        out = AudioOutput(
            device=7, sample_rate=48000, channels=2,
            callback=_silent_cb_out, asio_buffer_size=256,
        )
        out.STREAM_CLASS = _FakeStream
        ok = out._try_open_asio()

        assert ok is True
        assert captured["blocksize"] == 0            # driver-chosen buffer
        assert captured["samplerate"] == 44100       # native rate tried first
        assert abs(captured["latency"] - 256 / 44100) < 1e-9  # size/sr
        assert captured.get("started") is True
    finally:
        sb.is_device_on_asio = orig_is
        sb.query_asio_native_sample_rate = orig_native
        sb.resolve_asio_buffer_size = orig_resolve


def test_try_open_asio_auto_uses_latency_hint() -> None:
    """When the size is unknown (resolve -> 0), a ~10ms latency hint is used
    instead of leaving PortAudio to pick 'high' (= driver max)."""
    captured: dict = {}

    class _FakeStream:
        def __init__(self, **kw):
            captured.update(kw)
            self.samplerate = kw["samplerate"]
            self.latency = kw["latency"]

        def start(self):
            pass

    orig_is = sb.is_device_on_asio
    orig_native = sb.query_asio_native_sample_rate
    orig_resolve = sb.resolve_asio_buffer_size
    try:
        sb.is_device_on_asio = lambda dev, st="output": True
        sb.query_asio_native_sample_rate = lambda dev, st="output": 48000
        sb.resolve_asio_buffer_size = lambda dev, req, st="output": 0  # unknown

        out = AudioOutput(device=7, sample_rate=48000, channels=2,
                          callback=_silent_cb_out, asio_buffer_size=0)
        out.STREAM_CLASS = _FakeStream
        assert out._try_open_asio() is True
        assert captured["blocksize"] == 0
        assert abs(captured["latency"] - 0.010) < 1e-9
    finally:
        sb.is_device_on_asio = orig_is
        sb.query_asio_native_sample_rate = orig_native
        sb.resolve_asio_buffer_size = orig_resolve


def test_try_open_asio_falls_back_on_open_failure() -> None:
    """If every ASIO open attempt raises, _try_open_asio returns False (so
    start() proceeds to the generic fallback) and leaves no stream."""
    class _FailStream:
        def __init__(self, **kw):
            raise RuntimeError("device unavailable")

    orig_is = sb.is_device_on_asio
    orig_native = sb.query_asio_native_sample_rate
    orig_resolve = sb.resolve_asio_buffer_size
    try:
        sb.is_device_on_asio = lambda dev, st="output": True
        sb.query_asio_native_sample_rate = lambda dev, st="output": 44100
        sb.resolve_asio_buffer_size = lambda dev, req, st="output": 256

        out = AudioOutput(device=7, sample_rate=48000, channels=2,
                          callback=_silent_cb_out, asio_buffer_size=256)
        out.STREAM_CLASS = _FailStream
        assert out._try_open_asio() is False
        assert out._stream is None
    finally:
        sb.is_device_on_asio = orig_is
        sb.query_asio_native_sample_rate = orig_native
        sb.resolve_asio_buffer_size = orig_resolve


# ---------------------------------------------------------------------------
# 4. Plumbing
# ---------------------------------------------------------------------------

def test_streams_store_asio_buffer_size() -> None:
    out = AudioOutput(device=1, asio_buffer_size=512, callback=_silent_cb_out)
    inp = AudioInput(device=2, asio_buffer_size=128, callback=lambda a: None)
    assert out.asio_buffer_size == 512
    assert inp.asio_buffer_size == 128
    # Default is 0 (follow control panel)
    assert AudioOutput(device=1, callback=_silent_cb_out).asio_buffer_size == 0


if __name__ == "__main__":
    tests = [
        test_resolve_snaps_and_prefers,
        test_resolve_without_query,
        test_try_open_asio_rejects_default_device,
        test_try_open_asio_rejects_non_asio_device,
        test_try_open_asio_uses_blocksize0_and_latency,
        test_try_open_asio_auto_uses_latency_hint,
        test_try_open_asio_falls_back_on_open_failure,
        test_streams_store_asio_buffer_size,
    ]
    passed = 0
    for t in tests:
        t()
        print(f"OK: {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} passed")
