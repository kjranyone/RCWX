"""Regression tests for micro-hop streaming hot-path policies."""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

from rcwx.gui.widgets.latency_settings import _auto_params, _minimum_chunk_ms
from rcwx.pipeline.realtime_unified import (
    RealtimeConfig,
    RealtimeStats,
    RealtimeVoiceChangerUnified,
)


def test_f0_backend_micro_hop_floors() -> None:
    assert _minimum_chunk_ms("swiftf0") == 40
    assert _minimum_chunk_ms("none") == 40
    assert _minimum_chunk_ms("fcpe") == 100
    assert _minimum_chunk_ms("rmvpe") == 320

    assert RealtimeConfig(chunk_sec=0.01, f0_method="swiftf0").chunk_sec == 0.04
    assert RealtimeConfig(chunk_sec=0.04, f0_method="fcpe").chunk_sec == 0.10
    assert RealtimeConfig(chunk_sec=0.04, f0_method="rmvpe").chunk_sec == 0.32

    sub100 = _auto_params(0.04, "sub100")
    assert sub100["crossfade_sec"] == 0.01
    assert sub100["buffer_margin"] == 0.1
    assert sub100["latency_mode"] == "sub100"


def test_deadline_statistics_track_micro_hop_tail() -> None:
    vc = RealtimeVoiceChangerUnified.__new__(RealtimeVoiceChangerUnified)
    vc.config = SimpleNamespace(chunk_sec=0.04)
    vc.stats = RealtimeStats()
    vc._inference_times = deque(maxlen=256)

    for value in (15.0, 20.0, 25.0, 35.0, 45.0):
        vc._record_inference_timing(value)
        vc.stats.frames_processed += 1

    assert vc.stats.deadline_misses == 1
    assert vc.stats.deadline_miss_rate == 0.2
    assert vc.stats.inference_p50_ms == 25.0
    assert vc.stats.inference_p95_ms > 40.0
