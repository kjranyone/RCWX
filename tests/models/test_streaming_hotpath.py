"""Regression tests for micro-hop streaming hot-path policies."""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

from rcwx.gui.widgets.latency_settings import (
    _auto_params,
    _chunk_slider_spec,
    _minimum_chunk_ms,
)
from rcwx.pipeline.inference import _initial_streaming_history
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
    assert _minimum_chunk_ms("swiftf0", "frontier") == 20
    assert _minimum_chunk_ms("none", "frontier") == 20
    assert _minimum_chunk_ms("fcpe", "frontier") == 100

    assert RealtimeConfig(chunk_sec=0.01, f0_method="swiftf0").chunk_sec == 0.04
    assert RealtimeConfig(chunk_sec=0.04, f0_method="fcpe").chunk_sec == 0.10
    assert RealtimeConfig(chunk_sec=0.04, f0_method="rmvpe").chunk_sec == 0.32
    assert RealtimeConfig(latency_mode="sub100", prebuffer_chunks=1).prebuffer_chunks == 2
    frontier_cfg = RealtimeConfig(
        latency_mode="frontier",
        chunk_sec=0.01,
        f0_method="swiftf0",
        prebuffer_chunks=1,
    )
    assert frontier_cfg.chunk_sec == 0.02
    assert frontier_cfg.prebuffer_chunks == 3

    sub100 = _auto_params(0.04, "sub100")
    assert sub100["crossfade_sec"] == 0.01
    assert sub100["buffer_margin"] == 0.1
    assert sub100["latency_mode"] == "sub100"
    assert sub100["prebuffer_chunks"] == 2

    frontier = _auto_params(0.02, "frontier")
    assert frontier["crossfade_sec"] == 0.01
    assert frontier["latency_mode"] == "frontier"
    assert frontier["prebuffer_chunks"] == 3


def test_frontier_slider_uses_compact_effective_range() -> None:
    assert _chunk_slider_spec("swiftf0", "frontier") == (20, 100, 20)
    assert _chunk_slider_spec("none", "frontier") == (20, 100, 20)
    assert _chunk_slider_spec("fcpe", "frontier") == (100, 600, 20)
    assert _chunk_slider_spec("swiftf0", "balanced") == (40, 600, 20)


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


def test_deadline_modes_use_short_context_and_device_output_resample() -> None:
    vc = RealtimeVoiceChangerUnified.__new__(RealtimeVoiceChangerUnified)
    vc.config = SimpleNamespace(
        latency_mode="sub100",
        f0_method="swiftf0",
        hubert_context_sec=1.0,
        f0_context_sec=0.32,
    )
    vc.pipeline = SimpleNamespace(device="xpu")

    assert vc._effective_streaming_contexts() == (0.56, 0.10)
    assert vc._uses_device_output_resample() is True

    vc.config.latency_mode = "frontier"
    assert vc._effective_streaming_contexts() == (0.56, 0.10)
    assert vc._uses_device_output_resample() is True

    vc.config.latency_mode = "aggressive"
    assert vc._effective_streaming_contexts() == (1.0, 0.32)
    assert vc._uses_device_output_resample() is False


def test_deadline_history_priming_reaches_fixed_shape_immediately() -> None:
    import numpy as np

    audio = np.arange(640, dtype=np.float32)
    primed = _initial_streaming_history(audio, 8960, prime=True)
    normal = _initial_streaming_history(audio, 8960, prime=False)

    assert primed.shape == (8960,)
    assert np.array_equal(primed[-len(audio):], audio)
    assert np.array_equal(normal, audio)


def test_frontier_streaming_params_enable_history_priming() -> None:
    vc = RealtimeVoiceChangerUnified.__new__(RealtimeVoiceChangerUnified)
    vc.config = RealtimeConfig(
        latency_mode="frontier",
        chunk_sec=0.02,
        f0_method="swiftf0",
    )
    vc.pipeline = SimpleNamespace(device="xpu")
    vc._sola_extra_model = 960
    vc._runtime_output_sample_rate = 48000

    params = vc._build_streaming_params(index_rate=0.0, voice_gate_mode="off")

    assert params.prime_hubert_history is True
