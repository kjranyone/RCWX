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
    assert _minimum_chunk_ms("swiftf0", "aggressive") == 20
    assert _minimum_chunk_ms("none", "aggressive") == 20
    assert _minimum_chunk_ms("fcpe", "aggressive") == 100

    assert RealtimeConfig(chunk_sec=0.01, f0_method="swiftf0").chunk_sec == 0.04
    assert RealtimeConfig(chunk_sec=0.04, f0_method="fcpe").chunk_sec == 0.10
    assert RealtimeConfig(chunk_sec=0.04, f0_method="rmvpe").chunk_sec == 0.32
    aggressive_cfg = RealtimeConfig(
        latency_mode="aggressive",
        chunk_sec=0.01,
        f0_method="swiftf0",
        prebuffer_chunks=1,
    )
    assert aggressive_cfg.chunk_sec == 0.02
    assert aggressive_cfg.prebuffer_chunks == 3

    normal = _auto_params(0.04, "normal")
    assert normal["crossfade_sec"] == 0.01
    assert normal["buffer_margin"] == 0.25
    assert normal["latency_mode"] == "normal"
    assert normal["prebuffer_chunks"] == 1

    aggressive = _auto_params(0.02, "aggressive")
    assert aggressive["crossfade_sec"] == 0.01
    assert aggressive["latency_mode"] == "aggressive"
    assert aggressive["prebuffer_chunks"] == 3


def test_aggressive_slider_uses_compact_effective_range() -> None:
    assert _chunk_slider_spec("swiftf0", "aggressive") == (20, 100, 20)
    assert _chunk_slider_spec("none", "aggressive") == (20, 100, 20)
    assert _chunk_slider_spec("fcpe", "aggressive") == (100, 600, 20)
    assert _chunk_slider_spec("swiftf0", "normal") == (40, 600, 20)


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
        latency_mode="aggressive",
        f0_method="swiftf0",
        hubert_context_sec=1.0,
        f0_context_sec=0.32,
    )
    vc.pipeline = SimpleNamespace(device="xpu")

    assert vc._effective_streaming_contexts() == (0.56, 0.10)
    assert vc._uses_device_output_resample() is True

    vc.config.latency_mode = "normal"
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


def test_aggressive_streaming_params_enable_history_priming() -> None:
    vc = RealtimeVoiceChangerUnified.__new__(RealtimeVoiceChangerUnified)
    vc.config = RealtimeConfig(
        latency_mode="aggressive",
        chunk_sec=0.02,
        f0_method="swiftf0",
    )
    vc.pipeline = SimpleNamespace(device="xpu")
    vc._sola_extra_model = 960
    vc._runtime_output_sample_rate = 48000

    params = vc._build_streaming_params(index_rate=0.0, voice_gate_mode="off")

    assert params.prime_hubert_history is True
