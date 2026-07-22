"""Regression tests for work kept off the micro-hop hot path."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from rcwx.pipeline.inference import (
    RVCPipeline,
    _highpass_coefficients,
    highpass_filter,
)
from rcwx.pipeline.realtime_unified import RealtimeVoiceChangerUnified


def test_highpass_coefficients_are_reused() -> None:
    _highpass_coefficients.cache_clear()
    audio = np.linspace(-0.5, 0.5, 320, dtype=np.float32)

    first = highpass_filter(audio)
    second = highpass_filter(audio)

    assert np.array_equal(first, second)
    assert _highpass_coefficients.cache_info().misses == 1
    assert _highpass_coefficients.cache_info().hits == 1


def test_parallel_extraction_executor_is_reused_and_released() -> None:
    pipeline = RVCPipeline("unused.pth", device="cpu", use_compile=False)

    first = pipeline._get_parallel_executor()
    second = pipeline._get_parallel_executor()

    assert first is second
    pipeline.unload()
    assert pipeline._parallel_executor is None


def test_aggressive_publishes_telemetry_at_10hz() -> None:
    changer = RealtimeVoiceChangerUnified.__new__(RealtimeVoiceChangerUnified)
    changer.config = SimpleNamespace(latency_mode="aggressive")
    changer.stats = SimpleNamespace(frames_processed=1)

    assert changer._should_publish_stats() is True
    changer.stats.frames_processed = 2
    assert changer._should_publish_stats() is False
    changer.stats.frames_processed = 5
    assert changer._should_publish_stats() is True

    changer.config.latency_mode = "normal"
    changer.stats.frames_processed = 2
    assert changer._should_publish_stats() is True
