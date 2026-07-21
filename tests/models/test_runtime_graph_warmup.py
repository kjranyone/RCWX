"""Tests for steady-state real-time graph warmup."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from rcwx.pipeline.realtime_unified import RealtimeVoiceChangerUnified


class _FakePipeline:
    def __init__(self, history_limit: int) -> None:
        self.history_limit = history_limit
        self.device = "cpu"
        self.synthesizer = None
        self.accelerator_index = None
        self._streaming_audio_history = None
        self.infer_calls = 0
        self.clear_calls = 0
        self.prepare_index_calls = 0

    def prepare_accelerator_index(self) -> bool:
        self.prepare_index_calls += 1
        self.accelerator_index = object()
        return True

    def infer_streaming(self, chunk, overlap, params):
        del params
        new_hop = chunk[overlap:]
        if self._streaming_audio_history is None:
            history = chunk.copy()
        else:
            history = np.concatenate([self._streaming_audio_history, new_hop])
        self._streaming_audio_history = history[-self.history_limit :]
        self.infer_calls += 1
        return np.zeros(len(new_hop) * 2, dtype=np.float32)

    def clear_cache(self) -> None:
        self._streaming_audio_history = None
        self.clear_calls += 1


class _FakeResampler:
    def __init__(self) -> None:
        self.process_calls = 0
        self.reset_calls = 0

    def resample_chunk(self, audio):
        self.process_calls += 1
        return audio

    def reset(self) -> None:
        self.reset_calls += 1


def test_runtime_warmup_reaches_full_hubert_history() -> None:
    changer = RealtimeVoiceChangerUnified.__new__(RealtimeVoiceChangerUnified)
    changer.config = SimpleNamespace(hubert_context_sec=1.0)
    changer._hop_samples_16k = 3840
    changer._overlap_samples_16k = 3200
    changer.pipeline = _FakePipeline(history_limit=16000)
    changer.input_resampler = _FakeResampler()
    changer.output_resampler = _FakeResampler()
    changer._sola_state = SimpleNamespace(buffer=np.ones(4, dtype=np.float32))
    changer._overlap_buf = np.ones(4, dtype=np.float32)
    changer._reset_boundary_continuity_state = lambda: None
    changer._build_streaming_params = lambda **kwargs: kwargs

    changer._run_runtime_warmup()

    assert changer.pipeline.infer_calls == 4
    assert changer.output_resampler.process_calls == 4
    assert changer.pipeline.clear_calls == 1
    assert changer.pipeline._streaming_audio_history is None
    assert changer.input_resampler.reset_calls == 1
    assert changer.output_resampler.reset_calls == 1
    assert changer._sola_state.buffer is None
    assert changer._overlap_buf is None
    assert changer.pipeline.prepare_index_calls == 0


def test_sub100_runtime_warmup_prepares_accelerator_index() -> None:
    changer = RealtimeVoiceChangerUnified.__new__(RealtimeVoiceChangerUnified)
    changer.config = SimpleNamespace(
        hubert_context_sec=0.56,
        f0_context_sec=0.10,
        f0_method="swiftf0",
        latency_mode="sub100",
        index_rate=0.45,
    )
    changer._hop_samples_16k = 640
    changer._overlap_samples_16k = 960
    changer.pipeline = _FakePipeline(history_limit=8960)
    changer.input_resampler = _FakeResampler()
    changer.output_resampler = _FakeResampler()
    changer._sola_state = SimpleNamespace(buffer=None)
    changer._overlap_buf = None
    changer._reset_boundary_continuity_state = lambda: None
    changer._build_streaming_params = lambda **kwargs: kwargs

    changer._run_runtime_warmup()

    assert changer.pipeline.prepare_index_calls == 1
    assert changer.pipeline.infer_calls == 13
