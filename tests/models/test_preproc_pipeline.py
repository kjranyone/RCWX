"""Tests for the EXPERIMENTAL preprocess-thread pipeline
(RCWX_PREPROC_THREAD=1; the default path runs stages 1-3 inline).

When enabled, denoise must run on the dedicated preprocess thread, and the
catch-up drop path must NOT reset the input resampler (whose state stays
continuous because the preprocess thread consumes every mic hop in arrival
order).
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np

import rcwx.pipeline.realtime_unified as ru
from rcwx.pipeline.realtime_config import RealtimeConfig
from rcwx.pipeline.realtime_unified import RealtimeVoiceChangerUnified


class _FakePipeline:
    _loaded = True
    device = "cpu"
    sample_rate = 16000
    synthesizer = None
    accelerator_index = None
    stage_times: dict = {}

    def __init__(self) -> None:
        self.infer_threads: set[str] = set()
        self.clear_calls = 0

    def infer_streaming(self, chunk, overlap, params):
        self.infer_threads.add(threading.current_thread().name)
        return np.zeros(len(chunk) - overlap, dtype=np.float32)

    def clear_cache(self) -> None:
        self.clear_calls += 1


def _make_changer() -> RealtimeVoiceChangerUnified:
    config = RealtimeConfig(
        mic_sample_rate=16000,
        output_sample_rate=16000,
        chunk_sec=0.02,
        latency_mode="normal",
        denoise_enabled=True,
        denoise_method="spectral",
        use_sola=False,
    )
    return RealtimeVoiceChangerUnified(_FakePipeline(), config)


def test_denoise_runs_on_preproc_thread(monkeypatch) -> None:
    changer = _make_changer()
    changer._use_preproc_thread = True

    denoise_threads: set[str] = set()

    def spy_denoise(audio, **kwargs):
        denoise_threads.add(threading.current_thread().name)
        return audio

    monkeypatch.setattr(ru, "denoise_audio", spy_denoise)

    changer._running = True
    preproc = threading.Thread(
        target=changer._preproc_thread_loop, name="RCWX-Preproc", daemon=True
    )
    infer = threading.Thread(
        target=changer._inference_thread, name="RCWX-Inference-Unified", daemon=True
    )
    preproc.start()
    infer.start()
    try:
        hop = changer._hop_samples_mic
        for _ in range(6):
            changer.process_input_chunk(
                np.random.RandomState(0).randn(hop).astype(np.float32) * 0.01
            )
            time.sleep(0.005)

        deadline = time.time() + 5.0
        while changer._output_queue.qsize() < 3 and time.time() < deadline:
            time.sleep(0.01)
    finally:
        changer._running = False
        preproc.join(timeout=2.0)
        infer.join(timeout=2.0)

    assert changer._output_queue.qsize() >= 3, "no output produced through the pipeline"
    assert denoise_threads == {"RCWX-Preproc"}, denoise_threads
    assert changer.pipeline.infer_threads == {"RCWX-Inference-Unified"}


def test_catchup_drop_keeps_input_resampler_state() -> None:
    changer = RealtimeVoiceChangerUnified.__new__(RealtimeVoiceChangerUnified)
    changer._use_preproc_thread = True
    changer._preproc_queue = ru.Queue()
    for i in range(3):
        changer._preproc_queue.put(
            (np.full(4, i, dtype=np.float32), np.full(4, i, dtype=np.float32))
        )
    changer.pipeline = SimpleNamespace(clear_cache=lambda: None)
    changer._overlap_buf = np.ones(4, dtype=np.float32)
    changer._sola_state = SimpleNamespace(buffer=np.ones(4, dtype=np.float32))
    changer._reset_boundary_continuity_state = lambda: None

    input_resets = []
    output_resets = []
    changer.input_resampler = SimpleNamespace(reset=lambda: input_resets.append(1))
    changer.output_resampler = SimpleNamespace(reset=lambda: output_resets.append(1))

    hop_16k, hop_mic = changer._next_input_hop()

    # Newest hop wins; boundary state reset, but the input resampler state is
    # preserved (it consumed every mic hop in order on the preproc thread).
    assert float(hop_16k[0]) == 2.0
    assert changer._overlap_buf is None
    assert changer._sola_state.buffer is None
    assert output_resets and not input_resets
