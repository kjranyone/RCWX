"""Overload protection behavior for RealtimeVoiceChangerV2."""

from __future__ import annotations

import time

import numpy as np

from rcwx.pipeline.realtime import RealtimeConfig
from rcwx.pipeline.realtime_v2 import RealtimeVoiceChangerV2


class _DummyPipeline:
    sample_rate = 16000

    def clear_cache(self) -> None:
        return None

    def infer(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        # Passthrough inference
        return audio.astype(np.float32)


def test_overload_protection_triggers() -> None:
    config = RealtimeConfig(
        mic_sample_rate=16000,
        input_sample_rate=16000,
        output_sample_rate=16000,
        chunk_sec=0.1,
        crossfade_sec=0.02,
        context_sec=0.0,
        lookahead_sec=0.0,
        use_sola=False,
        denoise_enabled=False,
        max_queue_size=1,
        chunking_mode="wokada",
    )
    changer = RealtimeVoiceChangerV2(_DummyPipeline(), config=config)

    chunk_samples = int(config.mic_sample_rate * config.chunk_sec)
    chunk = np.zeros(chunk_samples, dtype=np.float32)

    # Rapidly push chunks to overflow the input queue.
    for _ in range(6):
        changer.process_input_chunk(chunk)

    # Overload should trigger after multiple drops within 1 second.
    assert changer._is_overloaded() is True

    # Wait for overload window to clear.
    time.sleep(2.2)
    assert changer._is_overloaded() is False


if __name__ == "__main__":
    test_overload_protection_triggers()
    print("OK")
