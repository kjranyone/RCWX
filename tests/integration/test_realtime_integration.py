"""
Real-time Integration Test

GUIと同等のスレッド構成でリアルタイム処理をテストする。
SimulatedAudioDeviceで実デバイスのタイミングをエミュレート。

Usage:
    uv run python tests/test_realtime_integration.py
    uv run python tests/test_realtime_integration.py --duration 30
    uv run python tests/test_realtime_integration.py --stress
"""
from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from rcwx.audio.resample import resample
from rcwx.config import RCWXConfig
from rcwx.pipeline.inference import RVCPipeline
from rcwx.pipeline.realtime import RealtimeConfig, RealtimeVoiceChanger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(threadName)s]: %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class IntegrationTestResult:
    """統合テスト結果"""
    duration_sec: float
    total_input_samples: int
    total_output_samples: int
    chunks_processed: int
    underruns: int = 0
    overruns: int = 0
    max_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    discontinuities: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        # Allow some underruns since RMVPE is slow
        return self.overruns < 5 and len(self.errors) == 0 and self.chunks_processed > 0

    @property
    def realtime_capable(self) -> bool:
        """True if processing keeps up with real-time."""
        return self.underruns == 0


class SimulatedAudioDevice:
    """実デバイスのタイミングをシミュレート"""

    def __init__(
        self,
        input_audio: np.ndarray,
        sample_rate: int = 48000,
        block_size: int = 1024,
        simulate_jitter: bool = True,
    ):
        self.input_audio = input_audio
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.simulate_jitter = simulate_jitter
        self.input_pos = 0
        self.output_buffer: list[np.ndarray] = []
        self.block_duration = block_size / sample_rate
        self._running = False

    def start(self, input_callback, output_callback):
        self._input_cb = input_callback
        self._output_cb = output_callback
        self._running = True
        self._in_thread = threading.Thread(target=self._input_loop, daemon=True)
        self._out_thread = threading.Thread(target=self._output_loop, daemon=True)
        self._in_thread.start()
        self._out_thread.start()

    def stop(self):
        self._running = False

    def _input_loop(self):
        while self._running and self.input_pos < len(self.input_audio):
            t0 = time.perf_counter()
            end = min(self.input_pos + self.block_size, len(self.input_audio))
            block = self.input_audio[self.input_pos:end]
            if len(block) < self.block_size:
                block = np.pad(block, (0, self.block_size - len(block)))
            self.input_pos = end
            if self._input_cb:
                self._input_cb(block)
            sleep = self.block_duration - (time.perf_counter() - t0)
            if self.simulate_jitter:
                sleep += np.random.uniform(-0.1, 0.1) * self.block_duration
            if sleep > 0:
                time.sleep(sleep)
        self._running = False

    def _output_loop(self):
        while self._running:
            t0 = time.perf_counter()
            if self._output_cb:
                block = self._output_cb(self.block_size)
                if block is not None and len(block) > 0:
                    self.output_buffer.append(block.copy())
            sleep = self.block_duration - (time.perf_counter() - t0)
            if sleep > 0:
                time.sleep(sleep)

    def get_output(self) -> np.ndarray:
        if self.output_buffer:
            return np.concatenate(self.output_buffer)
        return np.array([], dtype=np.float32)


def run_integration_test(
    pipeline: RVCPipeline,
    input_audio: np.ndarray,
    rt_config: RealtimeConfig,
    stress_mode: bool = False,
) -> IntegrationTestResult:
    """統合テスト実行"""
    duration_sec = len(input_audio) / rt_config.mic_sample_rate
    result = IntegrationTestResult(
        duration_sec=duration_sec,
        total_input_samples=len(input_audio),
        total_output_samples=0,
        chunks_processed=0,
    )

    changer = RealtimeVoiceChanger(pipeline, config=rt_config)
    pipeline.clear_cache()

    chunk_times: list[float] = []
    underrun_count = 0
    overrun_count = 0
    lock = threading.Lock()

    def input_callback(block: np.ndarray):
        nonlocal overrun_count
        try:
            changer.process_input_chunk(block)
        except Exception:
            with lock:
                overrun_count += 1

    def output_callback(requested_samples: int) -> Optional[np.ndarray]:
        nonlocal underrun_count
        if changer.output_buffer.available < requested_samples:
            with lock:
                underrun_count += 1
            return np.zeros(requested_samples, dtype=np.float32)
        return changer.get_output_chunk(requested_samples)

    inference_running = True
    inference_errors: list[str] = []

    chunks_count = [0]  # Use list for mutable reference in closure

    def inference_loop():
        nonlocal inference_running
        while inference_running:
            try:
                t0 = time.perf_counter()
                if stress_mode:
                    # 5-15ms の人工的な負荷
                    end = time.perf_counter() + np.random.uniform(0.005, 0.015)
                    while time.perf_counter() < end:
                        _ = np.random.randn(1000).sum()
                processed = changer.process_next_chunk()
                if processed:
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    with lock:
                        chunk_times.append(elapsed_ms)
                        chunks_count[0] += 1
                else:
                    time.sleep(0.001)
            except Exception as e:
                inference_errors.append(str(e))
                logger.error(f"Inference error: {e}")

    infer_thread = threading.Thread(target=inference_loop, daemon=True)
    infer_thread.start()
    changer._recalculate_buffers()
    changer._running = True

    device = SimulatedAudioDevice(input_audio, rt_config.mic_sample_rate)
    logger.info(f"Starting integration test: {duration_sec:.1f}s")
    device.start(input_callback, output_callback)

    while device._running:
        time.sleep(0.1)

    inference_running = False
    infer_thread.join(timeout=2.0)
    changer.flush_final_sola_buffer()
    device.stop()

    output_audio = device.get_output()
    result.total_output_samples = len(output_audio)
    result.underruns = underrun_count
    result.overruns = overrun_count
    result.errors = inference_errors
    result.chunks_processed = chunks_count[0]

    if chunk_times:
        result.avg_latency_ms = float(np.mean(chunk_times))
        result.max_latency_ms = float(np.max(chunk_times))

    if len(output_audio) > 1:
        diff = np.abs(np.diff(output_audio))
        result.discontinuities = int(np.sum(diff > 0.3))

    return result


def load_audio(path: Path, target_sr: int = 48000) -> np.ndarray:
    sr, audio = wavfile.read(path)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    if sr != target_sr:
        audio = resample(audio, sr, target_sr)
    return audio.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Real-time Integration Test")
    parser.add_argument("--test-file", type=Path, default=Path("sample_data/sustained_voice.wav"))
    parser.add_argument("--duration", type=float, default=10.0, help="Test duration (sec)")
    parser.add_argument("--stress", action="store_true", help="Add CPU load")
    parser.add_argument("--chunk-sec", type=float, default=0.35)
    parser.add_argument("--chunking-mode", choices=["wokada", "rvc_webui", "hybrid"], default="wokada")
    args = parser.parse_args()

    config = RCWXConfig.load()
    if not config.last_model_path:
        logger.error("No model configured. Run GUI first.")
        sys.exit(1)

    logger.info(f"Loading model: {config.last_model_path}")
    pipeline = RVCPipeline(config.last_model_path, device=config.device, use_compile=False)
    pipeline.load()

    if args.test_file.exists():
        audio = load_audio(args.test_file, target_sr=48000)
    else:
        logger.info("Generating 440Hz sine wave")
        t = np.linspace(0, args.duration, int(48000 * args.duration), dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    # Adjust duration
    target_samples = int(48000 * args.duration)
    if len(audio) < target_samples:
        repeats = (target_samples // len(audio)) + 1
        audio = np.tile(audio, repeats)[:target_samples]
    else:
        audio = audio[:target_samples]

    rt_config = RealtimeConfig(
        mic_sample_rate=48000,
        output_sample_rate=48000,
        chunk_sec=args.chunk_sec,
        context_sec=0.10,
        crossfade_sec=0.05,
        use_sola=True,
        prebuffer_chunks=1,
        pitch_shift=0,
        use_f0=True,
        f0_method="fcpe",
        chunking_mode=args.chunking_mode,
        index_rate=0.0,
        voice_gate_mode="off",
        use_feature_cache=True,
        use_parallel_extraction=True,
        denoise_enabled=False,
    )

    result = run_integration_test(pipeline, audio, rt_config, stress_mode=args.stress)

    logger.info("=" * 60)
    logger.info("INTEGRATION TEST RESULT")
    logger.info("=" * 60)
    logger.info(f"Duration: {result.duration_sec:.1f}s")
    logger.info(f"Chunks processed: {result.chunks_processed}")
    logger.info(f"Underruns: {result.underruns}")
    logger.info(f"Overruns: {result.overruns}")
    logger.info(f"Avg latency: {result.avg_latency_ms:.1f}ms")
    logger.info(f"Max latency: {result.max_latency_ms:.1f}ms")
    logger.info(f"Discontinuities: {result.discontinuities}")

    if result.realtime_capable:
        logger.info("Realtime: YES (no underruns)")
    else:
        logger.warning(f"Realtime: NO ({result.underruns} underruns - consider using FCPE)")

    if result.passed:
        logger.info("PASSED")
    else:
        logger.error("FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
