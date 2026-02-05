"""Real-time voice conversion pipeline (V2 reimplementation)."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Callable, Optional

import numpy as np

from rcwx.audio.analysis import AdaptiveParameterCalculator
from rcwx.audio.buffer import OutputBuffer
from rcwx.audio.chunking import ChunkConfig, create_chunking_strategy
from rcwx.audio.crossfade import SOLAState, apply_sola_crossfade, flush_sola_buffer
from rcwx.audio.crossfade_strategy import CrossfadeConfig, create_crossfade_strategy
from rcwx.audio.denoise import denoise as denoise_audio
from rcwx.audio.input import AudioInput
from rcwx.audio.output import AudioOutput
from rcwx.audio.resample import StatefulResampler, resample
from rcwx.pipeline.inference import RVCPipeline
from rcwx.pipeline.realtime import RealtimeConfig, RealtimeStats

logger = logging.getLogger(__name__)


@dataclass
class _PreparedChunk:
    chunk_mic: np.ndarray
    chunk_model: np.ndarray
    input_rms: float
    input_peak: float


class RealtimeVoiceChangerV2:
    """
    Reimplemented real-time voice changer.

    Goals:
    - Explicit stages with clear contracts
    - Deterministic chunk length handling per mode
    - Keep legacy behavior for comparability
    """

    def __init__(
        self,
        pipeline: RVCPipeline,
        config: Optional[RealtimeConfig] = None,
        on_warmup_progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> None:
        self.pipeline = pipeline
        self.config = config or RealtimeConfig()
        self.on_warmup_progress = on_warmup_progress

        self.stats = RealtimeStats()
        self.on_stats_update: Optional[Callable[[RealtimeStats], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._input_queue: Queue = Queue(maxsize=self.config.max_queue_size)
        self._output_queue: Queue = Queue(maxsize=self.config.max_queue_size)

        self._chunking_strategy = self._create_chunking_strategy()
        self._crossfade_strategy = self._create_crossfade_strategy()
        self._sola_state = getattr(self._crossfade_strategy, "_sola_state", None)

        self.mic_chunk_samples = self._chunking_strategy.chunk_samples
        self.mic_hop_samples = self._chunking_strategy.hop_samples
        self.mic_context_samples = self._chunking_strategy.context_samples
        self.mic_lookahead_samples = self._chunking_strategy.lookahead_samples
        self.output_crossfade_samples = int(
            self.config.output_sample_rate * self.config.crossfade_sec
        )

        max_latency_sec = self.config.chunk_sec * (
            self.config.prebuffer_chunks + self.config.buffer_margin
        )
        self.output_buffer = OutputBuffer(
            max_latency_samples=int(self.config.output_sample_rate * max_latency_sec),
            fade_samples=256,
        )

        self.input_resampler = StatefulResampler(
            self.config.mic_sample_rate,
            self.config.input_sample_rate,
        )
        self.output_resampler = StatefulResampler(
            self.pipeline.sample_rate,
            self.config.output_sample_rate,
        )

        self._input_stream: Optional[AudioInput] = None
        self._output_stream: Optional[AudioOutput] = None

        self._prebuffer_chunks = self.config.prebuffer_chunks
        self._chunks_ready = 0
        self._output_started = False

        # Feedback detection state
        self._output_history_size = self.config.output_sample_rate
        self._output_history = np.zeros(self._output_history_size, dtype=np.float32)
        self._output_history_pos = 0
        self._feedback_check_interval = 10
        self._feedback_warning_shown = False

        # Energy/peak normalization state
        self._input_rms_ema = 0.0
        self._input_peak_ema = 0.0
        self._energy_ratio_ema = 1.0
        self._output_rms_target = 0.0
        # F0 stability tracking (for adaptive SOLA fallback)
        self._f0_ema = 0.0
        self._f0_dev_ema = 0.0

        # Adaptive parameters
        self.adaptive_calc = (
            AdaptiveParameterCalculator(
                sample_rate=self.config.mic_sample_rate,
                base_crossfade_sec=self.config.crossfade_sec,
                base_context_sec=self.config.context_sec,
                base_sola_search_ms=30.0,
            )
            if self.config.use_adaptive_parameters
            else None
        )

        # Ensure SOLA state uses the same minimum buffer as V1
        self._ensure_sola_state()

    # ======== Lifecycle ========
    def start(self) -> None:
        if self._running:
            return

        self.pipeline.clear_cache()
        self._recalculate_buffers()
        self.stats.reset()
        self._chunking_strategy.clear()
        self._crossfade_strategy.reset()
        self.output_buffer.clear()
        self._chunks_ready = 0
        self._output_started = False
        self._ensure_sola_state()

        self._output_history.fill(0)
        self._output_history_pos = 0
        self._feedback_warning_shown = False

        self._clear_queues()

        self._running = True
        self._thread = threading.Thread(
            target=self._inference_thread,
            daemon=True,
            name="RCWX-Inference-V2",
        )
        self._thread.start()

        output_chunk_sec = self.config.chunk_sec / 4
        output_blocksize = int(self.config.output_sample_rate * output_chunk_sec)

        self._input_stream = AudioInput(
            device=self.config.input_device,
            sample_rate=self.config.mic_sample_rate,
            channels=self.config.input_channels,
            blocksize=int(self.config.mic_sample_rate * output_chunk_sec),
            callback=self._on_audio_input,
            channel_selection=self.config.input_channel_selection,
        )
        self._output_stream = AudioOutput(
            device=self.config.output_device,
            sample_rate=self.config.output_sample_rate,
            channels=self.config.output_channels,
            blocksize=output_blocksize,
            callback=self._on_audio_output,
        )

        self._input_stream.start()
        self._output_stream.start()

    def stop(self) -> None:
        if not self._running:
            return

        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        if self._input_stream:
            self._input_stream.stop()
            self._input_stream = None
        if self._output_stream:
            self._output_stream.stop()
            self._output_stream = None

        self._clear_queues()

    @property
    def is_running(self) -> bool:
        return self._running

    # ======== Configuration Mutators ========
    def set_pitch_shift(self, pitch: int) -> None:
        self.config.pitch_shift = pitch

    def set_f0_mode(self, enabled: bool) -> None:
        self.config.use_f0 = enabled

    def set_f0_method(self, method: str) -> None:
        self.config.f0_method = method

    def set_index_rate(self, rate: float) -> None:
        self.config.index_rate = rate

    def set_denoise(self, enabled: bool, method: str = "auto") -> None:
        self.config.denoise_enabled = enabled
        self.config.denoise_method = method

    def set_voice_gate_mode(self, mode: str) -> None:
        self.config.voice_gate_mode = mode

    def set_energy_threshold(self, value: float) -> None:
        self.config.energy_threshold = value

    def set_feature_cache(self, enabled: bool) -> None:
        self.config.use_feature_cache = enabled

    def set_chunk_sec(self, chunk_sec: float) -> None:
        old_chunk = self.config.chunk_sec
        self.config.chunk_sec = max(0.1, min(0.6, chunk_sec))
        if self._running:
            logger.info(
                f"Chunk size changed ({old_chunk}s -> {self.config.chunk_sec}s), "
                "restarting audio streams..."
            )
            self.stop()
            self.start()

    def set_prebuffer_chunks(self, chunks: int) -> None:
        self.config.prebuffer_chunks = max(0, min(3, int(chunks)))
        self._prebuffer_chunks = self.config.prebuffer_chunks

    def set_buffer_margin(self, margin: float) -> None:
        self.config.buffer_margin = max(0.1, min(2.0, float(margin)))

    def set_context(self, context_sec: float) -> None:
        self.config.context_sec = max(0.0, float(context_sec))

    def set_lookahead(self, lookahead_sec: float) -> None:
        self.config.lookahead_sec = max(0.0, float(lookahead_sec))

    def set_crossfade(self, crossfade_sec: float) -> None:
        if self.config.chunking_mode == "rvc_webui":
            max_crossfade = max(0.0, self.config.chunk_sec * 0.5)
        else:
            max_crossfade = max(0.0, min(self.config.context_sec, self.config.chunk_sec * 0.5))
        self.config.crossfade_sec = max(0.0, min(max_crossfade, crossfade_sec))
        self.output_crossfade_samples = int(
            self.config.output_sample_rate * self.config.crossfade_sec
        )
        self._ensure_sola_state()

    def set_sola(self, enabled: bool) -> None:
        self.config.use_sola = enabled

    # ======== Audio Callbacks ========
    def _on_audio_input(self, audio: np.ndarray) -> None:
        self.process_input_chunk(audio)

    def _on_audio_output(self, frames: int) -> np.ndarray:
        return self.get_output_chunk(frames)

    # ======== Core Processing ========
    def _inference_thread(self) -> None:
        logger.info("Inference thread (V2) started")
        while self._running:
            try:
                if not self.process_next_chunk():
                    continue
            except Empty:
                continue
            except Exception as e:
                logger.error(f"V2 inference error: {e}", exc_info=True)
                if self.on_error:
                    self.on_error(f"推論エラー: {e}")

        if self.config.use_sola:
            final_buffer = self._crossfade_strategy.flush()
            if len(final_buffer) > 0:
                try:
                    self._output_queue.put(final_buffer, timeout=0.5)
                    logger.info(f"Flushed final SOLA buffer: {len(final_buffer)} samples")
                except Exception:
                    logger.warning("Failed to flush final SOLA buffer (queue full)")

        logger.info("Inference thread (V2) stopped")

    def _prepare_input(self, chunk: np.ndarray) -> Optional[_PreparedChunk]:
        if self.config.input_gain_db != 0.0:
            gain_linear = 10 ** (self.config.input_gain_db / 20)
            chunk = chunk * gain_linear

        chunk_at_mic_rate = chunk.copy()
        input_rms = np.sqrt(np.mean(chunk**2)) if len(chunk) else 0.0
        input_peak = float(np.max(np.abs(chunk))) if len(chunk) else 0.0

        if self.config.use_energy_normalization and input_rms > 1e-6:
            alpha = self.config.energy_smoothing
            if self._input_rms_ema < 1e-6:
                self._input_rms_ema = input_rms
            else:
                self._input_rms_ema = alpha * self._input_rms_ema + (1 - alpha) * input_rms

        if self.config.use_peak_normalization and input_peak > 1e-6:
            alpha = self.config.peak_smoothing
            if self._input_peak_ema < 1e-6:
                self._input_peak_ema = input_peak
            else:
                self._input_peak_ema = alpha * self._input_peak_ema + (1 - alpha) * input_peak

        min_required = (
            self.mic_chunk_samples
            if self.config.chunking_mode == "rvc_webui"
            else self.mic_chunk_samples + self.mic_context_samples
        )
        if len(chunk) < min_required:
            logger.warning(
                f"Chunk too short: {len(chunk)} < {min_required} (mode={self.config.chunking_mode})"
            )
            return None

        if self.config.mic_sample_rate != self.config.input_sample_rate:
            chunk = self.input_resampler.resample_chunk(chunk)

        if self.config.denoise_enabled:
            chunk = denoise_audio(
                chunk,
                sample_rate=self.config.input_sample_rate,
                method=self.config.denoise_method,
                device="cpu",
            )

        if self.config.use_sola and self._sola_state is not None:
            f0_hz = self._estimate_f0_hz(chunk)
            if f0_hz > 0:
                self._sola_state.last_f0_hz = f0_hz
            self._update_sola_fallback(f0_hz)

        return _PreparedChunk(
            chunk_mic=chunk_at_mic_rate,
            chunk_model=chunk,
            input_rms=input_rms,
            input_peak=input_peak,
        )

    def _infer(self, chunk: np.ndarray) -> np.ndarray:
        pad_mode = "none" if self.config.chunking_mode == "wokada" else "chunk"
        history_sec = self.config.history_sec
        return self.pipeline.infer(
            chunk,
            input_sr=self.config.input_sample_rate,
            pitch_shift=self.config.pitch_shift,
            f0_method=self.config.f0_method if self.config.use_f0 else "none",
            index_rate=self.config.index_rate,
            index_k=self.config.index_k,
            voice_gate_mode=self.config.voice_gate_mode,
            energy_threshold=self.config.energy_threshold,
            use_feature_cache=self.config.use_feature_cache,
            use_parallel_extraction=self.config.use_parallel_extraction,
            allow_short_input=True,
            pad_mode=pad_mode,
            synth_min_frames=self.config.synth_min_frames,
            history_sec=history_sec,
        )

    def _stitch(self, output: np.ndarray, chunk_at_mic_rate: np.ndarray) -> np.ndarray:
        # RVC WebUI: defer to crossfade strategy (handles SOLA + hop trimming)
        if self.config.chunking_mode == "rvc_webui":
            if self.config.use_sola:
                cf_result = self._crossfade_strategy.process(
                    output.astype(np.float32),
                    self.stats.frames_processed,
                )
                output = cf_result.audio
                return output
            # SOLA disabled: manual hop trimming
            hop_sec = self.mic_hop_samples / self.config.mic_sample_rate
            output_hop_samples = int(self.config.output_sample_rate * hop_sec)
            if self.stats.frames_processed == 0:
                first_chunk_sec = self.config.chunk_sec - self.config.rvc_overlap_sec
                first_chunk_samples = int(
                    self.config.output_sample_rate * first_chunk_sec
                )
                output = output[:first_chunk_samples]
            else:
                output = output[:output_hop_samples]
            return output

        if self.config.use_sola and self._sola_state is not None:
            if self.config.chunking_mode == "hybrid":
                context_samples_output = int(
                    self.config.output_sample_rate * self.config.context_sec
                )
                cf_result = apply_sola_crossfade(
                    output,
                    self._sola_state,
                    wokada_mode=True,
                    context_samples=context_samples_output,
                )
                output = cf_result.audio
            else:
                sola_state_to_use = self._sola_state
                context_sec_to_use = self.config.context_sec
                if self.adaptive_calc is not None:
                    adaptive_params = self.adaptive_calc.analyze_and_adjust(
                        chunk_at_mic_rate,
                        f0=None,
                    )
                    adaptive_crossfade_samples = int(
                        self.config.output_sample_rate * adaptive_params["crossfade_sec"]
                    )
                    min_phase8_buffer = int(self.config.output_sample_rate * 0.08)
                    adaptive_buffer_size = max(adaptive_crossfade_samples, min_phase8_buffer)
                    adaptive_sola_state = SOLAState.create(
                        adaptive_buffer_size,
                        self.config.output_sample_rate,
                        use_advanced_sola=self.config.sola_use_advanced,
                        fallback_threshold=self.config.sola_fallback_threshold,
                    )
                    if self._sola_state.sola_buffer is not None:
                        old_size = len(self._sola_state.sola_buffer)
                        new_size = adaptive_buffer_size
                        if old_size == new_size:
                            adaptive_sola_state.sola_buffer = self._sola_state.sola_buffer.copy()
                        elif old_size < new_size:
                            adaptive_sola_state.sola_buffer = np.pad(
                                self._sola_state.sola_buffer,
                                (0, new_size - old_size),
                                mode="constant",
                            )
                        else:
                            adaptive_sola_state.sola_buffer = self._sola_state.sola_buffer[
                                old_size - new_size :
                            ].copy()
                        adaptive_sola_state.frames_processed = self._sola_state.frames_processed
                    adaptive_sola_state.is_first_chunk_boundary = (
                        self._sola_state.is_first_chunk_boundary
                    )
                    sola_state_to_use = adaptive_sola_state
                    context_sec_to_use = adaptive_params["context_sec"]

                context_samples_output = int(
                    self.config.output_sample_rate * context_sec_to_use
                )
                cf_result = apply_sola_crossfade(
                    output,
                    sola_state_to_use,
                    wokada_mode=True,
                    context_samples=context_samples_output,
                )
                output = cf_result.audio

                if self.adaptive_calc is not None and sola_state_to_use.sola_buffer is not None:
                    adaptive_size = len(sola_state_to_use.sola_buffer)
                    main_size = self._sola_state.sola_buffer_frame
                    if adaptive_size == main_size:
                        self._sola_state.sola_buffer = sola_state_to_use.sola_buffer.copy()
                    elif adaptive_size < main_size:
                        self._sola_state.sola_buffer = np.pad(
                            sola_state_to_use.sola_buffer,
                            (0, main_size - adaptive_size),
                            mode="constant",
                        )
                    else:
                        self._sola_state.sola_buffer = sola_state_to_use.sola_buffer[
                            adaptive_size - main_size :
                        ].copy()
                    self._sola_state.frames_processed = sola_state_to_use.frames_processed
                    self._sola_state.is_first_chunk_boundary = (
                        sola_state_to_use.is_first_chunk_boundary
                    )

        # Trim context for non-RVC modes
        if self.stats.frames_processed > 0 and self.config.context_sec > 0:
            context_samples_output = int(
                self.config.output_sample_rate * self.config.context_sec
            )
            if len(output) > context_samples_output:
                output = output[context_samples_output:]

        return output

    def _apply_leveling(self, output: np.ndarray) -> np.ndarray:
        if self.config.use_energy_normalization:
            output_rms = np.sqrt(np.mean(output**2)) if len(output) else 0.0
            if output_rms > 1e-6 and self._input_rms_ema > 1e-6:
                current_ratio = self._input_rms_ema / output_rms
                alpha = self.config.energy_smoothing
                if self._energy_ratio_ema < 0.1:
                    self._energy_ratio_ema = current_ratio
                else:
                    self._energy_ratio_ema = (
                        alpha * self._energy_ratio_ema + (1 - alpha) * current_ratio
                    )
                scale = np.clip(self._energy_ratio_ema, 0.7, 1.4)
                output = output * scale

        if self.config.use_peak_normalization and self._input_peak_ema > 1e-6:
            output_peak = float(np.max(np.abs(output))) if len(output) else 0.0
            if output_peak > 1e-6:
                peak_scale = self._input_peak_ema / output_peak
                output = output * np.clip(peak_scale, 0.5, 1.5)

        if self.config.use_chunk_gain_smoothing:
            output_rms = np.sqrt(np.mean(output**2)) if len(output) else 0.0
            if output_rms > 1e-6:
                alpha = self.config.chunk_gain_smoothing
                if self._output_rms_target < 1e-6:
                    self._output_rms_target = output_rms
                else:
                    self._output_rms_target = (
                        alpha * self._output_rms_target + (1 - alpha) * output_rms
                    )
                gain = self._output_rms_target / output_rms
                output = output * np.clip(gain, 0.9, 1.1)

        return output

    def _postprocess(self, output: np.ndarray, chunk_at_mic_rate: np.ndarray) -> np.ndarray:
        if self.pipeline.sample_rate != self.config.output_sample_rate:
            output = self.output_resampler.resample_chunk(output)

        max_val = np.max(np.abs(output)) if len(output) else 0.0
        if max_val > 1.0:
            output = np.tanh(output)

        output = self._stitch(output, chunk_at_mic_rate)
        output = self._apply_leveling(output)
        self._store_output_history(output)
        self._maybe_check_feedback(chunk_at_mic_rate)
        return output

    def _store_output_history(self, output: np.ndarray) -> None:
        if self.config.output_sample_rate != self.config.mic_sample_rate:
            output = resample(output, self.config.output_sample_rate, self.config.mic_sample_rate)
        out_len = len(output)
        if out_len >= self._output_history_size:
            self._output_history[:] = output[-self._output_history_size :]
            self._output_history_pos = 0
            return
        end_pos = self._output_history_pos + out_len
        if end_pos <= self._output_history_size:
            self._output_history[self._output_history_pos : end_pos] = output
        else:
            first = self._output_history_size - self._output_history_pos
            self._output_history[self._output_history_pos :] = output[:first]
            self._output_history[: out_len - first] = output[first:]
        self._output_history_pos = end_pos % self._output_history_size

    def _check_feedback(self, input_audio: np.ndarray) -> float:
        if len(input_audio) < 1000:
            return 0.0
        input_rms = np.sqrt(np.mean(input_audio**2))
        if input_rms < 0.01:
            return 0.0
        output_rms = np.sqrt(np.mean(self._output_history**2))
        if output_rms < 0.01:
            return 0.0

        input_norm = input_audio - np.mean(input_audio)
        output_norm = self._output_history - np.mean(self._output_history)
        input_std = np.std(input_norm)
        output_std = np.std(output_norm)
        if input_std < 1e-6 or output_std < 1e-6:
            return 0.0

        input_norm = input_norm / input_std
        output_norm = output_norm / output_std
        check_len = min(len(input_audio), self.config.mic_sample_rate // 2)
        corr = np.correlate(input_norm[:check_len], output_norm[:check_len], mode="valid")
        return float(np.max(np.abs(corr)) / check_len)

    def _maybe_check_feedback(self, chunk_at_mic_rate: np.ndarray) -> None:
        if (
            self.stats.frames_processed > 0
            and self.stats.frames_processed % self._feedback_check_interval == 0
        ):
            corr = self._check_feedback(chunk_at_mic_rate)
            self.stats.feedback_correlation = corr
            if corr > 0.3 and not self._feedback_warning_shown:
                self.stats.feedback_detected = True
                self._feedback_warning_shown = True
                logger.warning(
                    f"[FEEDBACK] Detected feedback (corr={corr:.2f})."
                )
                if self.on_error:
                    self.on_error(
                        "フィードバック検出: 入力と出力が接続されている可能性があります。"
                    )

    # ======== Public Testing Methods ========
    def process_input_chunk(self, audio: np.ndarray) -> None:
        self._chunking_strategy.add_input(audio)
        while self._chunking_strategy.has_chunk():
            result = self._chunking_strategy.get_chunk()
            if result is None:
                break
            try:
                self._input_queue.put_nowait(result.chunk)
            except Exception:
                logger.warning("Input queue full, dropping chunk")
                break

    def process_next_chunk(self) -> bool:
        try:
            chunk = self._input_queue.get_nowait()
        except Empty:
            return False

        start_time = time.perf_counter()
        prepared = self._prepare_input(chunk)
        if prepared is None:
            return True

        output = self._infer(prepared.chunk_model)
        output = self._postprocess(output, prepared.chunk_mic)

        self.stats.inference_ms = (time.perf_counter() - start_time) * 1000
        self.stats.frames_processed += 1
        self.stats.latency_ms = (
            self.config.chunk_sec * 1000
            + self.stats.inference_ms
            + (self.output_buffer.available / self.config.output_sample_rate) * 1000
        )

        try:
            self._output_queue.put_nowait(output)
        except Exception:
            logger.warning("Output queue full, dropping chunk")

        if self.on_stats_update:
            self.on_stats_update(self.stats)

        return True

    def flush_final_sola_buffer(self) -> None:
        if self.config.use_sola:
            final_buffer = self._crossfade_strategy.flush()
            if len(final_buffer) > 0:
                try:
                    self._output_queue.put(final_buffer, timeout=0.5)
                    logger.info(f"Flushed final SOLA buffer: {len(final_buffer)} samples")
                except Exception:
                    logger.warning("Failed to flush final SOLA buffer (queue full)")

    def get_output_chunk(self, frames: int) -> np.ndarray:
        try:
            while True:
                audio = self._output_queue.get_nowait()
                self.output_buffer.add(audio)
                self._chunks_ready += 1
        except Empty:
            pass

        if not self._output_started:
            if self._chunks_ready >= self._prebuffer_chunks:
                self._output_started = True
            else:
                return np.zeros(frames, dtype=np.float32)

        output = self.output_buffer.get(frames)
        if self.output_buffer.available == 0:
            self.stats.buffer_underruns += 1
        return output

    # ======== Internal Helpers ========
    def _create_chunking_strategy(self):
        chunk_config = ChunkConfig(
            mic_sample_rate=self.config.mic_sample_rate,
            chunk_sec=self.config.chunk_sec,
            context_sec=self.config.context_sec,
            lookahead_sec=self.config.lookahead_sec,
            rvc_overlap_sec=self.config.rvc_overlap_sec,
        )
        return create_chunking_strategy(
            self.config.chunking_mode,
            chunk_config,
            align_to_hubert=(self.config.chunking_mode != "rvc_webui"),
        )

    def _create_crossfade_strategy(self):
        crossfade_config = CrossfadeConfig(
            output_sample_rate=self.config.output_sample_rate,
            crossfade_sec=self.config.crossfade_sec,
            context_sec=self.config.context_sec,
            rvc_overlap_sec=self.config.rvc_overlap_sec,
            chunk_sec=self.config.chunk_sec,
        )
        return create_crossfade_strategy(
            self.config.chunking_mode,
            self.config.use_sola,
            crossfade_config,
        )

    def _estimate_f0_hz(self, audio_16k: np.ndarray) -> float:
        """Lightweight autocorrelation-based F0 estimate (for SOLA search range)."""
        if audio_16k.size < 1600:
            return 0.0
        # Use last 200ms for stability
        win = min(len(audio_16k), int(self.config.input_sample_rate * 0.2))
        segment = audio_16k[-win:]
        rms = float(np.sqrt(np.mean(segment**2)))
        if rms < 0.01:
            return 0.0

        # Autocorrelation (normalized by energy)
        segment = segment - np.mean(segment)
        corr = np.correlate(segment, segment, mode="full")
        corr = corr[corr.size // 2 :]
        if corr[0] <= 0:
            return 0.0

        # Search lag range for human voice
        min_f0 = 50.0
        max_f0 = 400.0
        min_lag = int(self.config.input_sample_rate / max_f0)
        max_lag = int(self.config.input_sample_rate / min_f0)
        if max_lag <= min_lag or max_lag >= corr.size:
            return 0.0

        corr[:min_lag] = 0
        peak_lag = int(np.argmax(corr[min_lag:max_lag]) + min_lag)
        if peak_lag <= 0:
            return 0.0

        f0_hz = float(self.config.input_sample_rate / peak_lag)
        if f0_hz < min_f0 or f0_hz > max_f0:
            return 0.0
        return f0_hz

    def _update_sola_fallback(self, f0_hz: float) -> None:
        """Adapt SOLA fallback threshold based on F0 stability."""
        if self._sola_state is None:
            return

        base = float(self.config.sola_fallback_threshold)
        if f0_hz <= 0.0:
            # Unvoiced/unstable: slightly more aggressive fallback
            self._sola_state.fallback_threshold = float(min(0.9, base + 0.05))
            self._sola_state.fallback_strength = 1.0
            return

        alpha = 0.9
        if self._f0_ema <= 0.0:
            self._f0_ema = f0_hz
            self._f0_dev_ema = 0.0
        else:
            dev = abs(f0_hz - self._f0_ema)
            self._f0_ema = alpha * self._f0_ema + (1 - alpha) * f0_hz
            self._f0_dev_ema = alpha * self._f0_dev_ema + (1 - alpha) * dev

        tolerance = max(30.0, self._f0_ema * 0.2)
        stability = 1.0 - min(1.0, self._f0_dev_ema / tolerance)

        # Stable -> lower threshold (less fallback). Unstable -> higher threshold.
        threshold = base + (0.5 - stability) * 0.2
        self._sola_state.fallback_threshold = float(np.clip(threshold, 0.5, 0.9))
        self._sola_state.fallback_strength = float(np.clip(1.0 - stability, 0.0, 1.0))

    def _effective_history_sec(self) -> float:
        """Compatibility hook for future history tuning (currently passthrough)."""
        return self.config.history_sec

    def _recalculate_buffers(self) -> None:
        self._chunking_strategy = self._create_chunking_strategy()
        self._crossfade_strategy = self._create_crossfade_strategy()
        self._sola_state = getattr(self._crossfade_strategy, "_sola_state", None)
        self.output_crossfade_samples = int(
            self.config.output_sample_rate * self.config.crossfade_sec
        )

        self.mic_chunk_samples = self._chunking_strategy.chunk_samples
        self.mic_hop_samples = self._chunking_strategy.hop_samples
        self.mic_context_samples = self._chunking_strategy.context_samples
        self.mic_lookahead_samples = self._chunking_strategy.lookahead_samples

        max_latency_sec = self.config.chunk_sec * (
            self.config.prebuffer_chunks + self.config.buffer_margin
        )
        self.output_buffer = OutputBuffer(
            max_latency_samples=int(self.config.output_sample_rate * max_latency_sec),
            fade_samples=256,
        )
        self._ensure_sola_state()

    def _clear_queues(self) -> None:
        while not self._input_queue.empty():
            try:
                self._input_queue.get_nowait()
            except Empty:
                break
        while not self._output_queue.empty():
            try:
                self._output_queue.get_nowait()
            except Empty:
                break

    def _ensure_sola_state(self) -> None:
        """Create SOLA state with V1-equivalent minimum buffer sizing."""
        if not self.config.use_sola:
            return
        # RVC WebUI uses its own crossfade strategy state
        if (
            self.config.chunking_mode == "rvc_webui"
            and hasattr(self._crossfade_strategy, "_sola_state")
        ):
            self._sola_state = getattr(self._crossfade_strategy, "_sola_state", None)
            return
        min_phase8_buffer = int(self.config.output_sample_rate * 0.08)  # 80ms
        sola_buffer_size = max(self.output_crossfade_samples, min_phase8_buffer)
        self._sola_state = SOLAState.create(
            sola_buffer_size,
            self.config.output_sample_rate,
            use_advanced_sola=self.config.sola_use_advanced,
            fallback_threshold=self.config.sola_fallback_threshold,
        )
