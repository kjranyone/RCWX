"""Unified real-time voice conversion pipeline.

Single processing path with:
- Audio-level overlap (no feature-level caching/blending)
- HuBERT frame-aligned chunks for deterministic output length
- Simple SOLA crossfade (~150 lines, no fallback)
- Pre-allocated ring output buffer
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Callable, Optional

import numpy as np

from rcwx.audio.buffer import RingOutputBuffer
from rcwx.audio.denoise import denoise as denoise_audio
from rcwx.audio.input import AudioInput
from rcwx.audio.output import AudioOutput
from rcwx.audio.resample import StatefulResampler, resample
from rcwx.audio.sola import SolaState, sola_crossfade, sola_flush
from rcwx.pipeline.inference import RVCPipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RealtimeConfig / RealtimeStats
# ---------------------------------------------------------------------------

@dataclass
class RealtimeStats:
    """Statistics for real-time processing."""

    latency_ms: float = 0.0
    inference_ms: float = 0.0
    buffer_underruns: int = 0
    buffer_overruns: int = 0
    frames_processed: int = 0
    feedback_detected: bool = False
    feedback_correlation: float = 0.0

    def reset(self) -> None:
        self.latency_ms = 0.0
        self.inference_ms = 0.0
        self.buffer_underruns = 0
        self.buffer_overruns = 0
        self.frames_processed = 0
        self.feedback_detected = False
        self.feedback_correlation = 0.0


@dataclass
class RealtimeConfig:
    """Configuration for unified real-time processing."""

    # Device selection
    input_device: Optional[int] = None
    output_device: Optional[int] = None

    # Sample rates
    mic_sample_rate: int = 48000
    input_sample_rate: int = 16000  # internal processing rate
    output_sample_rate: int = 48000

    # Channels
    input_channels: int = 1
    output_channels: int = 1
    input_channel_selection: str = "auto"

    # Core parameters
    chunk_sec: float = 0.15
    overlap_sec: float = 0.10  # audio-level overlap for HuBERT continuity
    crossfade_sec: float = 0.05
    sola_search_ms: float = 10.0  # SOLA search window in ms
    prebuffer_chunks: int = 1
    buffer_margin: float = 0.3

    # Pitch / F0
    pitch_shift: int = 0
    use_f0: bool = True
    f0_method: str = "fcpe"

    # Processing
    max_queue_size: int = 8
    input_gain_db: float = 0.0
    index_rate: float = 0.0
    index_k: int = 4
    use_parallel_extraction: bool = True

    # Denoise
    denoise_enabled: bool = False
    denoise_method: str = "auto"

    # Voice gate
    voice_gate_mode: str = "expand"
    energy_threshold: float = 0.05

    # SOLA
    use_sola: bool = True

    # Synthesizer
    synth_min_frames: int = 64

    def __post_init__(self) -> None:
        # Round chunk_sec to HuBERT frame boundary (20ms)
        frame_ms = 20
        chunk_ms = self.chunk_sec * 1000
        rounded_ms = round(chunk_ms / frame_ms) * frame_ms
        if rounded_ms != chunk_ms:
            logger.info(
                f"[RealtimeConfig] Rounding chunk_sec from {self.chunk_sec:.3f}s "
                f"to {rounded_ms / 1000:.3f}s (HuBERT frame alignment)"
            )
            object.__setattr__(self, "chunk_sec", rounded_ms / 1000)

        # Round overlap_sec to HuBERT frame boundary
        overlap_ms = self.overlap_sec * 1000
        rounded_overlap = round(overlap_ms / frame_ms) * frame_ms
        if rounded_overlap != overlap_ms:
            object.__setattr__(self, "overlap_sec", rounded_overlap / 1000)

        # Validate F0 method minimum chunk
        if self.use_f0:
            if self.f0_method == "fcpe" and self.chunk_sec < 0.10:
                object.__setattr__(self, "chunk_sec", 0.10)
            elif self.f0_method == "rmvpe" and self.chunk_sec < 0.32:
                object.__setattr__(self, "chunk_sec", 0.32)


# ---------------------------------------------------------------------------
# Unified Voice Changer
# ---------------------------------------------------------------------------

class RealtimeVoiceChangerUnified:
    """Unified real-time voice changer.

    Single processing path:
    1. Accumulate mic audio into hop-sized chunks
    2. Assemble [overlap | new_hop] aligned to HuBERT frames
    3. Resample 48k->16k, optional denoise
    4. pipeline.infer_streaming() — HuBERT sees full context, overlap trimmed
    5. Resample model_sr->48k
    6. Simple SOLA crossfade
    7. Write to RingOutputBuffer
    """

    def __init__(
        self,
        pipeline: RVCPipeline,
        config: Optional[RealtimeConfig] = None,
        on_warmup_progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> None:
        self.pipeline = pipeline
        self.config = config or RealtimeConfig()
        self._on_warmup_progress = on_warmup_progress

        self.stats = RealtimeStats()
        self.on_stats_update: Optional[Callable[[RealtimeStats], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None

        # Ensure pipeline is loaded
        if not self.pipeline._loaded:
            if self._on_warmup_progress:
                self._on_warmup_progress(0, 1, "モデル読み込み中...")
            self.pipeline.load()
            if self._on_warmup_progress:
                self._on_warmup_progress(1, 1, "準備完了")

        # Warmup
        if self._on_warmup_progress:
            self._on_warmup_progress(0, 2, "ウォームアップ中...")
        self._run_warmup()
        if self._on_warmup_progress:
            self._on_warmup_progress(2, 2, "準備完了")

        # Calculate sample counts (all at 16kHz for model input)
        hubert_hop = 320
        self._hop_samples_16k = self._align_to_hop(
            int(self.config.chunk_sec * 16000), hubert_hop
        )
        self._overlap_samples_16k = self._align_to_hop(
            int(self.config.overlap_sec * 16000), hubert_hop
        )

        # Corresponding counts at mic rate
        self._hop_samples_mic = int(
            self._hop_samples_16k * self.config.mic_sample_rate / 16000
        )

        # Resamplers
        self.input_resampler = StatefulResampler(
            self.config.mic_sample_rate, 16000
        )
        self.output_resampler = StatefulResampler(
            self.pipeline.sample_rate, self.config.output_sample_rate
        )

        # SOLA state
        crossfade_samples_out = int(
            self.config.output_sample_rate * self.config.crossfade_sec
        )
        search_samples_out = int(
            self.config.output_sample_rate * self.config.sola_search_ms / 1000
        )
        self._sola_state = SolaState(
            crossfade_samples=crossfade_samples_out,
            search_samples=search_samples_out,
        )

        # SOLA extra: produce additional output samples (at model_sr) so
        # SOLA has a full crossfade+search region that overlaps with the
        # previous chunk's tail.  This matches w-okada's approach where the
        # synthesizer output includes room for SOLA to crossfade without
        # clipping into non-overlapping audio.  The ring buffer absorbs the
        # resulting small surplus (~crossfade_sec per chunk) via overflow.
        sola_extra_out = crossfade_samples_out + search_samples_out
        self._sola_extra_model = int(
            sola_extra_out * self.pipeline.sample_rate
            / self.config.output_sample_rate
        )

        # Output buffer: 3x chunk capacity
        chunk_output_samples = int(
            self.config.output_sample_rate * self.config.chunk_sec
        )
        self.output_buffer = RingOutputBuffer(
            capacity_samples=max(chunk_output_samples * 3, 48000),  # at least 1s
            fade_samples=256,
        )

        # Input accumulator (ring buffer at mic rate)
        self._input_buf = np.array([], dtype=np.float32)

        # Overlap buffer at 16kHz (stores tail of previous chunk for prepending)
        self._overlap_buf: Optional[np.ndarray] = None

        # Threading
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._input_queue: Queue = Queue(maxsize=self.config.max_queue_size)
        self._output_queue: Queue = Queue(maxsize=self.config.max_queue_size)

        # Pre-buffering
        self._prebuffer_chunks = self.config.prebuffer_chunks
        self._chunks_ready = 0
        self._output_started = False

        # Audio streams
        self._input_stream: Optional[AudioInput] = None
        self._output_stream: Optional[AudioOutput] = None

        # Feedback detection
        self._output_history_size = self.config.output_sample_rate  # 1 second
        self._output_history = np.zeros(self._output_history_size, dtype=np.float32)
        self._output_history_pos = 0
        self._feedback_check_interval = 10
        self._feedback_warning_shown = False

        # Overload protection
        self._queue_full_times: deque[float] = deque(maxlen=50)
        self._overload_until: float = 0.0
        self._overload_active = False

    @staticmethod
    def _align_to_hop(samples: int, hop: int) -> int:
        """Round sample count up to nearest multiple of hop."""
        return ((samples + hop - 1) // hop) * hop

    def _recalculate_sizes(self) -> None:
        """Recalculate derived sample counts from current config."""
        hubert_hop = 320
        self._hop_samples_16k = self._align_to_hop(
            int(self.config.chunk_sec * 16000), hubert_hop
        )
        self._overlap_samples_16k = self._align_to_hop(
            int(self.config.overlap_sec * 16000), hubert_hop
        )
        self._hop_samples_mic = int(
            self._hop_samples_16k * self.config.mic_sample_rate / 16000
        )

        # SOLA extra samples
        crossfade_samples_out = int(
            self.config.output_sample_rate * self.config.crossfade_sec
        )
        search_samples_out = int(
            self.config.output_sample_rate * self.config.sola_search_ms / 1000
        )
        self._sola_state.crossfade_samples = crossfade_samples_out
        self._sola_state.search_samples = search_samples_out
        self._sola_state._hann_fade_in = None
        self._sola_state._hann_fade_out = None
        sola_extra_out = crossfade_samples_out + search_samples_out
        self._sola_extra_model = int(
            sola_extra_out * self.pipeline.sample_rate
            / self.config.output_sample_rate
        )

    # ======== Lifecycle ========

    def start(self) -> None:
        if self._running:
            return

        self._recalculate_sizes()
        self.pipeline.clear_cache()
        self.stats.reset()
        self.output_buffer.clear()
        self._input_buf = np.array([], dtype=np.float32)
        self._overlap_buf = None
        self._chunks_ready = 0
        self._output_started = False
        self.input_resampler.reset()
        self.output_resampler.reset()

        # Reset SOLA
        self._sola_state.buffer = None

        # Reset feedback
        self._output_history.fill(0)
        self._output_history_pos = 0
        self._feedback_warning_shown = False

        self._clear_queues()

        self._running = True
        self._thread = threading.Thread(
            target=self._inference_thread,
            daemon=True,
            name="RCWX-Inference-Unified",
        )
        self._thread.start()

        # Audio stream block size: chunk/4 for responsive I/O
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

        logger.info(
            f"Unified pipeline started: chunk={self.config.chunk_sec}s, "
            f"overlap={self.config.overlap_sec}s, "
            f"hop_16k={self._hop_samples_16k}, overlap_16k={self._overlap_samples_16k}"
        )

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

    def set_pitch_shift(self, semitones: int) -> None:
        self.config.pitch_shift = semitones

    def set_f0_mode(self, enabled: bool) -> None:
        self.config.use_f0 = enabled

    def set_f0_method(self, method: str) -> None:
        old = self.config.f0_method
        self.config.f0_method = method
        # Enforce minimum chunk_sec for the new F0 method
        if self.config.use_f0:
            min_chunk = {"rmvpe": 0.32, "fcpe": 0.10}.get(method, 0.0)
            if min_chunk > 0 and self.config.chunk_sec < min_chunk:
                logger.info(
                    f"F0 method {method} requires chunk_sec >= {min_chunk}s, "
                    f"adjusting from {self.config.chunk_sec}s"
                )
                self.set_chunk_sec(min_chunk)
                return
        if old != method:
            logger.info(f"F0 method changed: {old} -> {method}")

    def set_index_rate(self, rate: float) -> None:
        self.config.index_rate = rate

    def set_denoise(self, enabled: bool, method: str = "auto") -> None:
        self.config.denoise_enabled = enabled
        self.config.denoise_method = method

    def set_voice_gate_mode(self, mode: str) -> None:
        self.config.voice_gate_mode = mode

    def set_energy_threshold(self, value: float) -> None:
        self.config.energy_threshold = value

    def set_chunk_sec(self, chunk_sec: float) -> None:
        old = self.config.chunk_sec
        self.config.chunk_sec = max(0.1, min(0.6, chunk_sec))
        if self._running:
            logger.info(f"Chunk size changed ({old}s -> {self.config.chunk_sec}s), restarting...")
            self.stop()
            self.start()

    def set_prebuffer_chunks(self, chunks: int) -> None:
        self.config.prebuffer_chunks = max(0, min(3, int(chunks)))
        self._prebuffer_chunks = self.config.prebuffer_chunks

    def set_buffer_margin(self, margin: float) -> None:
        self.config.buffer_margin = max(0.1, min(2.0, float(margin)))

    def set_overlap(self, overlap_sec: float) -> None:
        self.config.overlap_sec = max(0.0, float(overlap_sec))
        # Recompute derived sample count
        self._overlap_samples_16k = self._align_to_hop(
            int(self.config.overlap_sec * 16000), 320
        )

    def set_crossfade(self, crossfade_sec: float) -> None:
        self.config.crossfade_sec = max(0.0, crossfade_sec)
        # Rebuild SOLA state with new crossfade length
        crossfade_samples_out = int(
            self.config.output_sample_rate * self.config.crossfade_sec
        )
        search_samples_out = int(
            self.config.output_sample_rate * self.config.sola_search_ms / 1000
        )
        self._sola_state = SolaState(
            crossfade_samples=crossfade_samples_out,
            search_samples=search_samples_out,
        )

    def set_sola(self, enabled: bool) -> None:
        self.config.use_sola = enabled

    # ======== Audio Callbacks ========

    def _on_audio_input(self, audio: np.ndarray) -> None:
        """Accumulate mic audio and queue hop-sized chunks."""
        self._input_buf = np.concatenate([self._input_buf, audio])

        while len(self._input_buf) >= self._hop_samples_mic:
            hop = self._input_buf[:self._hop_samples_mic].copy()
            self._input_buf = self._input_buf[self._hop_samples_mic:]
            try:
                self._input_queue.put_nowait(hop)
            except Exception:
                logger.warning("[INPUT] Queue full, dropping chunk")
                self._record_queue_full()
                break

    def _on_audio_output(self, frames: int) -> np.ndarray:
        """Drain output queue into ring buffer and serve playback."""
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

    # ======== Inference Thread ========

    def _inference_thread(self) -> None:
        logger.info("Unified inference thread started")

        while self._running:
            try:
                # Get hop audio at mic rate
                hop_mic = self._input_queue.get(timeout=0.5)
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Inference error: {e}", exc_info=True)
                if self.on_error:
                    self.on_error(f"推論エラー: {e}")
                continue

            try:
                start_time = time.perf_counter()

                # --- Stage 1: Input gain ---
                if self.config.input_gain_db != 0.0:
                    gain = 10 ** (self.config.input_gain_db / 20)
                    hop_mic = hop_mic * gain

                chunk_at_mic_rate = hop_mic.copy()

                # --- Stage 2: Resample 48k -> 16k ---
                hop_16k = self.input_resampler.resample_chunk(hop_mic)

                # --- Stage 3: Optional denoise ---
                if not self._is_overloaded() and self.config.denoise_enabled:
                    hop_16k = denoise_audio(
                        hop_16k,
                        sample_rate=16000,
                        method=self.config.denoise_method,
                        device="cpu",
                    )

                # --- Stage 4: Assemble chunk with overlap ---
                # Align hop to HuBERT frame boundary (320 samples)
                hubert_hop = 320
                aligned_hop = self._align_to_hop(len(hop_16k), hubert_hop)
                if len(hop_16k) < aligned_hop:
                    hop_16k = np.pad(hop_16k, (0, aligned_hop - len(hop_16k)))
                elif len(hop_16k) > aligned_hop:
                    hop_16k = hop_16k[:aligned_hop]

                if self._overlap_buf is not None:
                    # Ensure overlap is aligned
                    overlap = self._overlap_buf
                    if len(overlap) % hubert_hop != 0:
                        aligned_ovl = self._align_to_hop(len(overlap), hubert_hop)
                        if len(overlap) < aligned_ovl:
                            overlap = np.pad(overlap, (aligned_ovl - len(overlap), 0))
                        else:
                            overlap = overlap[-aligned_ovl:]

                    chunk_16k = np.concatenate([overlap, hop_16k])
                    overlap_samples = len(overlap)
                else:
                    # First chunk: use reflection padding as overlap
                    # Cap at hop length (can't reflect more than we have)
                    overlap_len = min(self._overlap_samples_16k, len(hop_16k))
                    if overlap_len > 0:
                        reflection = hop_16k[:overlap_len][::-1].copy()
                        chunk_16k = np.concatenate([reflection, hop_16k])
                        overlap_samples = overlap_len
                    else:
                        chunk_16k = hop_16k
                        overlap_samples = 0

                # Store tail for next chunk's overlap (accumulating)
                if self._overlap_samples_16k > 0:
                    if self._overlap_buf is not None:
                        combined = np.concatenate([self._overlap_buf, hop_16k])
                        self._overlap_buf = combined[-self._overlap_samples_16k:]
                    else:
                        tail = min(self._overlap_samples_16k, len(hop_16k))
                        self._overlap_buf = hop_16k[-tail:].copy()
                else:
                    self._overlap_buf = None

                # --- Stage 5: Inference ---
                if self._is_overloaded():
                    f0_method = "none"
                    index_rate = 0.0
                else:
                    f0_method = self.config.f0_method if self.config.use_f0 else "none"
                    index_rate = self.config.index_rate

                output_model = self.pipeline.infer_streaming(
                    audio_16k=chunk_16k,
                    overlap_samples=overlap_samples,
                    pitch_shift=self.config.pitch_shift,
                    f0_method=f0_method,
                    index_rate=index_rate,
                    index_k=self.config.index_k,
                    voice_gate_mode=self.config.voice_gate_mode,
                    energy_threshold=self.config.energy_threshold,
                    use_parallel_extraction=self.config.use_parallel_extraction,
                    sola_extra_samples=self._sola_extra_model,
                )

                # --- Stage 6: Resample model_sr -> 48kHz ---
                output_48k = self.output_resampler.resample_chunk(output_model)

                # Soft clip
                max_val = np.max(np.abs(output_48k)) if len(output_48k) else 0.0
                if max_val > 1.0:
                    output_48k = np.tanh(output_48k)

                # --- Stage 7: SOLA crossfade ---
                if self.config.use_sola:
                    output_48k = sola_crossfade(output_48k, self._sola_state)

                # --- Stage 7b: Output length ---
                # Hold-back SOLA + sola_extra_samples (= cf + search):
                #   output ≈ hop_mic + search - offset ≈ hop_mic (± search)
                # Small surplus (0–10ms/chunk) absorbed by RingOutputBuffer.

                # --- Stage 8: Feedback detection ---
                self._store_output_history(output_48k)
                self._maybe_check_feedback(chunk_at_mic_rate)

                # --- Stage 9: Queue output ---
                inference_ms = (time.perf_counter() - start_time) * 1000
                self.stats.inference_ms = inference_ms
                self.stats.frames_processed += 1
                self.stats.latency_ms = (
                    self.config.chunk_sec * 1000
                    + inference_ms
                    + (self.output_buffer.available / self.config.output_sample_rate) * 1000
                )

                try:
                    self._output_queue.put_nowait(output_48k)
                except Exception:
                    logger.warning("Output queue full, dropping chunk")
                    self.stats.buffer_overruns += 1

                if self.on_stats_update:
                    self.on_stats_update(self.stats)

                # Performance monitoring
                chunk_ms = self.config.chunk_sec * 1000
                if inference_ms > chunk_ms * 0.8 and self.stats.frames_processed % 50 == 0:
                    logger.warning(
                        f"[PERF] Inference slow: {inference_ms:.0f}ms > {chunk_ms:.0f}ms chunk"
                    )

                if self.stats.frames_processed <= 5:
                    logger.info(
                        f"[INFER] Chunk #{self.stats.frames_processed}: "
                        f"hop_16k={len(hop_16k)}, overlap={overlap_samples}, "
                        f"out_model={len(output_model)}, out_48k={len(output_48k)}, "
                        f"infer={inference_ms:.0f}ms"
                    )

            except Exception as e:
                logger.error(f"Inference error: {e}", exc_info=True)
                if self.on_error:
                    self.on_error(f"推論エラー: {e}")

        # Flush SOLA buffer on stop
        if self.config.use_sola:
            final = sola_flush(self._sola_state)
            if len(final) > 0:
                try:
                    self._output_queue.put(final, timeout=0.5)
                except Exception:
                    pass

        logger.info("Unified inference thread stopped")

    # ======== Feedback Detection ========

    def _store_output_history(self, output: np.ndarray) -> None:
        if self.config.output_sample_rate != self.config.mic_sample_rate:
            output = resample(output, self.config.output_sample_rate, self.config.mic_sample_rate)
        out_len = len(output)
        if out_len >= self._output_history_size:
            self._output_history[:] = output[-self._output_history_size:]
            self._output_history_pos = 0
            return
        end_pos = self._output_history_pos + out_len
        if end_pos <= self._output_history_size:
            self._output_history[self._output_history_pos:end_pos] = output
        else:
            first = self._output_history_size - self._output_history_pos
            self._output_history[self._output_history_pos:] = output[:first]
            self._output_history[:out_len - first] = output[first:]
        self._output_history_pos = end_pos % self._output_history_size

    def _check_feedback(self, input_audio: np.ndarray) -> float:
        if len(input_audio) < 1000:
            return 0.0
        input_rms = np.sqrt(np.mean(input_audio ** 2))
        if input_rms < 0.01:
            return 0.0
        output_rms = np.sqrt(np.mean(self._output_history ** 2))
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
                logger.warning(f"[FEEDBACK] Detected feedback (corr={corr:.2f})")
                if self.on_error:
                    self.on_error("フィードバック検出: 入力と出力が接続されている可能性があります。")

    # ======== Overload Protection ========

    def _record_queue_full(self) -> None:
        now = time.time()
        self._queue_full_times.append(now)
        recent = [t for t in self._queue_full_times if now - t <= 1.0]
        if len(recent) >= 3 and self._overload_until < now:
            self._overload_until = now + 2.0
            if not self._overload_active:
                self._overload_active = True
                logger.warning("Overload: temporarily disabling F0/index for stability")

    def _is_overloaded(self) -> bool:
        if self._overload_until <= 0:
            return False
        now = time.time()
        if now <= self._overload_until:
            return True
        if self._overload_active:
            self._overload_active = False
            logger.info("Overload cleared: restoring full quality")
        return False

    # ======== Warmup ========

    def _run_warmup(self) -> None:
        warmup_samples = int(self.config.mic_sample_rate * 0.5)
        dummy = np.random.randn(warmup_samples).astype(np.float32) * 0.001
        logger.info("[WARMUP] Starting warmup inference...")
        try:
            for i in range(2):
                _ = self.pipeline.infer(
                    audio=dummy,
                    input_sr=self.config.mic_sample_rate,
                    pitch_shift=0,
                    f0_method=self.config.f0_method,
                    index_rate=0.0,
                    use_feature_cache=False,
                    allow_short_input=True,
                    synth_min_frames=self.config.synth_min_frames,
                )
                logger.info(f"[WARMUP] Warmup {i + 1}/2 complete")
            self.pipeline.clear_cache()
        except Exception as e:
            logger.warning(f"[WARMUP] Failed (non-fatal): {e}")

    # ======== Helpers ========

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

    # ======== Public Testing Methods ========

    def process_input_chunk(self, audio: np.ndarray) -> None:
        """Feed audio input directly (for testing)."""
        self._on_audio_input(audio)

    def process_next_chunk(self) -> bool:
        """Process one chunk from input queue (for testing). Returns True if processed."""
        try:
            hop_mic = self._input_queue.get_nowait()
        except Empty:
            return False
        # Just put it back and let inference thread handle it
        # (This is a simplified test interface)
        self._input_queue.put(hop_mic)
        return True

    def get_output_chunk(self, frames: int) -> np.ndarray:
        """Get output audio directly (for testing)."""
        return self._on_audio_output(frames)
