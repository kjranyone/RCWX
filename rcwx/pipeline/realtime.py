"""Real-time voice conversion pipeline."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Callable, Optional

import numpy as np

from rcwx.audio.buffer import ChunkBuffer, OutputBuffer
from rcwx.audio.crossfade import SOLAState, apply_sola_crossfade, flush_sola_buffer
from rcwx.audio.denoise import denoise as denoise_audio
from rcwx.audio.input import AudioInput
from rcwx.audio.output import AudioOutput
from rcwx.audio.resample import resample
from rcwx.pipeline.inference import RVCPipeline

logger = logging.getLogger(__name__)


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
    """Configuration for real-time processing."""

    input_device: Optional[int] = None
    output_device: Optional[int] = None
    # Microphone capture rate (native device rate for better compatibility)
    mic_sample_rate: int = 48000
    # Input/output channels (1=mono, 2=stereo) - auto-detected from device
    input_channels: int = 1
    output_channels: int = 1
    # Channel selection for stereo input: "left", "right", "average"
    input_channel_selection: str = "average"
    # Internal processing rate (HuBERT/F0 models expect 16kHz)
    input_sample_rate: int = 16000
    output_sample_rate: int = 48000
    # Optimized: 100ms for ultra-low latency with FCPE (official minimum)
    # FCPE: 100-150ms (stable RTF ~0.5x, latency ~250-350ms)
    # RMVPE: 250-350ms (stable RTF 0.54x, latency ~530ms)
    chunk_sec: float = 0.10
    pitch_shift: int = 0
    use_f0: bool = True
    # F0 extraction method:
    # - "fcpe" (fast, low-latency, ~385ms total latency)
    # - "rmvpe" (high-quality, higher-latency, ~530ms total latency)
    f0_method: str = "fcpe"
    max_queue_size: int = 8
    # Number of chunks to pre-buffer before starting output (1 = minimal latency)
    prebuffer_chunks: int = 1
    # Buffer margin multiplier for max_latency calculation (0.3 = tight, 0.5 = balanced, 1.0 = relaxed)
    buffer_margin: float = 0.3
    # Input gain in dB
    input_gain_db: float = 0.0
    # FAISS index rate (0=disabled, 0.5=balanced, 1=index only)
    index_rate: float = 0.0
    # FAISS neighbors to search (4=fast, 8=quality)
    index_k: int = 4
    # Resampling method ("poly"=quality, "linear"=fast)
    resample_method: str = "linear"
    # Parallel HuBERT+F0 extraction (GPU streams, ~10-15% speedup)
    use_parallel_extraction: bool = True
    # Noise cancellation
    denoise_enabled: bool = False
    denoise_method: str = "auto"  # auto, deepfilter, spectral
    # Voice gate mode: off, strict, expand, energy
    voice_gate_mode: str = "expand"
    # Energy threshold for "energy" mode (0.01-0.2)
    energy_threshold: float = 0.05
    # Feature caching for chunk continuity
    use_feature_cache: bool = True

    # --- w-okada style processing ---
    # Each chunk: [left_context | main | right_context]
    # Output: keep only main portion, discard context edges
    # Crossfade: blend main outputs at boundaries

    # Context: extra audio on left side for stable edge processing
    # This is discarded from output but provides context for RVC
    # 0.05 = 50ms context (minimal for low latency)
    context_sec: float = 0.05

    # Extra discard: additional samples to discard beyond context
    extra_sec: float = 0.0

    # Crossfade region for SOLA blending
    # 50ms is sufficient for smooth transitions
    crossfade_sec: float = 0.05

    # Lookahead: future samples for right context (ADDS LATENCY!)
    # 0 = no lookahead (lowest latency)
    # Only increase if edge quality is poor
    lookahead_sec: float = 0.0

    # Enable SOLA (Synchronized Overlap-Add) for optimal crossfade position
    # w-okada compatible mode: uses left context for crossfading
    use_sola: bool = True

    # SOLA search range (as ratio of crossfade length)
    sola_search_ratio: float = 0.25


class RealtimeVoiceChanger:
    """
    Real-time voice conversion using RVC pipeline.

    Manages audio input/output streams and runs inference in a separate thread.
    """

    def __init__(
        self,
        pipeline: RVCPipeline,
        config: Optional[RealtimeConfig] = None,
    ):
        """
        Initialize the real-time voice changer.

        Args:
            pipeline: RVC pipeline for inference
            config: Real-time configuration
        """
        self.pipeline = pipeline
        self.config = config or RealtimeConfig()

        # Calculate buffer sizes (at mic sample rate for input buffering)
        # Main chunk size (the "useful" audio we want to output per iteration)
        self.mic_chunk_samples = int(self.config.mic_sample_rate * self.config.chunk_sec)

        # Context: extra audio on each side for stable processing (will be discarded)
        self.mic_context_samples = int(self.config.mic_sample_rate * self.config.context_sec)

        # Lookahead: additional future samples for right context (adds latency)
        self.mic_lookahead_samples = int(self.config.mic_sample_rate * self.config.lookahead_sec)

        # Total input chunk size = left_context + main + right_context
        # But we only need to buffer main chunks, context comes from adjacent chunks
        # ChunkBuffer keeps context_samples from previous chunk as left context
        self.mic_total_chunk = self.mic_chunk_samples + 2 * self.mic_context_samples

        # Audio streams
        self.audio_input: Optional[AudioInput] = None
        self.audio_output: Optional[AudioOutput] = None

        # Ensure pipeline is loaded and warmed up on initialization
        # This prevents thread conflicts during start()
        if not self.pipeline._loaded:
            self.pipeline.load()

        # Warmup inference (synchronous, in __init__)
        # XPU/CUDA kernel compilation happens here
        logger.info("Warming up inference pipeline...")
        warmup_samples = int(
            (self.mic_chunk_samples + self.mic_context_samples + self.mic_lookahead_samples)
            * self.config.input_sample_rate
            / self.config.mic_sample_rate
        )
        try:
            warmup_audio = np.zeros(warmup_samples, dtype=np.float32)
            self.pipeline.infer(
                warmup_audio,
                input_sr=self.config.input_sample_rate,
                pitch_shift=0,
                f0_method=self.config.f0_method if self.config.use_f0 else "none",
                index_rate=0.0,
                voice_gate_mode="off",
                use_feature_cache=False,
            )
            t = np.arange(warmup_samples) / self.config.input_sample_rate
            warmup_audio = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
            for _ in range(3):
                self.pipeline.infer(
                    warmup_audio,
                    input_sr=self.config.input_sample_rate,
                    pitch_shift=self.config.pitch_shift,
                    f0_method=self.config.f0_method if self.config.use_f0 else "none",
                    index_rate=self.config.index_rate,
                    voice_gate_mode=self.config.voice_gate_mode,
                    use_feature_cache=True,
                )
            logger.info("Warmup complete")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

        # Buffers - w-okada style processing with lookahead
        # - Chunk structure: [left_context | main | right_context]
        # - context_samples = kept for next chunk's left context
        # - lookahead_samples = future samples for right context (improves quality at chunk end)
        # IMPORTANT: chunk_samples must equal mic_chunk_samples (not + context)
        # so that input consumption matches output duration
        self.input_buffer = ChunkBuffer(
            self.mic_chunk_samples,  # main only - advance by this amount each chunk
            crossfade_samples=0,
            context_samples=self.mic_context_samples,  # Keep as left context for next
            lookahead_samples=self.mic_lookahead_samples,  # Right context for quality
        )

        # Output processing parameters (at output sample rate)
        # Crossfade: blending region between main outputs
        self.output_crossfade_samples = int(
            self.config.output_sample_rate * self.config.crossfade_sec
        )
        # Extra discard: additional samples to remove from edges
        self.output_extra_samples = int(self.config.output_sample_rate * self.config.extra_sec)
        # Context at output rate: this is discarded from output
        self.output_context_samples = int(self.config.output_sample_rate * self.config.context_sec)
        # Max buffer = prebuffer + margin (configurable for latency/stability tradeoff)
        # Tighter buffer = lower latency, but may cause underruns if inference is slow
        max_latency_sec = self.config.chunk_sec * (
            self.config.prebuffer_chunks + self.config.buffer_margin
        )
        self.output_buffer = OutputBuffer(
            max_latency_samples=int(self.config.output_sample_rate * max_latency_sec),
            fade_samples=256,
        )
        # SOLA state for RVC-style crossfade (on raw output, before trim)
        self._sola_state = SOLAState.create(
            self.output_crossfade_samples,
            self.config.output_sample_rate,
        )

        # Processing state
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._input_queue: Queue = Queue(maxsize=self.config.max_queue_size)
        self._output_queue: Queue = Queue(maxsize=self.config.max_queue_size)

        # Pre-buffering state (wait for N chunks before starting output)
        self._prebuffer_chunks = self.config.prebuffer_chunks
        self._chunks_ready = 0
        self._output_started = False

        # Statistics
        self.stats = RealtimeStats()

        # Callbacks
        self.on_stats_update: Optional[Callable[[RealtimeStats], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None

        # Error tracking
        self._last_error: Optional[str] = None
        self._error_count: int = 0

        # Feedback detection: store recent output for correlation check
        # Store ~1 second of output for comparison with input
        self._output_history_size = self.config.output_sample_rate  # 1 second
        self._output_history: np.ndarray = np.zeros(self._output_history_size, dtype=np.float32)
        self._output_history_pos = 0
        self._feedback_check_interval = 10  # Check every N chunks
        self._feedback_warning_shown = False

        # Crossfade windows (at output sample rate)
        # Use equal-power crossfade (sine curve) to maintain constant power
        # during crossfade, even when signals have phase differences
        if self.output_crossfade_samples > 0:
            # Equal-power: fade_in^2 + fade_out^2 = 1
            t = np.linspace(0, np.pi / 2, self.output_crossfade_samples, dtype=np.float32)
            self._fade_in = np.sin(t)
            self._fade_out = np.cos(t)
        else:
            self._fade_in = np.array([], dtype=np.float32)
            self._fade_out = np.array([], dtype=np.float32)

    def _recalculate_buffers(self) -> None:
        """Recalculate buffer sizes based on current config sample rates."""
        # Input buffer sizes (at mic sample rate)
        self.mic_chunk_samples = int(self.config.mic_sample_rate * self.config.chunk_sec)
        self.mic_context_samples = int(self.config.mic_sample_rate * self.config.context_sec)
        self.mic_lookahead_samples = int(self.config.mic_sample_rate * self.config.lookahead_sec)
        self.mic_total_chunk = self.mic_chunk_samples + 2 * self.mic_context_samples

        # Reconfigure input buffer
        self.input_buffer = ChunkBuffer(
            self.mic_chunk_samples,
            crossfade_samples=0,
            context_samples=self.mic_context_samples,
            lookahead_samples=self.mic_lookahead_samples,
        )

        # Output processing parameters (at output sample rate)
        self.output_crossfade_samples = int(
            self.config.output_sample_rate * self.config.crossfade_sec
        )
        self.output_extra_samples = int(self.config.output_sample_rate * self.config.extra_sec)
        self.output_context_samples = int(self.config.output_sample_rate * self.config.context_sec)

        # Crossfade windows
        if self.output_crossfade_samples > 0:
            t = np.linspace(0, np.pi / 2, self.output_crossfade_samples, dtype=np.float32)
            self._fade_in = np.sin(t)
            self._fade_out = np.cos(t)
        else:
            self._fade_in = np.array([], dtype=np.float32)
            self._fade_out = np.array([], dtype=np.float32)

        # Update output history size
        self._output_history_size = self.config.output_sample_rate
        self._output_history = np.zeros(self._output_history_size, dtype=np.float32)
        self._output_history_pos = 0

        logger.debug(
            f"Buffers recalculated: mic_chunk={self.mic_chunk_samples}, "
            f"mic_context={self.mic_context_samples}, out_cf={self.output_crossfade_samples}"
        )

    def _store_output_history(self, output: np.ndarray) -> None:
        """Store output samples for feedback detection."""
        # Resample to mic rate if different for comparison
        if self.config.output_sample_rate != self.config.mic_sample_rate:
            output = resample(output, self.config.output_sample_rate, self.config.mic_sample_rate)

        # Store in circular buffer
        out_len = len(output)
        if out_len >= self._output_history_size:
            # Output larger than buffer - just store the last part
            self._output_history[:] = output[-self._output_history_size :]
            self._output_history_pos = 0
        else:
            # Fit into circular buffer
            end_pos = self._output_history_pos + out_len
            if end_pos <= self._output_history_size:
                self._output_history[self._output_history_pos : end_pos] = output
            else:
                # Wrap around
                first_part = self._output_history_size - self._output_history_pos
                self._output_history[self._output_history_pos :] = output[:first_part]
                self._output_history[: out_len - first_part] = output[first_part:]
            self._output_history_pos = end_pos % self._output_history_size

    def _check_feedback(self, input_audio: np.ndarray) -> float:
        """
        Check for feedback by computing cross-correlation between input and output history.

        Returns correlation coefficient (0-1). High values (>0.5) indicate feedback.
        """
        if len(input_audio) < 1000:
            return 0.0

        # Only check if there's significant signal in input
        input_rms = np.sqrt(np.mean(input_audio**2))
        if input_rms < 0.01:  # Too quiet to detect
            return 0.0

        output_rms = np.sqrt(np.mean(self._output_history**2))
        if output_rms < 0.01:  # No output yet
            return 0.0

        # Normalize both signals
        input_norm = input_audio - np.mean(input_audio)
        output_norm = self._output_history - np.mean(self._output_history)

        input_std = np.std(input_norm)
        output_std = np.std(output_norm)

        if input_std < 1e-6 or output_std < 1e-6:
            return 0.0

        input_norm = input_norm / input_std
        output_norm = output_norm / output_std

        # Compute cross-correlation using FFT for efficiency
        # Look for correlation at various delays (100ms to 1s)
        try:
            # Use a subset of output history for speed
            check_len = min(len(input_audio), self.config.mic_sample_rate // 2)
            corr = np.correlate(input_norm[:check_len], output_norm[:check_len], mode="valid")
            max_corr = np.max(np.abs(corr)) / check_len
            return float(max_corr)
        except Exception:
            return 0.0

    def _on_audio_input(self, audio: np.ndarray) -> None:
        """Callback for audio input."""
        # Debug: log input audio signature to detect feedback
        if self.stats.frames_processed < 20:
            audio_hash = hash(audio[:100].tobytes()) % 10000
            rms = np.sqrt(np.mean(audio**2))
            logger.info(
                f"[INPUT-RAW] len={len(audio)}, hash={audio_hash}, "
                f"rms={rms:.6f}, first5={audio[:5]}"
            )

        # Add raw audio to buffer (resampling done in inference thread)
        self.input_buffer.add_input(audio)

        # Log input buffer state periodically
        if self.stats.frames_processed < 5 or self.stats.frames_processed % 50 == 0:
            logger.debug(
                f"[INPUT] received={len(audio)}, "
                f"input_buffer={self.input_buffer.buffered_samples}, "
                f"input_queue={self._input_queue.qsize()}"
            )

        # Process ALL available chunks (not just one) to prevent input buffer buildup
        chunks_queued = 0
        while self.input_buffer.has_chunk():
            chunk = self.input_buffer.get_chunk()
            if chunk is not None:
                try:
                    self._input_queue.put_nowait(chunk)
                    chunks_queued += 1
                except Exception:
                    self.stats.buffer_overruns += 1
                    logger.warning(f"[INPUT] Queue full, dropping chunk")
                    break  # Queue full, stop trying

        # Debug: log if multiple chunks were queued (indicates we're falling behind)
        if chunks_queued > 1:
            logger.warning(f"[INPUT] Falling behind: queued {chunks_queued} chunks at once")

    def _on_audio_output(self, frames: int) -> np.ndarray:
        """Callback for audio output."""
        # Check for new processed audio
        chunks_added = 0
        total_dropped = 0
        try:
            while True:
                audio = self._output_queue.get_nowait()
                dropped = self.output_buffer.add(audio)
                self._chunks_ready += 1
                chunks_added += 1
                if dropped > 0:
                    # Old samples were dropped to catch up to real-time
                    self.stats.buffer_overruns += 1
                    total_dropped += dropped
        except Empty:
            pass

        # Log output state periodically
        if self.stats.frames_processed < 10 or self.stats.frames_processed % 50 == 0:
            logger.debug(
                f"[OUTPUT] frames={frames}, chunks_added={chunks_added}, "
                f"output_buffer={self.output_buffer.available}, "
                f"output_queue={self._output_queue.qsize()}, "
                f"dropped={total_dropped}"
            )

        # Wait for pre-buffering before starting output
        if not self._output_started:
            if self._chunks_ready >= self._prebuffer_chunks:
                self._output_started = True
                logger.info(
                    f"Pre-buffering complete, starting output ({self.output_buffer.available} samples)"
                )
            else:
                # Return silence while pre-buffering
                return np.zeros(frames, dtype=np.float32)

        # Get output samples
        output = self.output_buffer.get(frames)

        if self.output_buffer.available == 0:
            self.stats.buffer_underruns += 1
            if self.stats.buffer_underruns <= 5 or self.stats.buffer_underruns % 20 == 0:
                logger.warning(f"[OUTPUT] Buffer underrun #{self.stats.buffer_underruns}")

        return output

    def _inference_thread(self) -> None:
        """Background thread for inference processing."""
        logger.info("Inference thread started")

        # Log audio flow for debugging sample rate issues
        logger.info(
            f"Audio flow: mic({self.config.mic_sample_rate}Hz) -> "
            f"process({self.config.input_sample_rate}Hz) -> "
            f"model({self.pipeline.sample_rate}Hz) -> "
            f"output({self.config.output_sample_rate}Hz)"
        )
        logger.info(
            f"w-okada style: context={self.config.context_sec}s, "
            f"extra={self.config.extra_sec}s, crossfade={self.config.crossfade_sec}s, "
            f"sola={self.config.use_sola}"
        )

        if self.config.input_sample_rate != 16000:
            logger.warning(
                f"Input sample rate {self.config.input_sample_rate}Hz differs from "
                "expected 16kHz for HuBERT/RMVPE"
            )

        # Calculate output trim amounts for w-okada style
        # Input: [left_context | main | right_context]
        # Output: keep only main, discard both contexts
        out_context_samples = self.output_context_samples
        out_extra_samples = self.output_extra_samples

        while self._running:
            try:
                # Get input chunk (at mic sample rate)
                chunk = self._input_queue.get(timeout=0.5)

                # Process timing
                start_time = time.perf_counter()

                # Apply input gain
                if self.config.input_gain_db != 0.0:
                    gain_linear = 10 ** (self.config.input_gain_db / 20)
                    chunk = chunk * gain_linear

                # Store raw input chunk at mic rate for feedback detection
                chunk_at_mic_rate = chunk.copy()

                # Debug: log input chunk stats
                if self.stats.frames_processed < 3:
                    rms = np.sqrt(np.mean(chunk**2))
                    # ChunkBuffer returns main + 2*context (left_ctx + main + right_ctx)
                    expected_len = self.mic_chunk_samples + 2 * self.mic_context_samples
                    logger.info(
                        f"Raw input chunk: len={len(chunk)} (expected={expected_len}, "
                        f"main={self.mic_chunk_samples}, context={self.mic_context_samples}x2), "
                        f"min={chunk.min():.4f}, max={chunk.max():.4f}, rms={rms:.4f}"
                    )

                # Resample from mic rate to processing rate
                if self.config.mic_sample_rate != self.config.input_sample_rate:
                    chunk = resample(
                        chunk,
                        self.config.mic_sample_rate,
                        self.config.input_sample_rate,
                    )
                    if self.stats.frames_processed < 3:
                        logger.info(
                            f"After resample: len={len(chunk)}, "
                            f"min={chunk.min():.4f}, max={chunk.max():.4f}"
                        )

                # Apply noise cancellation if enabled
                if self.config.denoise_enabled:
                    if self.stats.frames_processed < 3:
                        logger.info(f"Applying denoise (method={self.config.denoise_method})")
                    chunk = denoise_audio(
                        chunk,
                        sample_rate=self.config.input_sample_rate,
                        method=self.config.denoise_method,
                        device="cpu",  # ML denoiser runs on CPU for stability
                    )
                    if self.stats.frames_processed < 3:
                        logger.info(
                            f"After denoise: len={len(chunk)}, "
                            f"min={chunk.min():.4f}, max={chunk.max():.4f}"
                        )

                # Run inference
                output = self.pipeline.infer(
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
                )

                # Resample to output sample rate
                if self.pipeline.sample_rate != self.config.output_sample_rate:
                    output = resample(
                        output,
                        self.pipeline.sample_rate,
                        self.config.output_sample_rate,
                        method=self.config.resample_method,
                    )

                # Soft clipping to prevent harsh distortion
                max_val = np.max(np.abs(output))
                if max_val > 1.0:
                    output = np.tanh(output)  # Soft clip
                    if self.stats.frames_processed <= 5:
                        logger.warning(f"Audio clipping detected: max={max_val:.2f}")

                # Apply SOLA crossfade if enabled, OR trim context manually
                # w-okada mode SOLA: Uses left context for crossfading, returns trimmed output
                # If SOLA is disabled, we manually trim context
                if self.config.use_sola and self._sola_state is not None:
                    # Calculate context size for w-okada mode
                    # Skip context trim for first chunk (no left context)
                    context_samples_output = 0
                    if self.stats.frames_processed > 0 and self.config.context_sec > 0:
                        context_samples_output = int(
                            self.config.output_sample_rate * self.config.context_sec
                        )

                    # w-okada mode SOLA: crossfade with context, auto-trim
                    cf_result = apply_sola_crossfade(
                        output,
                        self._sola_state,
                        wokada_mode=True,
                        context_samples=context_samples_output
                    )
                    output = cf_result.audio

                    # Log SOLA stats periodically
                    if self.stats.frames_processed % 50 == 0:
                        logger.debug(
                            f"[SOLA] chunk={self.stats.frames_processed}, "
                            f"offset={cf_result.sola_offset}"
                        )
                else:
                    # No SOLA: Manually trim context from output to get only the "main" portion
                    # This is critical for matching batch processing output length
                    # Skip trimming for first chunk (no left context)
                    if self.stats.frames_processed > 0 and self.config.context_sec > 0:
                        context_samples_output = int(
                            self.config.output_sample_rate * self.config.context_sec
                        )
                        if len(output) > context_samples_output:
                            output = output[context_samples_output:]
                            # Log trimming for first few chunks
                            if self.stats.frames_processed < 5:
                                logger.info(
                                    f"[TRIM] Chunk #{self.stats.frames_processed}: trimmed {context_samples_output} samples from start"
                                )

                # Debug: log output audio signature to detect feedback
                if self.stats.frames_processed < 20:
                    output_hash = hash(output[:100].tobytes()) % 10000
                    output_rms = np.sqrt(np.mean(output**2))
                    logger.info(
                        f"[OUTPUT-PROC] len={len(output)}, hash={output_hash}, "
                        f"rms={output_rms:.6f}, first5={output[:5]}"
                    )

                # Store output for feedback detection
                self._store_output_history(output)

                # Check for feedback periodically
                if (
                    self.stats.frames_processed > 0
                    and self.stats.frames_processed % self._feedback_check_interval == 0
                ):
                    # Get raw input chunk for comparison (at mic rate)
                    raw_input = chunk_at_mic_rate if "chunk_at_mic_rate" in locals() else chunk
                    correlation = self._check_feedback(raw_input)
                    self.stats.feedback_correlation = correlation

                    if correlation > 0.3 and not self._feedback_warning_shown:
                        self.stats.feedback_detected = True
                        self._feedback_warning_shown = True
                        logger.warning(
                            f"[FEEDBACK] 音声フィードバックを検出しました (相関係数={correlation:.2f}). "
                            "Windowsの「このデバイスを聴く」が有効になっていないか確認してください。"
                        )
                        if self.on_error:
                            self.on_error(
                                "フィードバック検出: 入力と出力が接続されている可能性があります。\n"
                                "「サウンド設定」→「録音デバイス」→「プロパティ」→「聴く」タブで\n"
                                "「このデバイスを聴く」が無効になっているか確認してください。"
                            )

                # Update stats
                inference_time = time.perf_counter() - start_time
                self.stats.inference_ms = inference_time * 1000
                self.stats.frames_processed += 1

                # Log chunks for debugging
                if self.stats.frames_processed <= 10 or self.stats.frames_processed % 20 == 0:
                    logger.info(
                        f"[INFER] Chunk #{self.stats.frames_processed}: "
                        f"in={len(chunk)}, out={len(output)}, "
                        f"infer={self.stats.inference_ms:.0f}ms, "
                        f"f0_method={self.config.f0_method}, "
                        f"latency={self.stats.latency_ms:.0f}ms, "
                        f"buf={self.output_buffer.available}, "
                        f"under={self.stats.buffer_underruns}, over={self.stats.buffer_overruns}"
                    )

                # Calculate latency
                self.stats.latency_ms = (
                    self.config.chunk_sec * 1000
                    + self.stats.inference_ms
                    + (self.output_buffer.available / self.config.output_sample_rate) * 1000
                )

                # Send to output (block briefly if queue is full)
                try:
                    self._output_queue.put(output, timeout=0.1)
                except Exception:
                    self.stats.buffer_overruns += 1
                    logger.warning("Output queue full, dropping chunk")

                # Call stats callback
                if self.on_stats_update:
                    self.on_stats_update(self.stats)

            except Empty:
                continue
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Inference error: {error_msg}")

                # Provide helpful error message for common issues
                if (
                    "Output size is too small" in error_msg
                    or "size" in error_msg.lower()
                    and "0" in error_msg
                ):
                    user_msg = f"チャンクサイズが小さすぎます。バッファを350ms以上に設定してください。(技術詳細: {error_msg})"
                else:
                    user_msg = f"推論エラー: {error_msg}"

                # Notify UI of error (throttle to avoid spam)
                self._error_count += 1
                if self._last_error != error_msg or self._error_count % 10 == 1:
                    self._last_error = error_msg
                    if self.on_error:
                        self.on_error(user_msg)
                continue

        # Flush remaining SOLA buffer before exiting
        if self.config.use_sola and self._sola_state is not None:
            final_buffer = flush_sola_buffer(self._sola_state)
            if len(final_buffer) > 0:
                try:
                    self._output_queue.put(final_buffer, timeout=0.5)
                    logger.info(f"Flushed final SOLA buffer: {len(final_buffer)} samples")
                except Exception:
                    logger.warning("Failed to flush final SOLA buffer (queue full)")

        logger.info("Inference thread stopped")

    def start(self) -> None:
        """Start real-time voice conversion."""
        if self._running:
            return

        logger.info("Starting real-time voice changer...")

        # Clear feature cache for fresh start
        self.pipeline.clear_cache()

        # Validate chunk size based on F0 method
        # FCPE needs >= 100ms, RMVPE needs >= 320ms
        if self.config.use_f0:
            if self.config.f0_method == "fcpe":
                min_chunk_sec = 0.10
            elif self.config.f0_method == "rmvpe":
                min_chunk_sec = 0.32
            else:
                min_chunk_sec = 0.10  # Default to FCPE requirement

            if self.config.chunk_sec < min_chunk_sec:
                old_chunk = self.config.chunk_sec
                self.config.chunk_sec = min_chunk_sec + 0.03  # Add small margin
                logger.warning(
                    f"Chunk size {old_chunk}s too small for {self.config.f0_method}, "
                    f"increased to {self.config.chunk_sec}s"
                )

        # Recalculate buffer sizes
        self._recalculate_buffers()

        # Pipeline already loaded and warmed up in __init__, no need to do it again

        # Reset stats and buffers
        self.stats.reset()
        self.input_buffer.clear()
        self.output_buffer.clear()

        # Reset pre-buffering state
        self._chunks_ready = 0
        self._output_started = False
        self._sola_state = SOLAState.create(
            self.output_crossfade_samples,
            self.config.output_sample_rate,
        )

        # Reset feedback detection state
        self._output_history.fill(0)
        self._output_history_pos = 0
        self._feedback_warning_shown = False

        # Clear queues and reset pre-buffering state
        self._clear_queues()
        self._chunks_ready = 0
        self._output_started = False

        # Start inference thread
        self._running = True
        self._thread = threading.Thread(
            target=self._inference_thread,
            daemon=True,
            name="RCWX-Inference",
        )
        self._thread.start()

        # Calculate output blocksize
        output_chunk_sec = self.config.chunk_sec / 4
        output_blocksize = int(self.config.output_sample_rate * output_chunk_sec)

        # Calculate processing overhead ratio
        # We process main + 2*context, but output only main
        total_input = self.mic_chunk_samples + 2 * self.mic_context_samples
        processing_ratio = total_input / self.mic_chunk_samples

        # Start audio input at mic's native rate (resample to 16kHz in callback)
        logger.info(
            f"Audio config: mic_sr={self.config.mic_sample_rate}, "
            f"out_sr={self.config.output_sample_rate}, "
            f"chunk={self.config.chunk_sec}s ({self.mic_chunk_samples} samples), "
            f"context={self.config.context_sec}s ({self.mic_context_samples} samples each side), "
            f"extra={self.config.extra_sec}s, processing={processing_ratio:.1f}x, sola={self.config.use_sola}"
        )

        self.audio_input = AudioInput(
            device=self.config.input_device,
            sample_rate=self.config.mic_sample_rate,
            channels=self.config.input_channels,
            blocksize=int(self.config.mic_sample_rate * output_chunk_sec),
            callback=self._on_audio_input,
            channel_selection=self.config.input_channel_selection,
        )
        self.audio_input.start()

        # Check if actual sample rate differs from requested
        if self.audio_input.actual_sample_rate != self.config.mic_sample_rate:
            logger.warning(
                f"Input sample rate changed: {self.config.mic_sample_rate}Hz -> "
                f"{self.audio_input.actual_sample_rate}Hz (recalculating buffers)"
            )
            self.config.mic_sample_rate = self.audio_input.actual_sample_rate
            self._recalculate_buffers()

        # Start audio output
        self.audio_output = AudioOutput(
            device=self.config.output_device,
            sample_rate=self.config.output_sample_rate,
            blocksize=output_blocksize,
            callback=self._on_audio_output,
        )
        self.audio_output.start()

        # Check if actual sample rate differs from requested
        if self.audio_output.actual_sample_rate != self.config.output_sample_rate:
            logger.warning(
                f"Output sample rate changed: {self.config.output_sample_rate}Hz -> "
                f"{self.audio_output.actual_sample_rate}Hz (recalculating buffers)"
            )
            self.config.output_sample_rate = self.audio_output.actual_sample_rate
            self._recalculate_buffers()

        logger.info("Real-time voice changer started")

    def stop(self) -> None:
        """Stop real-time voice conversion."""
        if not self._running:
            return

        logger.info("Stopping real-time voice changer...")

        self._running = False

        # Stop audio streams
        if self.audio_input:
            self.audio_input.stop()
            self.audio_input = None

        if self.audio_output:
            self.audio_output.stop()
            self.audio_output = None

        # Wait for inference thread
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

        # Clear queues to prevent stale data
        self._clear_queues()

        logger.info("Real-time voice changer stopped")

    def _clear_queues(self) -> None:
        """Clear input and output queues."""
        # Drain input queue
        while not self._input_queue.empty():
            try:
                self._input_queue.get_nowait()
            except Empty:
                break

        # Drain output queue
        while not self._output_queue.empty():
            try:
                self._output_queue.get_nowait()
            except Empty:
                break

        logger.debug("Queues cleared")

    def set_pitch_shift(self, semitones: int) -> None:
        """Set pitch shift in semitones."""
        self.config.pitch_shift = semitones

    def set_f0_mode(self, use_f0: bool) -> None:
        """Set F0 mode (True for RMVPE, False for no-F0)."""
        self.config.use_f0 = use_f0

    def set_f0_method(self, method: str) -> None:
        """Set F0 extraction method.

        Args:
            method: "rmvpe" (accurate, 320ms min) or "fcpe" (fast, 100ms min)
        """
        self.config.f0_method = method

    def set_denoise(self, enabled: bool, method: str = "auto") -> None:
        """Set noise cancellation settings.

        Args:
            enabled: Enable/disable noise cancellation
            method: "auto", "deepfilter", or "spectral"
        """
        self.config.denoise_enabled = enabled
        self.config.denoise_method = method

    def set_index_rate(self, index_rate: float) -> None:
        """Set FAISS index blending rate.

        Args:
            index_rate: 0=disabled, 0.5=balanced, 1=index only
        """
        self.config.index_rate = index_rate

    def set_voice_gate_mode(self, mode: str) -> None:
        """Set voice gate mode.

        Args:
            mode: "off", "strict", "expand", or "energy"
        """
        self.config.voice_gate_mode = mode

    def set_energy_threshold(self, threshold: float) -> None:
        """Set energy threshold for energy mode.

        Args:
            threshold: Energy threshold (0.01-0.2)
        """
        self.config.energy_threshold = threshold

    def set_feature_cache(self, enabled: bool) -> None:
        """Set feature caching for chunk continuity.

        Args:
            enabled: Enable/disable HuBERT/F0 feature caching
        """
        self.config.use_feature_cache = enabled
        if not enabled:
            self.pipeline.clear_cache()

    def set_context(self, context_sec: float) -> None:
        """Set context size for w-okada style processing.

        Args:
            context_sec: Context duration in seconds (each side)
        """
        self.config.context_sec = max(0.0, min(0.3, context_sec))

        # Apply immediately if running
        if self._running:
            self.mic_context_samples = int(self.config.mic_sample_rate * self.config.context_sec)
            # Recreate input buffer with new context
            self.input_buffer = ChunkBuffer(
                self.mic_chunk_samples,
                crossfade_samples=0,
                context_samples=self.mic_context_samples,
                lookahead_samples=self.mic_lookahead_samples,
            )
            logger.info(f"Context applied: {self.config.context_sec}s")

    def set_extra(self, extra_sec: float) -> None:
        """Set extra edge discard.

        Args:
            extra_sec: Extra discard duration in seconds
        """
        self.config.extra_sec = max(0.0, min(0.1, extra_sec))
        logger.info(f"Extra discard set to {self.config.extra_sec}s")

    def set_lookahead(self, lookahead_sec: float) -> None:
        """Set lookahead (future samples for right context).

        WARNING: This adds latency! Only use if edge quality is poor.

        Args:
            lookahead_sec: Lookahead duration in seconds (0 = disabled)
        """
        self.config.lookahead_sec = max(0.0, min(0.1, lookahead_sec))

        # Apply immediately if running
        if self._running:
            self.mic_lookahead_samples = int(
                self.config.mic_sample_rate * self.config.lookahead_sec
            )
            # Recreate input buffer with new lookahead
            self.input_buffer = ChunkBuffer(
                self.mic_chunk_samples,
                crossfade_samples=0,
                context_samples=self.mic_context_samples,
                lookahead_samples=self.mic_lookahead_samples,
            )
            logger.info(
                f"Lookahead applied: {self.config.lookahead_sec}s (+{int(self.config.lookahead_sec * 1000)}ms latency)"
            )

    def set_sola(self, enabled: bool) -> None:
        """Set SOLA (Synchronized Overlap-Add) for optimal crossfade position.

        Args:
            enabled: Enable/disable SOLA
        """
        self.config.use_sola = enabled

        # Reset SOLA state when toggling
        if self._running:
            if enabled:
                self._sola_state = SOLAState.create(
                    self.output_crossfade_samples,
                    self.config.output_sample_rate,
                )
            else:
                self._sola_state = None

        logger.info(f"SOLA {'enabled' if enabled else 'disabled'}")

    def set_crossfade(self, crossfade_sec: float) -> None:
        """Set crossfade length for chunk boundaries.

        Args:
            crossfade_sec: Crossfade duration in seconds
        """
        self.config.crossfade_sec = max(0.0, min(0.2, crossfade_sec))

        # Apply immediately if running
        if self._running:
            self.output_crossfade_samples = int(
                self.config.output_sample_rate * self.config.crossfade_sec
            )
            if self.config.use_sola:
                self._sola_state = SOLAState.create(
                    self.output_crossfade_samples,
                    self.config.output_sample_rate,
                )

        logger.info(f"Crossfade set to {self.config.crossfade_sec}s")

    def set_prebuffer_chunks(self, chunks: int) -> None:
        """Set pre-buffer chunk count.

        Args:
            chunks: Number of chunks to buffer before starting playback (0-3)
        """
        self.config.prebuffer_chunks = max(0, min(3, int(chunks)))

        # Apply immediately if running
        if self._running:
            self._prebuffer_chunks = self.config.prebuffer_chunks
            # If we increase prebuffer while running, reset output start
            if not self._output_started and self._chunks_ready < self._prebuffer_chunks:
                self._output_started = False

        logger.info(f"Pre-buffer chunks set to {self.config.prebuffer_chunks}")

    def set_buffer_margin(self, margin: float) -> None:
        """Set output buffer margin multiplier.

        Args:
            margin: Buffer margin multiplier (0.3-2.0)
        """
        self.config.buffer_margin = max(0.3, min(2.0, margin))

        # Apply immediately if running
        if self._running:
            max_latency_sec = self.config.chunk_sec * (
                self.config.prebuffer_chunks + self.config.buffer_margin
            )
            max_latency_samples = int(self.config.output_sample_rate * max_latency_sec)
            self.output_buffer.set_max_latency(max_latency_samples)

        logger.info(f"Buffer margin set to {self.config.buffer_margin}x")

    def set_chunk_sec(self, chunk_sec: float) -> None:
        """Set chunk size in seconds.

        Automatically restarts audio streams if running.

        Args:
            chunk_sec: Chunk duration in seconds (0.1-0.6)
        """
        old_chunk = self.config.chunk_sec
        self.config.chunk_sec = max(0.1, min(0.6, chunk_sec))

        if self._running:
            logger.info(
                f"Chunk size changed ({old_chunk}s -> {self.config.chunk_sec}s), "
                "restarting audio streams..."
            )
            # Restart audio streams to apply new chunk size
            self.stop()
            self.start()
        else:
            logger.info(f"Chunk size set to {self.config.chunk_sec}s")

    @property
    def is_running(self) -> bool:
        """Check if voice changer is running."""
        return self._running

    # ========== Public Testing Methods ==========

    def process_input_chunk(self, audio: np.ndarray) -> None:
        """
        Add input audio and queue available chunks for processing.

        This is a public method for testing that replicates the behavior
        of _on_audio_input without requiring an actual audio stream.

        Args:
            audio: Input audio at mic_sample_rate
        """
        # Add raw audio to buffer
        self.input_buffer.add_input(audio)

        # Queue all available chunks
        while self.input_buffer.has_chunk():
            chunk = self.input_buffer.get_chunk()
            if chunk is not None:
                try:
                    self._input_queue.put_nowait(chunk)
                except Exception:
                    logger.warning("Input queue full, dropping chunk")
                    break

    def process_next_chunk(self) -> bool:
        """
        Process one chunk from input queue to output queue.

        This is a public method for testing that replicates one iteration
        of _inference_thread without running in a separate thread.

        Returns:
            True if a chunk was processed, False if queue was empty
        """
        try:
            # Get input chunk (at mic sample rate) - non-blocking
            chunk = self._input_queue.get_nowait()
        except Empty:
            return False

        # Apply input gain
        if self.config.input_gain_db != 0.0:
            gain_linear = 10 ** (self.config.input_gain_db / 20)
            chunk = chunk * gain_linear

        # Store for feedback detection (optional in tests)
        chunk_at_mic_rate = chunk.copy()

        # Resample from mic rate to processing rate
        if self.config.mic_sample_rate != self.config.input_sample_rate:
            chunk = resample(
                chunk,
                self.config.mic_sample_rate,
                self.config.input_sample_rate,
            )

        # Apply noise cancellation if enabled
        if self.config.denoise_enabled:
            chunk = denoise_audio(
                chunk,
                sample_rate=self.config.input_sample_rate,
                method=self.config.denoise_method,
                device="cpu",
            )

        # Run inference
        output = self.pipeline.infer(
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
        )

        # Resample to output sample rate
        if self.pipeline.sample_rate != self.config.output_sample_rate:
            output = resample(
                output,
                self.pipeline.sample_rate,
                self.config.output_sample_rate,
                method=self.config.resample_method,
            )

        # Soft clipping
        max_val = np.max(np.abs(output))
        if max_val > 1.0:
            output = np.tanh(output)

        # Apply SOLA crossfade if enabled, OR trim context manually
        # w-okada mode SOLA: Uses left context for crossfading, returns trimmed output
        # If SOLA is disabled, we manually trim context
        if self.config.use_sola and self._sola_state is not None:
            # Calculate context size for w-okada mode
            # Skip context trim for first chunk (no left context)
            context_samples_output = 0
            if self.stats.frames_processed > 0 and self.config.context_sec > 0:
                context_samples_output = int(
                    self.config.output_sample_rate * self.config.context_sec
                )

            # w-okada mode SOLA: crossfade with context, auto-trim
            cf_result = apply_sola_crossfade(
                output,
                self._sola_state,
                wokada_mode=True,
                context_samples=context_samples_output
            )
            output = cf_result.audio
        else:
            # No SOLA: Manually trim context from output to get only the "main" portion
            # This is critical for matching batch processing output length
            # Skip trimming for first chunk (no left context)
            if self.stats.frames_processed > 0 and self.config.context_sec > 0:
                context_samples_output = int(self.config.output_sample_rate * self.config.context_sec)
                if len(output) > context_samples_output:
                    output = output[context_samples_output:]
                    # Log trimming for first few chunks
                    if self.stats.frames_processed < 5:
                        logger.info(
                            f"[TRIM] Chunk #{self.stats.frames_processed}: trimmed {context_samples_output} samples from start"
                        )

        # Store output history for feedback detection
        self._store_output_history(output)

        # Update stats
        self.stats.frames_processed += 1

        # Send to output queue
        try:
            self._output_queue.put_nowait(output)
        except Exception:
            logger.warning("Output queue full, dropping chunk")

        return True

    def get_output_chunk(self, frames: int) -> np.ndarray:
        """
        Get output audio from buffer.

        This is a public method for testing that replicates the behavior
        of _on_audio_output without requiring an actual audio stream.

        Args:
            frames: Number of samples to retrieve

        Returns:
            Output audio at output_sample_rate
        """
        # Check for new processed audio in queue
        try:
            while True:
                audio = self._output_queue.get_nowait()
                self.output_buffer.add(audio)
                self._chunks_ready += 1
        except Empty:
            pass

        # Check pre-buffering
        if not self._output_started:
            if self._chunks_ready >= self._prebuffer_chunks:
                self._output_started = True
            else:
                # Return silence while pre-buffering
                return np.zeros(frames, dtype=np.float32)

        # Get output samples
        output = self.output_buffer.get(frames)

        if self.output_buffer.available == 0:
            self.stats.buffer_underruns += 1

        return output
