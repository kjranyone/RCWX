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
    buffer_trims: int = 0
    frames_processed: int = 0
    feedback_detected: bool = False
    feedback_correlation: float = 0.0

    def reset(self) -> None:
        self.latency_ms = 0.0
        self.inference_ms = 0.0
        self.buffer_underruns = 0
        self.buffer_overruns = 0
        self.buffer_trims = 0
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
    buffer_margin: float = 0.5

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
    voice_gate_mode: str = "off"
    energy_threshold: float = 0.05

    # SOLA
    use_sola: bool = True

    # HuBERT context window in seconds (longer = more stable timbre across chunks)
    hubert_context_sec: float = 1.0

    # Pre-HuBERT pitch shift ratio (0.0=disabled, 1.0=full pitch shift before HuBERT)
    pre_hubert_pitch_ratio: float = 0.0
    # Moe voice style strength (0.0=off, 1.0=strong)
    moe_boost: float = 0.0

    # WAV file input (empty string = use microphone)
    wav_input_path: str = ""

    # Synthesizer
    synth_min_frames: int = 64

    # Pitch clarity
    noise_scale: float = 0.4
    f0_lowpass_cutoff_hz: float = 16.0
    enable_octave_flip_suppress: bool = True
    enable_f0_slew_limit: bool = True
    f0_slew_max_step_st: float = 2.8

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

        # Actual stream rates can differ from requested config rates
        # (for example 48kHz request falling back to 44.1kHz).
        self._runtime_mic_sample_rate = int(self.config.mic_sample_rate)
        self._runtime_output_sample_rate = int(self.config.output_sample_rate)

        # Calculate sample counts (all at 16kHz for model input)
        hubert_hop = 320
        self._hop_samples_16k = self._align_to_hop(
            int(self.config.chunk_sec * 16000), hubert_hop
        )
        self._overlap_samples_16k = self._align_to_hop(
            int(self.config.overlap_sec * 16000), hubert_hop
        )

        # Corresponding counts at runtime stream rates
        self._hop_samples_mic = int(
            self._hop_samples_16k * self._runtime_mic_sample_rate / 16000
        )
        self._hop_samples_out = int(
            self._hop_samples_16k * self._runtime_output_sample_rate / 16000
        )

        # Resamplers
        self.input_resampler = StatefulResampler(
            self._runtime_mic_sample_rate, 16000
        )
        self.output_resampler = StatefulResampler(
            self.pipeline.sample_rate, self._runtime_output_sample_rate
        )

        # SOLA state
        crossfade_samples_out = int(
            self._runtime_output_sample_rate * self.config.crossfade_sec
        )
        search_samples_out = int(
            self._runtime_output_sample_rate * self.config.sola_search_ms / 1000
        )
        self._sola_state = SolaState(
            crossfade_samples=crossfade_samples_out,
            search_samples=search_samples_out,
        )

        # SOLA extra: produce additional output samples (at model_sr) so
        # SOLA has a full crossfade+search region that overlaps with the
        # previous chunk's tail.  Worst-case SOLA needs cf+search+target_len
        # from the audio, and we produce hop+sola_extra.  Add an extra
        # search_samples_out as margin so SOLA target_len always succeeds
        # even at the maximum search offset.
        # Round up to model's zc boundary (= sample_rate // 100) so that
        # trim_left in infer_streaming is always a multiple of
        # samples_per_frame, eliminating sub-frame residual trim.
        sola_extra_out = crossfade_samples_out + 2 * search_samples_out
        zc_model = self.pipeline.sample_rate // 100
        sola_extra_raw = int(
            sola_extra_out * self.pipeline.sample_rate
            / self._runtime_output_sample_rate
        )
        self._sola_extra_model = (
            (sola_extra_raw + zc_model - 1) // zc_model * zc_model
        )

        # Output buffer: 4x chunk capacity (physical ring size)
        chunk_output_samples = int(
            self._runtime_output_sample_rate * self.config.chunk_sec
        )
        self.output_buffer = RingOutputBuffer(
            capacity_samples=chunk_output_samples * 4,
            fade_samples=256,
        )

        # Adaptive drift control target.
        #
        # The output callback runs 4x per chunk (blocksize = hop/4).
        # After an inference burst fills the ring, successive callbacks
        # drain it:  3/4 hop → 2/4 → 1/4 → 0 → (next burst) → 3/4 …
        # Natural steady-state peak = hop * 3/4.
        #
        # When the ring exceeds this target, we read *extra* samples and
        # resample (np.interp) to the requested frame count.  This speeds
        # up playback by an imperceptible amount (≤1.5%) instead of
        # skipping audio, which would cause clicks.
        self._drain_target = self._compute_drain_target()

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

        # Audio streams (AudioInput or WavFileInput)
        self._input_stream = None
        self._output_stream: Optional[AudioOutput] = None

        # Feedback detection
        self._output_history_size = self._runtime_mic_sample_rate  # 1 second
        self._output_history = np.zeros(self._output_history_size, dtype=np.float32)
        self._output_history_pos = 0
        self._feedback_check_interval = 10
        self._feedback_warning_shown = False

        # Boundary gain continuity across emitted chunks (post-SOLA).
        self._prev_tail_rms: float = 0.0

        # Cached arrays for adaptive drift np.interp (avoid per-callback alloc)
        self._interp_cache_frames: int = 0
        self._interp_cache_x_base: Optional[np.ndarray] = None

        # Overload protection
        self._queue_full_times: deque[float] = deque(maxlen=50)
        self._overload_until: float = 0.0
        self._overload_active = False

    @staticmethod
    def _align_to_hop(samples: int, hop: int) -> int:
        """Round sample count up to nearest multiple of hop."""
        return ((samples + hop - 1) // hop) * hop

    def _compute_drain_target_for_frames(self, frames: int) -> int:
        """Compute drift-control target for the current callback frame size."""
        frame_count = max(1, int(frames))
        # Natural post-callback level after one hop-sized enqueue.
        base_target = max(0, self._hop_samples_out - frame_count)
        margin = max(0.1, min(2.0, float(self.config.buffer_margin)))
        # 0.5 = neutral (natural target), <0.5 tighter, >0.5 more relaxed.
        margin_offset = int((margin - 0.5) * self._hop_samples_out)
        return max(0, base_target + margin_offset)

    def _compute_drain_target(self) -> int:
        """Compute nominal drift-control target (for non-callback contexts)."""
        nominal_frames = max(1, self._hop_samples_out // 4)
        return self._compute_drain_target_for_frames(nominal_frames)

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
            self._hop_samples_16k * self._runtime_mic_sample_rate / 16000
        )
        self._hop_samples_out = int(
            self._hop_samples_16k * self._runtime_output_sample_rate / 16000
        )

        # SOLA extra samples
        crossfade_samples_out = int(
            self._runtime_output_sample_rate * self.config.crossfade_sec
        )
        search_samples_out = int(
            self._runtime_output_sample_rate * self.config.sola_search_ms / 1000
        )
        self._sola_state.crossfade_samples = crossfade_samples_out
        self._sola_state.search_samples = search_samples_out
        self._sola_state._hann_fade_in = None
        self._sola_state._hann_fade_out = None
        sola_extra_out = crossfade_samples_out + 2 * search_samples_out
        zc_model = self.pipeline.sample_rate // 100
        sola_extra_raw = int(
            sola_extra_out * self.pipeline.sample_rate
            / self._runtime_output_sample_rate
        )
        self._sola_extra_model = (
            (sola_extra_raw + zc_model - 1) // zc_model * zc_model
        )

        # Adaptive drift control target
        self._drain_target = self._compute_drain_target()

    def _apply_runtime_sample_rates(self, mic_rate: int, output_rate: int) -> None:
        """Apply actual stream sample rates and rebuild dependent state."""
        self._runtime_mic_sample_rate = int(mic_rate)
        self._runtime_output_sample_rate = int(output_rate)
        self._recalculate_sizes()

        self.input_resampler = StatefulResampler(
            self._runtime_mic_sample_rate, 16000
        )
        self.output_resampler = StatefulResampler(
            self.pipeline.sample_rate, self._runtime_output_sample_rate
        )

        chunk_output_samples = max(
            1, int(self._runtime_output_sample_rate * self.config.chunk_sec)
        )
        self.output_buffer = RingOutputBuffer(
            capacity_samples=chunk_output_samples * 4,
            fade_samples=256,
        )

        self._output_history_size = max(1, self._runtime_mic_sample_rate)
        self._output_history = np.zeros(self._output_history_size, dtype=np.float32)
        self._output_history_pos = 0

    # ======== Lifecycle ========

    def start(self) -> None:
        if self._running:
            return

        # Reset runtime rates to configured values; may be corrected after stream open.
        self._apply_runtime_sample_rates(
            self.config.mic_sample_rate,
            self.config.output_sample_rate,
        )

        self.pipeline.clear_cache()
        self.stats.reset()
        self.output_buffer.clear()
        self._input_buf = np.array([], dtype=np.float32)
        self._overlap_buf = None
        self._reset_boundary_continuity_state()
        self._chunks_ready = 0
        self._output_started = False
        self.input_resampler.reset()
        self.output_resampler.reset()
        self._sola_state.buffer = None
        self._output_history.fill(0)
        self._output_history_pos = 0
        self._feedback_warning_shown = False
        self._clear_queues()
        self._run_denoise_warmup()
        self._run_runtime_warmup()

        # Audio stream block size: chunk/4 for responsive I/O
        output_chunk_sec = self.config.chunk_sec / 4
        output_blocksize = int(self.config.output_sample_rate * output_chunk_sec)

        # Start output stream FIRST so its callback is active before any
        # audio arrives.  AudioOutput.start() may take time due to API
        # fallback (WASAPI → DirectSound → MME), and during that time the
        # input stream would queue audio with no output drain — causing a
        # burst of buffered audio and large initial trims.
        self._output_stream = AudioOutput(
            device=self.config.output_device,
            sample_rate=self.config.output_sample_rate,
            channels=self.config.output_channels,
            blocksize=output_blocksize,
            callback=self._on_audio_output,
        )
        self._output_stream.start()

        if self.config.wav_input_path:
            from rcwx.audio.wav_input import WavFileInput

            self._input_stream = WavFileInput(
                path=self.config.wav_input_path,
                sample_rate=self.config.mic_sample_rate,
                blocksize=int(self.config.mic_sample_rate * output_chunk_sec),
                callback=self._on_audio_input,
            )
        else:
            self._input_stream = AudioInput(
                device=self.config.input_device,
                sample_rate=self.config.mic_sample_rate,
                channels=self.config.input_channels,
                blocksize=int(self.config.mic_sample_rate * output_chunk_sec),
                callback=self._on_audio_input,
                channel_selection=self.config.input_channel_selection,
            )
        self._input_stream.start()

        actual_output_rate = int(round(self._output_stream.actual_sample_rate))
        actual_mic_rate = int(round(self._input_stream.actual_sample_rate))
        if (
            actual_output_rate != self.config.output_sample_rate
            or actual_mic_rate != self.config.mic_sample_rate
        ):
            logger.warning(
                "Using runtime sample rates different from config: "
                f"mic {self.config.mic_sample_rate}->{actual_mic_rate}, "
                f"output {self.config.output_sample_rate}->{actual_output_rate}"
            )

        self._apply_runtime_sample_rates(actual_mic_rate, actual_output_rate)
        self.input_resampler.reset()
        self.output_resampler.reset()
        self._sola_state.buffer = None
        self._input_buf = np.array([], dtype=np.float32)
        self._overlap_buf = None
        self._reset_boundary_continuity_state()
        self._chunks_ready = 0
        self._output_started = False
        self._clear_queues()
        self._feedback_warning_shown = False

        self._running = True
        self._thread = threading.Thread(
            target=self._inference_thread,
            daemon=True,
            name="RCWX-Inference-Unified",
        )
        self._thread.start()

        logger.info(
            f"Unified pipeline started: chunk={self.config.chunk_sec}s, "
            f"overlap={self.config.overlap_sec}s, "
            f"hop_16k={self._hop_samples_16k}, overlap_16k={self._overlap_samples_16k}, "
            f"mic_sr={self._runtime_mic_sample_rate}, out_sr={self._runtime_output_sample_rate}, "
            f"hop_in={self._hop_samples_mic}, hop_out={self._hop_samples_out}"
        )

    def stop(self) -> None:
        if (
            not self._running
            and self._thread is None
            and self._input_stream is None
            and self._output_stream is None
        ):
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
        self._drain_target = self._compute_drain_target()

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
            self._runtime_output_sample_rate * self.config.crossfade_sec
        )
        search_samples_out = int(
            self._runtime_output_sample_rate * self.config.sola_search_ms / 1000
        )
        self._sola_state = SolaState(
            crossfade_samples=crossfade_samples_out,
            search_samples=search_samples_out,
        )

    def set_sola(self, enabled: bool) -> None:
        self.config.use_sola = enabled

    def set_pre_hubert_pitch_ratio(self, ratio: float) -> None:
        self.config.pre_hubert_pitch_ratio = max(0.0, min(1.0, ratio))

    def set_moe_boost(self, strength: float) -> None:
        self.config.moe_boost = max(0.0, min(1.0, float(strength)))

    def set_noise_scale(self, scale: float) -> None:
        self.config.noise_scale = max(0.0, min(1.0, float(scale)))

    def set_f0_lowpass_cutoff_hz(self, cutoff: float) -> None:
        self.config.f0_lowpass_cutoff_hz = max(4.0, min(30.0, float(cutoff)))

    def set_enable_octave_flip_suppress(self, enabled: bool) -> None:
        self.config.enable_octave_flip_suppress = bool(enabled)

    def set_enable_f0_slew_limit(self, enabled: bool) -> None:
        self.config.enable_f0_slew_limit = bool(enabled)

    def set_f0_slew_max_step_st(self, step_st: float) -> None:
        self.config.f0_slew_max_step_st = max(1.0, min(6.0, float(step_st)))

    # ======== Audio Callbacks ========

    def _on_audio_input(self, audio: np.ndarray) -> None:
        """Accumulate mic audio and queue hop-sized chunks."""
        if not self._running:
            return

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
        if not self._running:
            return np.zeros(frames, dtype=np.float32)

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

        # --- Adaptive drift control ---
        # Instead of skipping samples (which causes clicks), read extra
        # samples from the ring and linearly interpolate back to `frames`.
        # This speeds up playback by ≤1.5% — inaudible — while smoothly
        # converging the buffer level toward the natural steady-state peak.
        drain_target = self._compute_drain_target_for_frames(frames)
        threshold = drain_target + frames
        avail = self.output_buffer.available
        extra = 0
        if avail > threshold:
            excess = avail - threshold
            # Cap extra at ~1.5% of frames to keep pitch shift inaudible
            max_extra = max(1, frames // 64)
            extra = min(max_extra, excess)

        if extra > 0 and avail >= frames + extra:
            total = frames + extra
            raw = self.output_buffer.get(total)
            # Linear interpolation: compress (frames+extra) → frames.
            # Cache the base x_new array for `frames` (constant across calls)
            # and build x_src from it.  Only np.interp itself runs per-call.
            if self._interp_cache_frames != frames:
                self._interp_cache_frames = frames
                self._interp_cache_x_base = np.arange(frames, dtype=np.float32)
            x_src = self._interp_cache_x_base * (total - 1) / (frames - 1)
            x_raw = np.arange(total, dtype=np.float32)
            output = np.interp(x_src, x_raw, raw).astype(np.float32)
            self.stats.buffer_trims += 1
            if self.stats.buffer_trims <= 3 or self.stats.buffer_trims % 20 == 0:
                logger.info(
                    "[DRIFT] Time-compressed %d samples (%.1fms), avail=%d target=%d (trim #%d)",
                    extra,
                    extra * 1000.0 / self._runtime_output_sample_rate,
                    avail,
                    drain_target,
                    self.stats.buffer_trims,
                )
        else:
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
                # Keep latency bounded: if backlog exists, drop stale hops and
                # process only the newest chunk.
                stale_dropped = 0
                while True:
                    try:
                        hop_mic = self._input_queue.get_nowait()
                        stale_dropped += 1
                    except Empty:
                        break
                if stale_dropped > 0:
                    self._overlap_buf = None
                    self.pipeline.clear_cache()
                    self.input_resampler.reset()
                    self.output_resampler.reset()
                    self._sola_state.buffer = None
                    self._reset_boundary_continuity_state()
                    logger.warning(
                        "[CATCHUP] Dropped %d stale input chunk(s) to bound latency",
                        stale_dropped,
                    )
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Inference error: {e}", exc_info=True)
                self._reset_boundary_continuity_state()
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

                # --- Stage 2: Resample mic_sr -> 16k ---
                hop_16k = self.input_resampler.resample_chunk(hop_mic)

                # --- Stage 3: Optional denoise ---
                if not self._is_overloaded() and self.config.denoise_enabled:
                    hop_16k = denoise_audio(
                        hop_16k,
                        sample_rate=16000,
                        method=self.config.denoise_method,
                        device=self.pipeline.device,
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
                preprocess_ms = (time.perf_counter() - start_time) * 1000
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
                    noise_scale=self.config.noise_scale,
                    sola_extra_samples=self._sola_extra_model,
                    pre_hubert_pitch_ratio=self.config.pre_hubert_pitch_ratio,
                    moe_boost=self.config.moe_boost,
                    f0_lowpass_cutoff_hz=self.config.f0_lowpass_cutoff_hz,
                    enable_octave_flip_suppress=self.config.enable_octave_flip_suppress,
                    enable_f0_slew_limit=self.config.enable_f0_slew_limit,
                    f0_slew_max_step_st=self.config.f0_slew_max_step_st,
                    hubert_context_sec=self.config.hubert_context_sec,
                )

                # --- Stage 6: Resample model_sr -> 48kHz ---
                output_48k = self.output_resampler.resample_chunk(output_model)

                # --- Stage 7: SOLA crossfade ---
                # target_len = hop_out: forces output to exactly output-hop
                # samples and places the hold-back contiguously after
                # the output boundary, preventing latency drift.
                if self.config.use_sola:
                    output_48k = sola_crossfade(
                        output_48k, self._sola_state,
                        target_len=self._hop_samples_out,
                    )

                # --- Stage 7.5: Boundary gain continuity ---
                # Keep this after SOLA so splice alignment is unaffected.
                output_48k = self._apply_output_boundary_gain(output_48k)

                # Soft clip final output
                max_val = np.max(np.abs(output_48k)) if len(output_48k) else 0.0
                if max_val > 1.0:
                    output_48k = np.tanh(output_48k)

                # --- Stage 8: Feedback detection ---
                if not self.config.wav_input_path:
                    self._store_output_history(output_48k)
                    self._maybe_check_feedback(chunk_at_mic_rate)

                # --- Stage 9: Queue output ---
                inference_ms = (time.perf_counter() - start_time) * 1000
                self.stats.inference_ms = inference_ms
                self.stats.frames_processed += 1
                # Estimated E2E latency (display):
                # - input capture delay: average sample in a chunk ~= chunk/2
                # - processing delay: measured inference thread time
                # - playback delay: ring buffer + pending output-queue chunks
                # - SOLA hold-back delay
                capture_ms = self.config.chunk_sec * 500.0
                buffer_ms = (
                    self.output_buffer.available
                    / self._runtime_output_sample_rate * 1000
                )
                try:
                    queued_chunks = self._output_queue.qsize()
                except Exception:
                    queued_chunks = 0
                queue_ms = (
                    queued_chunks * self._hop_samples_out
                    / self._runtime_output_sample_rate * 1000
                )
                sola_ms = self.config.crossfade_sec * 1000 if self.config.use_sola else 0
                self.stats.latency_ms = (
                    capture_ms
                    + inference_ms
                    + buffer_ms
                    + queue_ms
                    + sola_ms
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
                        f"[PERF] Inference slow: {inference_ms:.0f}ms "
                        f"(pre={preprocess_ms:.0f}ms) > {chunk_ms:.0f}ms chunk"
                    )

                if self.stats.frames_processed <= 5:
                    logger.info(
                        f"[INFER] Chunk #{self.stats.frames_processed}: "
                        f"hop_16k={len(hop_16k)}, overlap={overlap_samples}, "
                        f"out_model={len(output_model)}, out_48k={len(output_48k)}, "
                        f"pre={preprocess_ms:.0f}ms, infer={inference_ms:.0f}ms"
                    )

            except Exception as e:
                logger.error(f"Inference error: {e}", exc_info=True)
                self._reset_boundary_continuity_state()
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
        if self._runtime_output_sample_rate != self._runtime_mic_sample_rate:
            output = resample(
                output,
                self._runtime_output_sample_rate,
                self._runtime_mic_sample_rate,
            )
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
        check_len = min(len(input_audio), self._runtime_mic_sample_rate // 2)
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
                logger.warning("Overload: temporarily bypassing denoise for stability")

    def _is_overloaded(self) -> bool:
        if self._overload_until <= 0:
            return False
        now = time.time()
        if now <= self._overload_until:
            return True
        if self._overload_active:
            self._overload_active = False
            logger.info("Overload cleared: restoring denoise path")
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
                    moe_boost=self.config.moe_boost,
                    enable_octave_flip_suppress=self.config.enable_octave_flip_suppress,
                    enable_f0_slew_limit=self.config.enable_f0_slew_limit,
                    f0_slew_max_step_st=self.config.f0_slew_max_step_st,
                )
                logger.info(f"[WARMUP] Warmup {i + 1}/2 complete")
            self.pipeline.clear_cache()
        except Exception as e:
            logger.warning(f"[WARMUP] Failed (non-fatal): {e}")

    def _run_denoise_warmup(self) -> None:
        """Pre-load denoiser before streams start to avoid first-chunk stalls."""
        if not self.config.denoise_enabled:
            return
        try:
            warmup_len = int(max(self.config.chunk_sec, 0.2) * 16000)
            dummy = np.random.randn(warmup_len).astype(np.float32) * 0.001
            _ = denoise_audio(
                dummy,
                sample_rate=16000,
                method=self.config.denoise_method,
                device=self.pipeline.device,
            )
            logger.info("[WARMUP] Denoiser warmup complete")
        except Exception as e:
            logger.warning(f"[WARMUP] Denoiser warmup failed (non-fatal): {e}")

    def _run_runtime_warmup(self) -> None:
        """Warm up streaming inference path with runtime chunk shape.

        This pre-builds shape-dependent kernels before streams start,
        avoiding first-chunk stalls that can overflow input queues.
        """
        try:
            hop_16k = self._hop_samples_16k
            overlap_16k = min(self._overlap_samples_16k, hop_16k)
            warmup_hop = np.random.randn(hop_16k).astype(np.float32) * 0.001
            if overlap_16k > 0:
                reflection = warmup_hop[:overlap_16k][::-1].copy()
                chunk_16k = np.concatenate([reflection, warmup_hop])
            else:
                chunk_16k = warmup_hop

            f0_method = self.config.f0_method if self.config.use_f0 else "none"
            output_model = self.pipeline.infer_streaming(
                audio_16k=chunk_16k,
                overlap_samples=overlap_16k,
                pitch_shift=self.config.pitch_shift,
                f0_method=f0_method,
                index_rate=0.0,
                index_k=self.config.index_k,
                voice_gate_mode="off",
                energy_threshold=self.config.energy_threshold,
                use_parallel_extraction=self.config.use_parallel_extraction,
                noise_scale=self.config.noise_scale,
                sola_extra_samples=self._sola_extra_model,
                pre_hubert_pitch_ratio=self.config.pre_hubert_pitch_ratio,
                moe_boost=self.config.moe_boost,
                f0_lowpass_cutoff_hz=self.config.f0_lowpass_cutoff_hz,
                enable_octave_flip_suppress=self.config.enable_octave_flip_suppress,
                enable_f0_slew_limit=self.config.enable_f0_slew_limit,
                f0_slew_max_step_st=self.config.f0_slew_max_step_st,
                hubert_context_sec=self.config.hubert_context_sec,
            )
            _ = self.output_resampler.resample_chunk(output_model)

            # Reset warmup side-effects so first real chunk starts cleanly.
            self.pipeline.clear_cache()
            self.input_resampler.reset()
            self.output_resampler.reset()
            self._sola_state.buffer = None
            self._overlap_buf = None
            self._reset_boundary_continuity_state()
            logger.info("[WARMUP] Streaming path warmup complete")
        except Exception as e:
            self._reset_boundary_continuity_state()
            logger.warning(f"[WARMUP] Streaming path warmup failed (non-fatal): {e}")

    # ======== Helpers ========

    def _reset_boundary_continuity_state(self) -> None:
        """Reset chunk-boundary gain continuity state."""
        self._prev_tail_rms = 0.0

    def _apply_output_boundary_gain(self, output: np.ndarray) -> np.ndarray:
        """Apply a mild post-SOLA gain ramp to smooth boundary loudness.

        The adjustment is intentionally conservative:
        - short analysis window (5ms),
        - tight gain clamp (0.9x..1.1x),
        - short ramp (10ms),
        - disabled for quiet heads (e.g., breaths).
        """
        if len(output) == 0:
            return output

        boundary = min(
            len(output),
            max(1, int(self._runtime_output_sample_rate * 0.005)),
        )
        head_rms = float(np.sqrt(np.mean(output[:boundary] ** 2) + 1e-10))
        min_rms = 2e-3

        if self._prev_tail_rms >= min_rms and head_rms >= min_rms:
            ratio = float(np.clip(self._prev_tail_rms / head_rms, 0.9, 1.1))
            if abs(ratio - 1.0) >= 0.02:
                ramp_len = min(
                    len(output),
                    max(boundary, int(self._runtime_output_sample_rate * 0.010)),
                )
                ramp = np.linspace(ratio, 1.0, ramp_len, dtype=np.float32)
                adjusted = output.copy()
                adjusted[:ramp_len] *= ramp
                output = adjusted

        self._prev_tail_rms = float(
            np.sqrt(np.mean(output[-boundary:] ** 2) + 1e-10)
        )
        return output

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
