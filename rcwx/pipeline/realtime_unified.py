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
from scipy.signal import resample_poly

from rcwx.audio.buffer import RingOutputBuffer
from rcwx.audio.denoise import denoise as denoise_audio
from rcwx.audio.duplex import AsioDuplexStream
from rcwx.audio.input import AudioInput
from rcwx.audio.output import AudioOutput
from rcwx.audio.postprocess import Postprocessor, PostprocessConfig
from rcwx.audio.resample import StatefulResampler, resample
from rcwx.audio.sola import SolaState, sola_crossfade, sola_flush
from rcwx.audio.stream_base import is_device_on_asio, parse_output_channel_pair
from rcwx.pipeline.inference import RVCPipeline, StreamingParams

logger = logging.getLogger(__name__)

# GPU/driver-level failure signatures (Intel level_zero / CUDA).  Once the
# device context is lost, every following chunk fails the same way — these
# are unrecoverable without restarting the process.
_DEVICE_ERROR_TOKENS = (
    "UR_RESULT_ERROR_DEVICE_LOST",
    "UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY",
    "UR_RESULT_ERROR_OUT_OF_RESOURCES",
    "level_zero backend failed",
    "CUDA error",
)
_DEVICE_ERROR_STREAK_LIMIT = 3


def _is_device_error(exc: BaseException) -> bool:
    msg = str(exc)
    return any(token in msg for token in _DEVICE_ERROR_TOKENS)


def _max_normalized_lag_correlation(
    needle: np.ndarray, haystack: np.ndarray
) -> float:
    """Max normalized cross-correlation of ``needle`` over all lags of
    ``haystack`` (both 1D).  Returns a value in [0, 1].

    Used for feedback detection: when the played output re-enters the input,
    the input chunk is a (delayed, near-identical) copy of some window of the
    output history, giving a correlation near 1 at the corresponding lag.
    Per-window normalization uses cumulative sums (O(N), same technique as
    SOLA's offset search).
    """
    n = len(needle)
    if n < 8 or len(haystack) <= n:
        return 0.0

    needle = needle - float(np.mean(needle))
    needle_norm = float(np.sqrt(np.sum(needle**2)))
    if needle_norm < 1e-8:
        return 0.0

    # Since sum(needle)==0, dot(needle, w - mean(w)) == dot(needle, w)
    dots = np.correlate(haystack, needle, mode="valid")  # len(hay) - n + 1

    x = haystack.astype(np.float64)
    cumsum = np.empty(len(x) + 1, dtype=np.float64)
    cumsum[0] = 0.0
    np.cumsum(x, out=cumsum[1:])
    cumsum_sq = np.empty(len(x) + 1, dtype=np.float64)
    cumsum_sq[0] = 0.0
    np.cumsum(x * x, out=cumsum_sq[1:])

    window_sums = cumsum[n:] - cumsum[:-n]
    window_sq_sums = cumsum_sq[n:] - cumsum_sq[:-n]
    norms_sq = np.maximum(window_sq_sums - window_sums * window_sums / n, 0.0)
    norms = np.sqrt(norms_sq)

    valid = norms > 1e-8
    if not np.any(valid):
        return 0.0
    corrs = np.abs(dots[valid]) / (needle_norm * norms[valid])
    return float(np.max(corrs))


# ---------------------------------------------------------------------------
# RealtimeConfig / RealtimeStats
# ---------------------------------------------------------------------------


@dataclass
class RealtimeStats:
    """Statistics for real-time processing."""

    latency_ms: float = 0.0
    inference_ms: float = 0.0
    # Per-stage breakdown of inference_ms (from RVCPipeline.stage_times)
    hubert_ms: float = 0.0
    f0_ms: float = 0.0
    faiss_ms: float = 0.0
    synth_ms: float = 0.0
    buffer_underruns: int = 0
    buffer_overruns: int = 0
    buffer_trims: int = 0
    frames_processed: int = 0
    feedback_detected: bool = False
    feedback_correlation: float = 0.0
    gpu_memory_percent: float = 0.0
    input_rms_db: float = -60.0
    input_peak_db: float = -60.0
    output_rms_db: float = -60.0
    output_peak_db: float = -60.0

    def reset(self) -> None:
        self.latency_ms = 0.0
        self.inference_ms = 0.0
        self.hubert_ms = 0.0
        self.f0_ms = 0.0
        self.faiss_ms = 0.0
        self.synth_ms = 0.0
        self.buffer_underruns = 0
        self.buffer_overruns = 0
        self.buffer_trims = 0
        self.frames_processed = 0
        self.feedback_detected = False
        self.feedback_correlation = 0.0
        self.gpu_memory_percent = 0.0
        self.input_rms_db = -60.0
        self.input_peak_db = -60.0
        self.output_rms_db = -60.0
        self.output_peak_db = -60.0


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
    output_channel_selection: str = "auto"  # "auto", "0,1", "2,3", ...

    # Core parameters
    chunk_sec: float = 0.15
    overlap_sec: float = 0.10  # audio-level overlap for HuBERT continuity
    crossfade_sec: float = 0.05
    # SOLA search window (ms): one period of the lowest expected output F0
    # (70Hz -> 14.3ms) + margin, so the splice can always phase-align.
    sola_search_ms: float = 15.0
    prebuffer_chunks: int = 1
    buffer_margin: float = 0.5

    # Pitch / F0
    pitch_shift: int = 0
    use_f0: bool = True
    f0_method: str = "fcpe"

    # Processing
    max_queue_size: int = 8
    input_gain_db: float = 0.0
    output_gain_db: float = 0.0  # Output level adjustment (post-processing)
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

    # F0 extraction window: leading context (sec) before the new hop.
    # <= 0 extracts F0 on the full HuBERT context (legacy behavior).
    f0_context_sec: float = 0.32

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
    fixed_harmonics: bool = True
    f0_lowpass_cutoff_hz: float = 16.0
    enable_octave_flip_suppress: bool = True
    enable_f0_slew_limit: bool = True
    f0_slew_max_step_st: float = 2.8
    # Longest unvoiced hole (ms) inside a voiced run to fill (<= 0 disables)
    f0_hole_fill_ms: float = 30.0
    # Voiced/unvoiced excitation crossfade ramp in ms (0 = hard switch)
    uv_ramp_ms: float = 5.0

    # ASIO buffer size in frames (0 = follow the driver control panel)
    asio_buffer_size: int = 0

    # Decoder overlap for cross-chunk continuity (feature frames, 1 frame = 10ms)
    decoder_overlap_frames: int = 5

    # Post-processing (treble boost + limiter)
    postprocess_enabled: bool = True
    treble_boost_db: float = 4.0
    treble_cutoff_hz: float = 2800.0
    limiter_threshold_db: float = -1.0
    limiter_release_ms: float = 80.0

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
        self._hop_samples_16k = self._align_to_hop(int(self.config.chunk_sec * 16000), hubert_hop)
        self._overlap_samples_16k = self._align_to_hop(
            int(self.config.overlap_sec * 16000), hubert_hop
        )

        # Corresponding counts at runtime stream rates
        self._hop_samples_mic = int(self._hop_samples_16k * self._runtime_mic_sample_rate / 16000)
        self._hop_samples_out = int(
            self._hop_samples_16k * self._runtime_output_sample_rate / 16000
        )

        # Resamplers
        self.input_resampler = StatefulResampler(self._runtime_mic_sample_rate, 16000)
        self.output_resampler = StatefulResampler(
            self.pipeline.sample_rate, self._runtime_output_sample_rate
        )

        # Postprocessor (treble boost + limiter)
        pp_cfg = PostprocessConfig(
            enabled=getattr(self.config, "postprocess_enabled", True),
            treble_boost_db=getattr(self.config, "treble_boost_db", 4.0),
            treble_cutoff_hz=getattr(self.config, "treble_cutoff_hz", 2800.0),
            limiter_threshold_db=getattr(self.config, "limiter_threshold_db", -1.0),
            limiter_release_ms=getattr(self.config, "limiter_release_ms", 80.0),
        )
        self._postprocessor = Postprocessor(self._runtime_output_sample_rate, pp_cfg)

        # SOLA state
        crossfade_samples_out = int(self._runtime_output_sample_rate * self.config.crossfade_sec)
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
            sola_extra_out * self.pipeline.sample_rate / self._runtime_output_sample_rate
        )
        # Decoder overlap: extra context frames so decoder Conv/SineGen
        # have warm start, preventing cold-start discontinuity at chunk edges.
        decoder_overlap_model = self.config.decoder_overlap_frames * zc_model
        self._sola_extra_model = (
            (sola_extra_raw + decoder_overlap_model + zc_model - 1)
            // zc_model * zc_model
        )

        # Output buffer: 4x chunk capacity (physical ring size)
        chunk_output_samples = int(self._runtime_output_sample_rate * self.config.chunk_sec)
        self.output_buffer = RingOutputBuffer(
            capacity_samples=chunk_output_samples * 4,
            fade_samples=256,
        )

        # Adaptive drift control state (floor-based, skip-only).
        #
        # The ring conserves ~prebuffer_chunks hops of standing latency
        # (output tracks the mic rate, so the prebuffer fill is never
        # consumed), so the windowed MINIMUM post-read level (the floor)
        # sits at ~prebuffer_chunks*hop in steady state and must NOT be
        # shed.  Every callback records its post-read level into a sliding
        # window (~2 chunk periods); only when genuine accumulation
        # (inference stall, startup transient, clock drift on split
        # devices) lifts the floor a full chunk above the standing level do
        # we hard-skip the excess.  No per-callback time-compression: with
        # small device blocks (e.g. 64-frame ASIO) an np.interp every
        # callback restarts the resampling phase each block and buzzes.
        # See _on_audio_output / _compute_shed_threshold.
        self._level_window: Optional[deque] = None
        self._level_window_frames = 0

        # Input accumulator (ring buffer at mic rate)
        self._input_buf = np.array([], dtype=np.float32)

        # Overlap buffer at 16kHz (stores tail of previous chunk for prepending)
        self._overlap_buf: Optional[np.ndarray] = None

        # Threading
        self._running = False
        self._thread: Optional[threading.Thread] = None
        # Consecutive GPU/driver-level failures (device lost, VRAM exhausted).
        # After a device loss every subsequent chunk fails the same way, so
        # the inference thread stops itself instead of error-spamming.
        self._device_error_streak = 0
        self._input_queue: Queue = Queue(maxsize=self.config.max_queue_size)
        self._output_queue: Queue = Queue(maxsize=self.config.max_queue_size)

        # Pre-buffering
        self._prebuffer_chunks = self.config.prebuffer_chunks
        self._chunks_ready = 0
        self._output_started = False

        # Audio streams (AudioInput, WavFileInput, or AsioDuplexStream)
        self._input_stream = None
        self._output_stream = None

        # Feedback detection
        self._output_history_size = self._runtime_mic_sample_rate  # 1 second
        self._output_history = np.zeros(self._output_history_size, dtype=np.float32)
        self._output_history_pos = 0
        self._feedback_check_interval = 10
        self._feedback_warning_shown = False

        # Boundary gain continuity across emitted chunks (post-SOLA).
        self._prev_tail_rms: float = 0.0

        # GPU total memory cache (bytes); fetched once on first use.
        self._gpu_total_memory: int = 0

        # Overload protection
        self._queue_full_times: deque[float] = deque(maxlen=50)
        self._overload_until: float = 0.0
        self._overload_active = False

        # True when prepare() already ran the warmups for the current
        # config; start() consumes this to skip its synchronous warmup.
        self._prepared = False

    @staticmethod
    def _align_to_hop(samples: int, hop: int) -> int:
        """Round sample count up to nearest multiple of hop."""
        return ((samples + hop - 1) // hop) * hop

    def _compute_shed_threshold(self) -> tuple[int, int]:
        """Return (shed_threshold, shed_target) for the post-read floor.

        The ring conserves roughly ``prebuffer_chunks`` hops of *standing*
        latency: output never runs ahead of input (it tracks the mic rate),
        so the initial prebuffer fill is never consumed — the windowed floor
        sits at ~``prebuffer_chunks * hop`` and wobbles with the
        burst/callback phase beat.  This standing level is NOT drift; it is
        the latency the prebuffer setting deliberately holds, so shedding it
        would just defeat the prebuffer (and, with small device blocks,
        click or buzz on every callback).

        We only shed once the floor exceeds the natural standing level by a
        full chunk plus a margin-scaled tolerance — anything below that is
        harmless wobble / deliberate prebuffer.

        Returns:
            (shed_threshold, shed_target): skip down to ``shed_target``
            (≈ standing + half a hop) whenever ``floor > shed_threshold``.
        """
        margin = max(0.1, min(2.0, float(self.config.buffer_margin)))
        standing = self._prebuffer_chunks * self._hop_samples_out
        # Tolerance: a full chunk of wobble headroom plus margin scaling
        # (margin 0.5 → ~1.5 chunks above standing; tighter/looser below/above).
        tol = int((1.0 + margin) * self._hop_samples_out)
        shed_threshold = standing + tol
        # Land back near the natural floor + a small cushion so post-skip
        # wobble doesn't immediately re-trigger a skip.
        shed_target = standing + self._hop_samples_out // 2
        return shed_threshold, shed_target

    def _recalculate_sizes(self) -> None:
        """Recalculate derived sample counts from current config."""
        hubert_hop = 320
        self._hop_samples_16k = self._align_to_hop(int(self.config.chunk_sec * 16000), hubert_hop)
        self._overlap_samples_16k = self._align_to_hop(
            int(self.config.overlap_sec * 16000), hubert_hop
        )
        self._hop_samples_mic = int(self._hop_samples_16k * self._runtime_mic_sample_rate / 16000)
        self._hop_samples_out = int(
            self._hop_samples_16k * self._runtime_output_sample_rate / 16000
        )

        # SOLA extra samples
        crossfade_samples_out = int(self._runtime_output_sample_rate * self.config.crossfade_sec)
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
            sola_extra_out * self.pipeline.sample_rate / self._runtime_output_sample_rate
        )
        decoder_overlap_model = self.config.decoder_overlap_frames * zc_model
        self._sola_extra_model = (
            (sola_extra_raw + decoder_overlap_model + zc_model - 1)
            // zc_model * zc_model
        )

        # Hop change invalidates the drift-control level window
        self._level_window = None

    def _apply_runtime_sample_rates(self, mic_rate: int, output_rate: int) -> None:
        """Apply actual stream sample rates and rebuild dependent state."""
        self._runtime_mic_sample_rate = int(mic_rate)
        self._runtime_output_sample_rate = int(output_rate)
        self._recalculate_sizes()

        self.input_resampler = StatefulResampler(self._runtime_mic_sample_rate, 16000)
        self.output_resampler = StatefulResampler(
            self.pipeline.sample_rate, self._runtime_output_sample_rate
        )

        chunk_output_samples = max(1, int(self._runtime_output_sample_rate * self.config.chunk_sec))
        self.output_buffer = RingOutputBuffer(
            capacity_samples=chunk_output_samples * 4,
            fade_samples=256,
        )

        self._output_history_size = max(1, self._runtime_mic_sample_rate)
        self._output_history = np.zeros(self._output_history_size, dtype=np.float32)
        self._output_history_pos = 0

    # ======== Lifecycle ========

    def prepare(self) -> None:
        """Run all warmups ahead of start().

        Contains no PortAudio or UI calls, so it can run on a background
        thread — unlike stream creation in start(), which must stay on the
        main thread on Windows.  After prepare(), start() skips the
        warmups and only opens the audio streams (milliseconds instead of
        seconds), keeping the GUI responsive during the expensive XPU
        kernel compilation.
        """
        self._apply_runtime_sample_rates(
            self.config.mic_sample_rate,
            self.config.output_sample_rate,
        )
        if self._on_warmup_progress:
            self._on_warmup_progress(0, 2, "デノイザ準備中...")
        self._run_denoise_warmup()
        if self._on_warmup_progress:
            self._on_warmup_progress(1, 2, "ストリーミング準備中...")
        self._run_runtime_warmup()
        if self._on_warmup_progress:
            self._on_warmup_progress(2, 2, "準備完了")
        self._prepared = True

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
        if not self._prepared:
            self._run_denoise_warmup()
            self._run_runtime_warmup()
        # Consumed: restarts (e.g. after a chunk-size change) must re-warm
        # because shape-dependent kernels change with the config.
        self._prepared = False

        # Audio stream block size: chunk/4 for responsive I/O
        output_chunk_sec = self.config.chunk_sec / 4
        output_blocksize = int(self.config.output_sample_rate * output_chunk_sec)

        # --- ASIO duplex detection ---
        # ASIO drivers are exclusive full-duplex: opening separate streams
        # fails with "Device unavailable".  Use a single sd.Stream instead.
        use_asio_duplex = (
            not self.config.wav_input_path
            and is_device_on_asio(self.config.input_device, "input")
            and is_device_on_asio(self.config.output_device, "output")
        )

        if use_asio_duplex:
            logger.info("Both devices are ASIO — using duplex stream")
            in_dev = self.config.input_device
            out_dev = self.config.output_device
            # Resolve None → default device index for sd.Stream(device=(...))
            if in_dev is None:
                from rcwx.audio.stream_base import get_default_device
                in_dev = get_default_device("input")
            if out_dev is None:
                from rcwx.audio.stream_base import get_default_device
                out_dev = get_default_device("output")
            if in_dev is None or out_dev is None:
                logger.warning(
                    "ASIO duplex: default device not found (in=%s, out=%s), "
                    "falling back to separate streams",
                    in_dev, out_dev,
                )
                use_asio_duplex = False

        # Voice changer output is mono — only a stereo pair is needed.
        # Open all channels ONLY when an explicit, parseable channel pair is
        # selected.  An unparseable selection must NOT open all channels: the
        # duplex callback would then fall back to auto routing across a fully
        # opened multi-channel device (previously this fed ASIO LOOPBACK
        # outputs and closed a feedback loop).
        output_pair = parse_output_channel_pair(self.config.output_channel_selection)
        if use_asio_duplex and output_pair is not None:
            output_channels = self.config.output_channels
        else:
            output_channels = min(2, self.config.output_channels)

        if use_asio_duplex:
            duplex = AsioDuplexStream(
                input_device=in_dev,
                output_device=out_dev,
                sample_rate=self.config.output_sample_rate,
                input_channels=self.config.input_channels,
                output_channels=output_channels,
                blocksize=0,  # let ASIO driver choose preferred buffer size
                input_callback=self._on_audio_input,
                output_callback=self._on_audio_output,
                channel_selection=self.config.input_channel_selection,
                output_channel_selection=self.config.output_channel_selection,
                requested_buffer_size=self.config.asio_buffer_size,
            )
            duplex.start()
            # Both references point to the same object so stop() only closes once.
            self._input_stream = duplex
            self._output_stream = duplex
        else:
            # --- Separate streams (WASAPI / DirectSound / MME) ---
            # Start output stream FIRST so its callback is active before any
            # audio arrives.  AudioOutput.start() may take time due to API
            # fallback (WASAPI → DirectSound → MME), and during that time the
            # input stream would queue audio with no output drain — causing a
            # burst of buffered audio and large initial trims.
            self._output_stream = AudioOutput(
                device=self.config.output_device,
                sample_rate=self.config.output_sample_rate,
                channels=output_channels,
                blocksize=output_blocksize,
                callback=self._on_audio_output,
                output_channel_selection=self.config.output_channel_selection,
                asio_buffer_size=self.config.asio_buffer_size,
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
                    asio_buffer_size=self.config.asio_buffer_size,
                )
            self._input_stream.start()

        actual_output_rate = int(round(self._output_stream.actual_sample_rate))
        # For ASIO duplex, input and output share a single stream at one rate.
        if use_asio_duplex:
            actual_mic_rate = actual_output_rate
        else:
            actual_mic_rate = int(round(self._input_stream.actual_sample_rate))
        if (
            actual_output_rate != self.config.output_sample_rate
            or actual_mic_rate != self.config.mic_sample_rate
        ):
            reason = " (ASIO native)" if use_asio_duplex else ""
            logger.warning(
                "Using runtime sample rates different from config%s: "
                "mic %d->%d, output %d->%d",
                reason,
                self.config.mic_sample_rate, actual_mic_rate,
                self.config.output_sample_rate, actual_output_rate,
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

        # When using ASIO duplex, both references point to the same object.
        # Stop only once to avoid double-close.
        if self._input_stream is self._output_stream:
            if self._input_stream is not None:
                self._input_stream.stop()
                self._input_stream = None
                self._output_stream = None
        else:
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

    def set_input_gain_db(self, gain_db: float) -> None:
        self.config.input_gain_db = float(gain_db)

    def set_output_gain_db(self, gain_db: float) -> None:
        self.config.output_gain_db = float(gain_db)

    def set_denoise(self, enabled: bool, method: str = "auto") -> None:
        self.config.denoise_enabled = enabled
        self.config.denoise_method = method

    def set_voice_gate_mode(self, mode: str) -> None:
        self.config.voice_gate_mode = mode

    def set_energy_threshold(self, value: float) -> None:
        self.config.energy_threshold = value

    def set_postprocess_config(self, cfg) -> None:
        from rcwx.audio.postprocess import PostprocessConfig

        if hasattr(self, "_postprocessor"):
            self._postprocessor.config.enabled = cfg.enabled
            self._postprocessor.config.treble_boost_db = cfg.treble_boost_db
            self._postprocessor.config.treble_cutoff_hz = cfg.treble_cutoff_hz
            self._postprocessor.config.limiter_threshold_db = cfg.limiter_threshold_db
            self._postprocessor.config.limiter_release_ms = cfg.limiter_release_ms
            self._postprocessor._treble._design_filter()
            self._postprocessor._limiter._threshold = 10 ** (cfg.limiter_threshold_db / 20)
            self._postprocessor._limiter._release_coeff = np.exp(
                -1.0 / (cfg.limiter_release_ms * self._postprocessor._limiter.sample_rate / 1000)
            )

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
        self._overlap_samples_16k = self._align_to_hop(int(self.config.overlap_sec * 16000), 320)

    def set_crossfade(self, crossfade_sec: float) -> None:
        self.config.crossfade_sec = max(0.0, crossfade_sec)
        # Rebuild SOLA state with new crossfade length
        crossfade_samples_out = int(self._runtime_output_sample_rate * self.config.crossfade_sec)
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

    def set_fixed_harmonics(self, enabled: bool) -> None:
        self.config.fixed_harmonics = bool(enabled)

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
            hop = self._input_buf[: self._hop_samples_mic].copy()
            self._input_buf = self._input_buf[self._hop_samples_mic :]
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

        # --- Adaptive drift control (floor-based, skip-only) ---
        # Latency state is judged on the MINIMUM post-read ring level over
        # the last ~2 chunk periods (the floor), never on the instantaneous
        # level — the burst/drain oscillation and device blocksize phase
        # beats don't lift the floor, so they can't trigger false skips.
        #
        # The ring conserves ~prebuffer_chunks hops of standing latency
        # (output tracks the mic rate, so the prebuffer fill is never
        # consumed), so the natural floor sits at ~prebuffer_chunks*hop and
        # must NOT be shed.  We only skip once genuine accumulation pushes
        # the floor a full chunk-plus-margin above the standing level.
        #
        # Shedding is done exclusively via hard-skip (one splice + fade-in),
        # NEVER per-callback time-compression: with small device blocks
        # (e.g. 64-frame ASIO) an np.interp every callback restarts the
        # resampling phase each block, producing a audible tone at
        # sample_rate/blocksize.  A periodic clean skip is inaudible by
        # comparison.
        if self._level_window is None or self._level_window_frames != frames:
            self._level_window_frames = frames
            window_len = max(4, -(-2 * self._hop_samples_out // max(1, frames)))
            self._level_window = deque(maxlen=window_len)

        if len(self._level_window) == self._level_window.maxlen:
            floor = min(self._level_window)
            shed_threshold, shed_target = self._compute_shed_threshold()
            if floor > shed_threshold:
                # Drop the accumulated excess in one splice.  Land below the
                # threshold (at shed_target) so post-skip wobble doesn't
                # immediately re-trigger.  skip() flags the next read for
                # fade-in, so the splice doesn't click.
                skip_amount = max(0, floor - shed_target)
                skipped = self.output_buffer.skip(skip_amount)
                self._level_window.clear()  # re-measure from the new level
                self.stats.buffer_trims += 1
                logger.info(
                    "[DRIFT] Skipped %d samples (%.0fms) of accumulated latency, "
                    "floor=%d threshold=%d target=%d (trim #%d)",
                    skipped,
                    skipped * 1000.0 / self._runtime_output_sample_rate,
                    floor,
                    shed_threshold,
                    shed_target,
                    self.stats.buffer_trims,
                )

        output = self.output_buffer.get(frames)

        # Record the post-read level: the true sawtooth trough occurs right
        # before a burst lands, and only the post-read sample at the
        # preceding callback captures it (pre-read sampling sits one
        # callback higher and hides the real floor).
        self._level_window.append(self.output_buffer.available)

        # Count REAL underruns only: the ring's counter increments when a
        # read could not be fully served (zero-padded output = audible gap).
        # Previously "available == 0 after a full read" was also counted,
        # over-reporting: an exactly-drained ring that refills before the
        # next callback causes no gap at all.
        ring_underruns = self.output_buffer.underrun_count
        if ring_underruns > self.stats.buffer_underruns:
            self.stats.buffer_underruns = ring_underruns
            if ring_underruns <= 3 or ring_underruns % 20 == 0:
                logger.warning(
                    "[UNDERRUN] Output buffer starved (#%d, frames=%d, avail=%d)",
                    ring_underruns,
                    frames,
                    self.output_buffer.available,
                )

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

                # Measure input level (post-gain) for the GUI input meter. This
                # is the only way to monitor input on exclusive ASIO devices,
                # which cannot be opened by a separate standalone InputStream.
                if chunk_at_mic_rate.size:
                    in_rms = float(np.sqrt(np.mean(chunk_at_mic_rate**2)))
                    in_peak = float(np.max(np.abs(chunk_at_mic_rate)))
                    self.stats.input_rms_db = max(-60.0, 20.0 * np.log10(max(in_rms, 1e-6)))
                    self.stats.input_peak_db = max(-60.0, 20.0 * np.log10(max(in_peak, 1e-6)))

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
                        self._overlap_buf = combined[-self._overlap_samples_16k :]
                    else:
                        tail = min(self._overlap_samples_16k, len(hop_16k))
                        self._overlap_buf = hop_16k[-tail:].copy()
                else:
                    self._overlap_buf = None

                # --- Stage 5: Inference ---
                preprocess_ms = (time.perf_counter() - start_time) * 1000

                output_model = self.pipeline.infer_streaming(
                    chunk_16k,
                    overlap_samples,
                    self._build_streaming_params(
                        index_rate=self.config.index_rate,
                        voice_gate_mode=self.config.voice_gate_mode,
                    ),
                )

                # --- Stage 6: Resample model_sr -> 48kHz ---
                output_48k = self.output_resampler.resample_chunk(output_model)

                # --- Stage 7: SOLA crossfade ---
                # target_len = hop_out: forces output to exactly output-hop
                # samples and places the hold-back contiguously after
                # the output boundary, preventing latency drift.
                sola_in_len = len(output_48k)
                if self.config.use_sola:
                    output_48k = sola_crossfade(
                        output_48k,
                        self._sola_state,
                        target_len=self._hop_samples_out,
                    )

                # --- Boundary-continuity diagnostics ---
                # Tracks the per-chunk content advance vs the expected
                # hop.  A non-zero SOLA offset skips that many samples of
                # the new chunk (replaced by the held-back prev tail), and a
                # SOLA output shorter than hop_out is a content deficit —
                # both manifest as "phonemes cut at the boundary" / periodic
                # stutter on sustained tones.  Logged throttled.
                if not hasattr(self, "_bdiag_n"):
                    self._bdiag_n = 0
                    self._bdiag_balance = 0  # emitted - expected (samples)
                self._bdiag_n += 1
                sola_out_len = len(output_48k)
                self._bdiag_balance += sola_out_len - self._hop_samples_out
                if (
                    self._bdiag_n <= 5
                    or self._bdiag_n % 50 == 0
                    or (self.config.use_sola and self._sola_state.last_offset != 0
                        and self._bdiag_n <= 200)
                ):
                    off = self._sola_state.last_offset if self.config.use_sola else 0
                    logger.info(
                        "[BOUNDARY] chunk#%d sola_in=%d out=%d hop=%d offset=%d "
                        "balance=%d (%.1fms)",
                        self._bdiag_n, sola_in_len, sola_out_len,
                        self._hop_samples_out, off,
                        self._bdiag_balance,
                        self._bdiag_balance * 1000.0 / self._runtime_output_sample_rate,
                    )

                # --- Stage 7.5: Boundary gain continuity ---
                # Keep this after SOLA so splice alignment is unaffected.
                output_48k = self._apply_output_boundary_gain(output_48k)

                # --- Stage 7.6: Post-processing (treble boost + limiter) ---
                output_48k = self._postprocessor.process(output_48k)

                # --- Stage 7.7: User output level adjustment ---
                if self.config.output_gain_db != 0.0:
                    output_48k = output_48k * (10.0 ** (self.config.output_gain_db / 20.0))

                # Hard clip final output (safety net after limiter)
                output_48k = np.clip(output_48k, -1.0, 1.0)

                # Measure final output level for the GUI output meter
                if output_48k.size:
                    out_rms = float(np.sqrt(np.mean(output_48k**2)))
                    out_peak = float(np.max(np.abs(output_48k)))
                    self.stats.output_rms_db = max(-60.0, 20.0 * np.log10(max(out_rms, 1e-6)))
                    self.stats.output_peak_db = max(-60.0, 20.0 * np.log10(max(out_peak, 1e-6)))

                # --- Stage 8: Feedback detection ---
                if not self.config.wav_input_path:
                    self._store_output_history(output_48k)
                    self._maybe_check_feedback(chunk_at_mic_rate)

                # --- Stage 9: Queue output ---
                inference_ms = (time.perf_counter() - start_time) * 1000
                self.stats.inference_ms = inference_ms
                stage = getattr(self.pipeline, "stage_times", None) or {}
                self.stats.hubert_ms = float(stage.get("hubert_ms", 0.0))
                self.stats.f0_ms = float(stage.get("f0_ms", 0.0))
                self.stats.faiss_ms = float(stage.get("faiss_ms", 0.0))
                self.stats.synth_ms = float(stage.get("synth_ms", 0.0))
                self.stats.frames_processed += 1

                # --- GPU memory usage ---
                try:
                    import torch

                    device = self.pipeline.device
                    device_str = str(device)
                    if "xpu" in device_str:
                        if self._gpu_total_memory == 0:
                            self._gpu_total_memory = (
                                torch.xpu.get_device_properties(device).total_memory
                            )
                        if self._gpu_total_memory > 0:
                            alloc = torch.xpu.memory_allocated(device)
                            self.stats.gpu_memory_percent = alloc / self._gpu_total_memory * 100
                    elif "cuda" in device_str:
                        if self._gpu_total_memory == 0:
                            self._gpu_total_memory = (
                                torch.cuda.get_device_properties(device).total_memory
                            )
                        if self._gpu_total_memory > 0:
                            alloc = torch.cuda.memory_allocated(device)
                            self.stats.gpu_memory_percent = alloc / self._gpu_total_memory * 100
                except Exception:
                    pass

                # Estimated E2E latency (display):
                # - input capture delay: average sample in a chunk ~= chunk/2
                # - processing delay: measured inference thread time
                # - playback delay: ring buffer + pending output-queue chunks
                # - SOLA hold-back delay
                capture_ms = self.config.chunk_sec * 500.0
                buffer_ms = self.output_buffer.available / self._runtime_output_sample_rate * 1000
                try:
                    queued_chunks = self._output_queue.qsize()
                except Exception:
                    queued_chunks = 0
                queue_ms = (
                    queued_chunks * self._hop_samples_out / self._runtime_output_sample_rate * 1000
                )
                sola_ms = self.config.crossfade_sec * 1000 if self.config.use_sola else 0
                self.stats.latency_ms = capture_ms + inference_ms + buffer_ms + queue_ms + sola_ms

                try:
                    self._output_queue.put_nowait(output_48k)
                except Exception:
                    logger.warning("Output queue full, dropping chunk")
                    self.stats.buffer_overruns += 1

                if self.on_stats_update:
                    self.on_stats_update(self.stats)

                # Performance monitoring
                chunk_ms = self.config.chunk_sec * 1000
                stage_detail = (
                    f"pre={preprocess_ms:.0f} hubert={self.stats.hubert_ms:.0f} "
                    f"f0={self.stats.f0_ms:.0f} faiss={self.stats.faiss_ms:.0f} "
                    f"synth={self.stats.synth_ms:.0f}ms"
                )
                if inference_ms > chunk_ms * 0.8 and self.stats.frames_processed % 50 == 0:
                    logger.warning(
                        f"[PERF] Inference slow: {inference_ms:.0f}ms "
                        f"({stage_detail}) > {chunk_ms:.0f}ms chunk"
                    )

                if self.stats.frames_processed % 100 == 0:
                    logger.info(
                        f"[PERF] Stage breakdown: infer={inference_ms:.0f}ms "
                        f"({stage_detail}) / chunk={chunk_ms:.0f}ms"
                    )

                if self.stats.frames_processed <= 5:
                    logger.info(
                        f"[INFER] Chunk #{self.stats.frames_processed}: "
                        f"hop_16k={len(hop_16k)}, overlap={overlap_samples}, "
                        f"out_model={len(output_model)}, out_48k={len(output_48k)}, "
                        f"infer={inference_ms:.0f}ms ({stage_detail})"
                    )

                self._device_error_streak = 0

            except Exception as e:
                logger.error(f"Inference error: {e}", exc_info=True)
                self._reset_boundary_continuity_state()
                if _is_device_error(e):
                    self._device_error_streak += 1
                    if self._device_error_streak >= _DEVICE_ERROR_STREAK_LIMIT:
                        logger.error(
                            "[FATAL] %d consecutive GPU device errors — "
                            "stopping stream (device context is likely lost; "
                            "restart the app to recover)",
                            self._device_error_streak,
                        )
                        if self.on_error:
                            self.on_error(
                                "GPUデバイスエラーが継続したため停止しました。"
                                "アプリの再起動が必要です"
                            )
                        self._running = False
                        break
                else:
                    self._device_error_streak = 0
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
        """Return the max normalized cross-correlation between the current
        input chunk and the last ~1s of played output, scanned over all lags.

        Feedback returns to the input delayed by one full E2E latency, so the
        correlation must be searched across lags — a zero-lag comparison
        never fires on real feedback.  The output history ring is unrolled
        chronologically before the scan.  Both signals are decimated to keep
        the lag scan cheap (runs every ``_feedback_check_interval`` chunks).
        """
        if len(input_audio) < 1000:
            return 0.0
        input_rms = np.sqrt(np.mean(input_audio**2))
        if input_rms < 0.01:
            return 0.0
        output_rms = np.sqrt(np.mean(self._output_history**2))
        if output_rms < 0.01:
            return 0.0

        # Unroll ring buffer chronologically (oldest -> newest)
        pos = self._output_history_pos
        history = np.concatenate(
            [self._output_history[pos:], self._output_history[:pos]]
        )

        # Decimate to ~11-12kHz (anti-aliased) for a cheap lag scan
        decim = max(1, self._runtime_mic_sample_rate // 11025)
        if decim > 1:
            needle = resample_poly(input_audio, 1, decim).astype(np.float32)
            haystack = resample_poly(history, 1, decim).astype(np.float32)
        else:
            needle = input_audio
            haystack = history

        return _max_normalized_lag_correlation(needle, haystack)

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
                    self.on_error(
                        "フィードバック検出: 入力と出力が接続されている可能性があります。"
                    )

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

            # Warm up with the same params as production (index/gate disabled)
            # so shape-dependent kernels match the real path.
            output_model = self.pipeline.infer_streaming(
                chunk_16k,
                overlap_16k,
                self._build_streaming_params(index_rate=0.0, voice_gate_mode="off"),
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

    def _build_streaming_params(
        self,
        *,
        index_rate: float,
        voice_gate_mode: str,
    ) -> StreamingParams:
        """Build a StreamingParams bundle from the current config.

        Centralizes the config -> infer_streaming() mapping so the production
        and warmup call sites don't each transcribe every field. ``index_rate``
        and ``voice_gate_mode`` are passed explicitly because warmup overrides
        them (0.0 / "off").
        """
        cfg = self.config
        return StreamingParams(
            pitch_shift=cfg.pitch_shift,
            f0_method=cfg.f0_method if cfg.use_f0 else "none",
            index_rate=index_rate,
            index_k=cfg.index_k,
            voice_gate_mode=voice_gate_mode,
            energy_threshold=cfg.energy_threshold,
            use_parallel_extraction=cfg.use_parallel_extraction,
            noise_scale=cfg.noise_scale,
            sola_extra_samples=self._sola_extra_model,
            pre_hubert_pitch_ratio=cfg.pre_hubert_pitch_ratio,
            moe_boost=cfg.moe_boost,
            f0_lowpass_cutoff_hz=cfg.f0_lowpass_cutoff_hz,
            enable_octave_flip_suppress=cfg.enable_octave_flip_suppress,
            enable_f0_slew_limit=cfg.enable_f0_slew_limit,
            f0_slew_max_step_st=cfg.f0_slew_max_step_st,
            hubert_context_sec=cfg.hubert_context_sec,
            fixed_harmonics=cfg.fixed_harmonics,
            f0_context_sec=cfg.f0_context_sec,
            f0_hole_fill_ms=cfg.f0_hole_fill_ms,
            uv_ramp_ms=cfg.uv_ramp_ms,
        )

    def _reset_boundary_continuity_state(self) -> None:
        """Reset chunk-boundary gain continuity state."""
        self._prev_tail_rms = 0.0
        self._postprocessor.reset()

    def _apply_output_boundary_gain(self, output: np.ndarray) -> np.ndarray:
        """Apply a mild post-SOLA gain ramp to smooth boundary loudness.

        DISABLED: this runs *after* SOLA and applies a gain step at the chunk
        boundary (prev tail ends at gain 1.0, next head starts at ``ratio``
        0.9..1.1, ramping back over 10ms).  On sustained tones any >2% RMS
        difference between the SOLA splice region and the previous tail fires
        this every chunk, amplitude-modulating the seam at the chunk rate
        (~4Hz) — audible as periodic stutter.  SOLA + the stateful
        limiter/treble already keep boundary loudness smooth, so the
        post-SOLA gain step does more harm than good.  Kept as a no-op for
        easy re-enable / experimentation.
        """
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
