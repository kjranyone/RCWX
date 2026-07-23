"""Configuration and statistics dataclasses for the realtime pipeline.

Split out of realtime_unified.py so GUI widgets and tests can import the
dataclasses without pulling in the full processing stack.  Both names are
re-exported from rcwx.pipeline.realtime_unified for backward compatibility.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

LATENCY_MODES = {"normal", "aggressive"}
DEADLINE_MODES = {"aggressive"}


@dataclass
class RealtimeStats:
    """Statistics for real-time processing."""

    latency_ms: float = 0.0
    hop_latency_ms: float = 0.0
    buffer_latency_ms: float = 0.0
    queue_latency_ms: float = 0.0
    sola_latency_ms: float = 0.0
    inference_ms: float = 0.0
    inference_p50_ms: float = 0.0
    inference_p95_ms: float = 0.0
    inference_p99_ms: float = 0.0
    deadline_misses: int = 0
    deadline_miss_rate: float = 0.0
    # Per-stage breakdown of inference_ms (from RVCPipeline.stage_times)
    hubert_ms: float = 0.0
    f0_ms: float = 0.0
    faiss_ms: float = 0.0
    synth_ms: float = 0.0
    output_resample_ms: float = 0.0
    jitter_guard_ms: float = 0.0
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
        self.hop_latency_ms = 0.0
        self.buffer_latency_ms = 0.0
        self.queue_latency_ms = 0.0
        self.sola_latency_ms = 0.0
        self.inference_ms = 0.0
        self.inference_p50_ms = 0.0
        self.inference_p95_ms = 0.0
        self.inference_p99_ms = 0.0
        self.deadline_misses = 0
        self.deadline_miss_rate = 0.0
        self.hubert_ms = 0.0
        self.f0_ms = 0.0
        self.faiss_ms = 0.0
        self.synth_ms = 0.0
        self.output_resample_ms = 0.0
        self.jitter_guard_ms = 0.0
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
    latency_mode: str = "normal"
    overlap_sec: float = 0.10  # audio-level overlap for HuBERT continuity
    crossfade_sec: float = 0.02
    # SOLA search window (ms): one period of the lowest expected output F0
    # (70Hz -> 14.3ms) + margin, so the splice can always phase-align.
    sola_search_ms: float = 15.0
    prebuffer_chunks: int = 1
    buffer_margin: float = 0.25

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
    denoise_strength: float = 1.0

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
        if self.latency_mode not in LATENCY_MODES:
            object.__setattr__(self, "latency_mode", "normal")
        object.__setattr__(
            self,
            "denoise_strength",
            max(0.5, min(2.0, float(self.denoise_strength))),
        )
        minimum_prebuffer = 3
        if self.latency_mode in DEADLINE_MODES and self.prebuffer_chunks < minimum_prebuffer:
            object.__setattr__(self, "prebuffer_chunks", minimum_prebuffer)
        # Round chunk_sec to HuBERT frame boundary (20ms)
        frame_ms = 20
        chunk_ms = self.chunk_sec * 1000
        minimum_chunk_ms = 20 if self.latency_mode == "aggressive" else 40
        rounded_ms = max(
            minimum_chunk_ms,
            min(600, round(chunk_ms / frame_ms) * frame_ms),
        )
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
