"""Configuration management with JSON persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


def _default_models_dir() -> Path:
    return Path.home() / ".cache" / "rcwx" / "models"


@dataclass
class AudioConfig:
    """Audio configuration."""

    input_device_name: Optional[str] = None  # Device name (more stable than index)
    output_device_name: Optional[str] = None
    input_hostapi_filter: str = "WASAPI"  # Host API filter for input devices
    output_hostapi_filter: str = "WASAPI"  # Host API filter for output devices
    sample_rate: int = 16000
    output_sample_rate: int = 48000
    chunk_sec: float = 0.3
    crossfade_sec: float = 0.05
    input_gain_db: float = 0.0  # Input gain in dB
    # Input channel selection for stereo devices: "left", "right", "average"
    input_channel_selection: str = "auto"
    # Latency settings
    prebuffer_chunks: int = 1  # Chunks to buffer before output (0=lowest latency)
    buffer_margin: float = 0.5  # Buffer margin multiplier (0.3=tight, 0.5=balanced, 1.0=relaxed)


@dataclass
class DenoiseConfig:
    """Noise cancellation configuration."""

    enabled: bool = True
    method: str = "ml"  # auto, ml, spectral, off
    # Spectral gate parameters (used when method=spectral)
    threshold_db: float = 6.0
    reduction_db: float = -24.0


@dataclass
class InferenceConfig:
    """Inference configuration."""

    pitch_shift: int = 0  # semitones
    use_f0: bool = True
    # F0 extraction method: "rmvpe" (accurate), "fcpe" (fast), or "swiftf0" (ultra-fast ONNX/CPU)
    f0_method: str = "rmvpe"
    use_index: bool = True
    index_ratio: float = 0.15
    index_k: int = 4  # FAISS neighbors to search (4=fast, 8=quality)
    # torch.compile: Disabled for Windows XPU stability (unstable performance)
    use_compile: bool = False
    # Resampling method: "poly" (high quality) or "linear" (fast)
    resample_method: str = "linear"
    # Parallel HuBERT+F0 extraction (GPU streams, ~10-15% speedup)
    use_parallel_extraction: bool = True
    # Voice gate mode: "off", "strict", "expand", "energy"
    # - off: no voice gate (all audio passes through)
    # - strict: F0-based only (original, may cut plosives)
    # - expand: expand voiced regions to include plosives
    # - energy: use energy + F0 (plosives with energy pass through)
    voice_gate_mode: str = "off"
    # Energy threshold for "energy" mode (0.01-0.2, default 0.05)
    # Lower = more sensitive (catches quieter sounds but may pass noise)
    # Higher = less sensitive (better noise rejection but may cut soft sounds)
    energy_threshold: float = 0.2

    # Audio-level overlap for HuBERT continuity
    overlap_sec: float = 0.20

    # Crossfade length for SOLA blending
    crossfade_sec: float = 0.08

    # Enable SOLA (Synchronized Overlap-Add) for optimal crossfade position
    use_sola: bool = True

    # SOLA search window in ms
    sola_search_ms: float = 10.0

    # HuBERT context window in seconds (longer = more stable timbre across chunks)
    hubert_context_sec: float = 1.0

    # Pre-HuBERT pitch shift ratio (0.0=disabled, 1.0=full pitch shift applied before HuBERT)
    pre_hubert_pitch_ratio: float = 0.08

    # Moe voice style strength (0.0=off, 1.0=strong)
    moe_boost: float = 0.45

    # Synthesizer noise scale (0.0=deterministic, 0.66666=original RVC default)
    noise_scale: float = 0.45
    # F0 lowpass cutoff frequency in Hz (higher = more pitch detail preserved)
    f0_lowpass_cutoff_hz: float = 16.0
    # Stabilize 1-octave frame flips in F0 contour
    enable_octave_flip_suppress: bool = True
    # Limit frame-to-frame F0 slew in semitones
    enable_f0_slew_limit: bool = True
    # Max frame-to-frame F0 step (semitones) when slew limiter is enabled
    f0_slew_max_step_st: float = 3.6

    denoise: DenoiseConfig = field(default_factory=DenoiseConfig)


@dataclass
class RCWXConfig:
    """Main configuration for RCWX."""

    models_dir: str = field(default_factory=lambda: str(_default_models_dir()))
    rvc_models_dir: Optional[str] = None  # RVC model directory for dropdown scan
    last_model_path: Optional[str] = None
    device: str = "auto"  # auto, xpu, cuda, cpu
    dtype: str = "float16"  # float16, float32, bfloat16

    audio: AudioConfig = field(default_factory=AudioConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> RCWXConfig:
        """Load configuration from JSON file."""
        if path is None:
            path = cls.default_path()

        if not path.exists():
            return cls()

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        audio_data = data.pop("audio", {})
        inference_data = data.pop("inference", {})

        # Migrate old config keys
        if "input_device" in audio_data:
            audio_data.pop("input_device")  # Remove old key (was int, now using name)
        if "output_device" in audio_data:
            audio_data.pop("output_device")  # Remove old key (was int, now using name)

        # Handle nested denoise config
        denoise_data = inference_data.pop("denoise", {})
        denoise_config = DenoiseConfig(**denoise_data) if denoise_data else DenoiseConfig()

        # Filter out any unknown keys to prevent TypeError
        import dataclasses
        valid_fields = {f.name for f in dataclasses.fields(InferenceConfig)} - {"denoise"}
        filtered_inference = {k: v for k, v in inference_data.items() if k in valid_fields}

        return cls(
            audio=AudioConfig(**audio_data),
            inference=InferenceConfig(denoise=denoise_config, **filtered_inference),
            **data,
        )

    def save(self, path: Optional[Path] = None) -> None:
        """Save configuration to JSON file."""
        if path is None:
            path = self.default_path()

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @staticmethod
    def default_path() -> Path:
        """Return the default configuration file path."""
        return Path.home() / ".config" / "rcwx" / "config.json"

    def get_models_dir(self) -> Path:
        """Return the models directory as Path."""
        return Path(self.models_dir)
