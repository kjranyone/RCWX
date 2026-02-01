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
    chunk_sec: float = 0.10  # Ultra-low latency: 100ms (FCPE official min, RMVPE needs >= 0.32 sec)
    crossfade_sec: float = 0.05
    input_gain_db: float = 0.0  # Input gain in dB
    # Input channel selection for stereo devices: "left", "right", "average"
    input_channel_selection: str = "average"
    # Latency settings
    prebuffer_chunks: int = 1  # Chunks to buffer before output (0=lowest latency)
    buffer_margin: float = 0.3  # Buffer margin multiplier (0.3=tight, 0.5=balanced, 1.0=relaxed)


@dataclass
class DenoiseConfig:
    """Noise cancellation configuration."""

    enabled: bool = False
    method: str = "auto"  # auto, deepfilter, spectral, off
    # Spectral gate parameters (used when method=spectral)
    threshold_db: float = 6.0
    reduction_db: float = -24.0


@dataclass
class InferenceConfig:
    """Inference configuration."""

    pitch_shift: int = 0  # semitones
    use_f0: bool = True
    # F0 extraction method: "fcpe" (fast, 100ms min) or "rmvpe" (accurate, 320ms min)
    f0_method: str = "fcpe"
    use_index: bool = False
    index_ratio: float = 0.5
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
    voice_gate_mode: str = "expand"
    # Energy threshold for "energy" mode (0.01-0.2, default 0.05)
    # Lower = more sensitive (catches quieter sounds but may pass noise)
    # Higher = less sensitive (better noise rejection but may cut soft sounds)
    energy_threshold: float = 0.05
    # Feature caching for chunk continuity (blends HuBERT/F0 at boundaries)
    use_feature_cache: bool = True

    # --- Low-latency processing ---
    # Context: extra audio on left side for stable edge processing
    # 0.05 = 50ms context (minimal for low latency)
    context_sec: float = 0.05

    # Lookahead: future samples (ADDS LATENCY!)
    # 0 = no lookahead (lowest latency)
    lookahead_sec: float = 0.0

    # Extra discard: additional samples to remove beyond context
    extra_sec: float = 0.0

    # Chunking mode: "wokada" (context-based), "rvc_webui" (overlap-based), "hybrid" (RVC hop + w-okada context)
    chunking_mode: str = "wokada"

    # Crossfade length for SOLA blending (50ms is sufficient)
    crossfade_sec: float = 0.05

    # Enable SOLA (Synchronized Overlap-Add) for optimal crossfade position
    # Uses RVC-style correlation-based phase alignment
    use_sola: bool = True

    # Chunking mode: "wokada" (context-based, default) or "rvc_webui" (overlap-based, perfect continuity)
    chunking_mode: str = "wokada"

    denoise: DenoiseConfig = field(default_factory=DenoiseConfig)


@dataclass
class RCWXConfig:
    """Main configuration for RCWX."""

    models_dir: str = field(default_factory=lambda: str(_default_models_dir()))
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

        # Migrate old inference config keys to w-okada style
        if "use_input_overlap" in inference_data:
            inference_data.pop("use_input_overlap")
        if "use_overlap_crossfade" in inference_data:
            inference_data.pop("use_overlap_crossfade")
        if "overlap_sec" in inference_data:
            # Migrate overlap_sec to context_sec
            inference_data["context_sec"] = inference_data.pop("overlap_sec")

        return cls(
            audio=AudioConfig(**audio_data),
            inference=InferenceConfig(denoise=denoise_config, **inference_data),
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
