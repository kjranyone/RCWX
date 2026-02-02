"""RVC inference pipeline."""

from __future__ import annotations

import logging
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt

from rcwx.audio.denoise import denoise as denoise_audio
from rcwx.audio.resample import resample
from rcwx.device import get_device, get_dtype
from rcwx.downloader import get_hubert_path, get_rmvpe_path
from rcwx.models.hubert_loader import HuBERTLoader
from rcwx.models.rmvpe import RMVPE
from rcwx.models.fcpe import FCPE, is_fcpe_available
from rcwx.models.synthesizer import SynthesizerLoader

logger = logging.getLogger(__name__)

# Minimum feature frames required by the synthesizer decoder
# The decoder uses upsampling convolutions that require sufficient input length
# 64 frames @ 100fps = 640ms worth of features
MIN_SYNTH_FEATURE_FRAMES = 64


def highpass_filter(audio: np.ndarray, sr: int = 16000, cutoff: int = 48) -> np.ndarray:
    """Apply high-pass filter to remove DC offset and low-frequency noise.

    Original RVC uses 5th order Butterworth filter with 48Hz cutoff at 16kHz.
    """
    if len(audio) < 100:  # Too short to filter
        return audio
    nyquist = sr / 2
    normalized_cutoff = cutoff / nyquist
    b, a = butter(5, normalized_cutoff, btype="high")
    return filtfilt(b, a, audio).astype(np.float32)


class RVCPipeline:
    """
    Complete RVC voice conversion pipeline.

    Integrates HuBERT feature extraction, optional RMVPE F0 extraction,
    FAISS index retrieval, and synthesizer inference.
    """

    def __init__(
        self,
        model_path: str,
        index_path: Optional[str] = None,
        device: str = "auto",
        dtype: str = "float16",
        use_f0: bool = True,
        use_compile: bool = True,
        models_dir: Optional[str] = None,
    ):
        """
        Initialize the RVC pipeline.

        Args:
            model_path: Path to the RVC .pth model
            index_path: Path to FAISS .index file (optional, auto-detected if None)
            device: Device preference (auto, xpu, cuda, cpu)
            dtype: Data type (float16, float32, bfloat16)
            use_f0: Whether to use F0 extraction (if model supports it)
            use_compile: Whether to use torch.compile optimization
            models_dir: Directory containing HuBERT and RMVPE models
        """
        self.model_path = Path(model_path)

        # Auto-detect index file if not provided
        if index_path:
            self.index_path = Path(index_path)
        else:
            # Look for .index file in same directory as model
            index_candidates = list(self.model_path.parent.glob("*.index"))
            self.index_path = index_candidates[0] if index_candidates else None

        # Resolve device and dtype
        self.device = get_device(device)
        self.dtype = get_dtype(self.device, dtype)

        # torch.compile: XPU uses inductor backend (no Triton needed)
        # CUDA on Windows needs Triton, which is not supported
        if use_compile and sys.platform == "win32" and self.device == "cuda":
            logger.info("torch.compile disabled for CUDA on Windows (Triton not supported)")
            use_compile = False
        self.use_compile = use_compile

        # Model directory for HuBERT/RMVPE
        if models_dir:
            self.models_dir = Path(models_dir)
        else:
            self.models_dir = Path.home() / ".cache" / "rcwx" / "models"

        # Components (initialized lazily)
        self.hubert: Optional[HuBERTFeatureExtractor] = None
        self.rmvpe: Optional[RMVPE] = None
        self.fcpe: Optional[FCPE] = None
        self.synthesizer: Optional[SynthesizerLoader] = None

        # FAISS index components
        self.faiss_index = None
        self.index_features: Optional[np.ndarray] = None  # big_npy

        # Model properties
        self.has_f0: bool = use_f0
        self.sample_rate: int = 40000
        self._loaded: bool = False

        # Feature cache for chunk continuity
        # Stores the last N frames of HuBERT features for blending with next chunk
        self._cached_features: Optional[torch.Tensor] = None
        self._cached_f0: Optional[torch.Tensor] = None
        self._feature_cache_frames: int = 10  # Cache 10 frames (200ms at 50fps) - optimal for chunk continuity

    def load(self) -> None:
        """Load all models."""
        if self._loaded:
            return

        # Set deterministic behavior for reproducible inference
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.manual_seed_all(0)

        logger.info(f"Loading RVC pipeline on {self.device} with {self.dtype}")

        # Load synthesizer first to detect model type
        self.synthesizer = SynthesizerLoader(
            str(self.model_path),
            device=self.device,
            dtype=self.dtype,
            use_compile=self.use_compile,
        )
        self.synthesizer.load()

        # Update properties based on loaded model
        self.sample_rate = self.synthesizer.sample_rate
        self.has_f0 = self.synthesizer.has_f0 and self.has_f0

        # Load HuBERT
        hubert_path = get_hubert_path(self.models_dir)
        logger.info(f"Loading HuBERT from: {hubert_path}")

        # Use the new loader that handles both RVC and transformers formats
        self.hubert = HuBERTLoader(
            str(hubert_path) if hubert_path.exists() else None,
            device=self.device,
            dtype=self.dtype,
        )

        if self.use_compile and hasattr(self.hubert, "model"):
            logger.info("Compiling HuBERT model...")
            self.hubert.model = torch.compile(self.hubert.model, mode="reduce-overhead")

        # Load F0 models if F0 is used
        if self.has_f0:
            # Load RMVPE
            rmvpe_path = get_rmvpe_path(self.models_dir)
            if rmvpe_path.exists():
                self.rmvpe = RMVPE(
                    str(rmvpe_path),
                    device=self.device,
                    dtype=self.dtype,
                )
                if self.use_compile:
                    logger.info("Compiling RMVPE model...")
                    self.rmvpe.model = torch.compile(self.rmvpe.model, mode="reduce-overhead")
            else:
                logger.warning("RMVPE model not found")

            # Load FCPE if available (lightweight alternative)
            if is_fcpe_available():
                try:
                    self.fcpe = FCPE(
                        device=self.device,
                        dtype=self.dtype,
                    )
                    logger.info("FCPE model loaded (low-latency F0 available)")
                except Exception as e:
                    logger.warning(f"Failed to load FCPE: {e}")
                    self.fcpe = None
            else:
                logger.info("FCPE not available (install with: pip install torchfcpe)")

            # Disable F0 if neither model is available
            if self.rmvpe is None and self.fcpe is None:
                logger.warning("No F0 model available, F0 extraction disabled")
                self.has_f0 = False

        # Load FAISS index if available
        if self.index_path and self.index_path.exists():
            self._load_faiss_index()

        self._loaded = True
        logger.info("RVC pipeline loaded successfully")

    def _load_faiss_index(self) -> None:
        """Load FAISS index for feature retrieval."""
        try:
            import faiss

            logger.info(f"Loading FAISS index from: {self.index_path}")
            self.faiss_index = faiss.read_index(str(self.index_path))

            # Reconstruct all feature vectors from the index
            # These are used for weighted averaging during retrieval
            self.index_features = self.faiss_index.reconstruct_n(0, self.faiss_index.ntotal)
            logger.info(
                f"FAISS index loaded: {self.faiss_index.ntotal} vectors, "
                f"dim={self.index_features.shape[1]}"
            )
        except ImportError:
            logger.warning("faiss-cpu not installed, index retrieval disabled")
            self.faiss_index = None
            self.index_features = None
        except Exception as e:
            logger.warning(f"Failed to load FAISS index: {e}")
            self.faiss_index = None
            self.index_features = None

    def _search_index(
        self,
        features: torch.Tensor,
        index_rate: float = 0.5,
        k: int = 4,
    ) -> torch.Tensor:
        """
        Search FAISS index and blend retrieved features with original.

        Args:
            features: HuBERT features [B, T, C]
            index_rate: Blending ratio (0=original only, 1=index only)
            k: Number of nearest neighbors to retrieve (4=fast, 8=quality)

        Returns:
            Blended features [B, T, C]
        """
        if self.faiss_index is None or index_rate <= 0:
            return features

        # Convert to numpy for FAISS search
        npy = features[0].cpu().numpy()
        if self.dtype == torch.float16:
            npy = npy.astype(np.float32)

        # Search for k nearest neighbors (k=4 for speed, k=8 for quality)
        logger.debug(
            f"FAISS search: input shape={npy.shape}, k={k}, index_vectors={self.faiss_index.ntotal}"
        )
        score, ix = self.faiss_index.search(npy, k=k)
        logger.debug(
            f"FAISS search results: scores shape={score.shape}, min={score.min():.4f}, max={score.max():.4f}"
        )

        # Compute inverse squared distance weights
        # Add small epsilon to avoid division by zero
        weight = np.square(1 / (score + 1e-6))
        weight /= weight.sum(axis=1, keepdims=True)

        # Weighted average of retrieved features
        # index_features[ix] has shape [T, k, C]
        # weight has shape [T, k]
        retrieved = np.sum(
            self.index_features[ix] * np.expand_dims(weight, axis=2),
            axis=1,
        )  # [T, C]
        logger.debug(f"Retrieved features: mean={retrieved.mean():.4f}, std={retrieved.std():.4f}")

        if self.dtype == torch.float16:
            retrieved = retrieved.astype(np.float16)

        # Blend with original features
        retrieved_tensor = torch.from_numpy(retrieved).unsqueeze(0).to(features.device)
        blended = index_rate * retrieved_tensor + (1 - index_rate) * features
        logger.debug(
            f"Blended features: mean={blended.mean():.4f}, std={blended.std():.4f}, index_rate={index_rate}"
        )

        return blended

    def unload(self) -> None:
        """Unload all models to free memory."""
        self.hubert = None
        self.rmvpe = None
        self.fcpe = None
        self.synthesizer = None
        self.faiss_index = None
        self.index_features = None
        self._loaded = False
        self.clear_cache()

        # Clear CUDA/XPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()

    def clear_cache(self) -> None:
        """Clear feature cache for chunk continuity.

        Call this when starting a new audio stream or after a long pause.
        """
        self._cached_features = None
        self._cached_f0 = None

    @torch.no_grad()
    def infer(
        self,
        audio: np.ndarray | torch.Tensor,
        input_sr: int = 16000,
        pitch_shift: int = 0,
        f0_method: str = "rmvpe",
        index_rate: float = 0.0,
        index_k: int = 4,
        voice_gate_mode: str = "expand",
        energy_threshold: float = 0.05,
        denoise: bool = False,
        noise_reference: Optional[np.ndarray] = None,
        use_feature_cache: bool = True,
        use_parallel_extraction: bool = True,
        allow_short_input: bool = False,
    ) -> np.ndarray:
        """
        Convert voice using the RVC pipeline.

        Args:
            audio: Input audio (1D numpy array or tensor)
            input_sr: Input sample rate (default 16kHz)
            pitch_shift: Pitch shift in semitones
            f0_method: F0 extraction method ("rmvpe" or "none")
            index_rate: FAISS index blending ratio (0=off, 0.5=balanced, 1=index only)
            index_k: Number of FAISS neighbors to search (4=fast, 8=quality, default 4)
            voice_gate_mode: Voice gate mode for unvoiced segments:
                - "off": no gating, all audio passes through
                - "strict": F0-based only (may cut plosives like p/t/k)
                - "expand": expand voiced regions to include adjacent plosives
                - "energy": use energy + F0 (plosives with energy pass through)
            energy_threshold: Energy threshold for "energy" mode (0.01-0.2, default 0.05)
            denoise: If True, apply spectral gate noise reduction before processing
            noise_reference: Optional noise sample for denoiser (auto-learns if None)
            use_feature_cache: Enable feature caching for chunk continuity (default True)
            use_parallel_extraction: Enable parallel HuBERT+F0 extraction (~10-15% speedup)

        Returns:
            Converted audio at model sample rate (usually 40kHz)
        """
        if not self._loaded:
            self.load()

        # Convert to tensor if needed
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()

        # ALWAYS resample to 16kHz regardless of input_sr parameter
        # This prevents double-resampling bug when input_sr != 16000
        if len(audio.shape) == 1:
            audio_np = audio.numpy()  # [T]
            if input_sr != 16000:
                audio_np = resample(audio_np, input_sr, 16000)
            audio = torch.from_numpy(audio_np).float()
        else:  # Multi-channel
            raise ValueError(f"Expected 1D audio, got {len(audio.shape)}D")

        # Apply noise reduction if enabled
        if denoise:
            audio_np = audio.numpy()
            # Resample noise reference if provided (ONLY resample ONCE)
            if noise_reference is not None and len(noise_reference) > 0:
                # Resample noise reference if sample rate differs
                if input_sr != 16000:
                    noise_reference = resample(noise_reference, input_sr, 16000)
                # Use DeepFilterNet if available, otherwise spectral gate
            audio_np = denoise_audio(
                audio_np,
                sample_rate=16000,
                method="auto",  # DeepFilterNet if available, else spectral gate
                noise_reference=noise_reference,
                device=self.device,
            )
            audio = torch.from_numpy(audio_np).float()
            logger.debug("Applied noise reduction")

        # Apply high-pass filter to remove DC offset and low-frequency noise
        # Original RVC uses 48Hz cutoff Butterworth filter
        audio_np = highpass_filter(audio_np, sr=16000, cutoff=48)

        # Add reflection padding for edge handling
        # Base padding: 50ms (800 samples @ 16kHz)
        # For short inputs, increase padding to ensure we get MIN_SYNTH_FEATURE_FRAMES
        # without needing feature-level padding (which causes length mismatch issues)
        original_length = len(audio_np)
        hubert_hop = 320

        # Base padding (50ms each side = 1600 total)
        base_pad = int(16000 * 0.05)  # 800 samples

        # For chunk processing, use minimal padding to avoid excessive padding artifacts
        logger.info(f"Padding check: allow_short_input={allow_short_input}, original_length={original_length}")
        if allow_short_input:
            t_pad = base_pad
        else:
            # Calculate minimum input samples needed for MIN_SYNTH_FEATURE_FRAMES features
            # MIN_SYNTH_FEATURE_FRAMES is at 100fps, HuBERT produces 50fps, so /2
            # HuBERT produces approximately (samples / hop) - 1 frames due to its internal handling,
            # so we add 2 extra hops to ensure we get enough frames
            min_hubert_frames = MIN_SYNTH_FEATURE_FRAMES // 2  # 32 frames
            min_input_samples = (min_hubert_frames + 2) * hubert_hop  # 10880 samples (with buffer)

            # Check if we need extra padding to meet minimum
            total_with_base = base_pad + original_length + base_pad
            if total_with_base < min_input_samples:
                # Need more padding - distribute evenly
                extra_needed = min_input_samples - total_with_base
                extra_per_side = (extra_needed + 1) // 2
                t_pad = base_pad + extra_per_side
                logger.info(
                    f"Short input: increased padding from {base_pad} to {t_pad} samples per side"
                )
            else:
                t_pad = base_pad

        t_pad_tgt = int(t_pad * self.sample_rate / 16000)  # Output padding samples (proportional)

        # Pad to multiple of HuBERT hop size (320) to avoid frame truncation
        padded_for_hubert = t_pad + original_length + t_pad
        remainder = padded_for_hubert % hubert_hop
        if remainder != 0:
            extra_pad = hubert_hop - remainder
        else:
            extra_pad = 0

        audio_np = np.pad(audio_np, (t_pad, t_pad + extra_pad), mode="reflect")
        logger.info(
            f"Padding: original={original_length}, t_pad={t_pad}, extra_pad={extra_pad}, final={len(audio_np)}"
        )

        audio = torch.from_numpy(audio_np).float()

        # Ensure 2D for batch processing
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        audio = audio.to(self.device)

        # Debug: input audio stats
        logger.info(
            f"Input audio: shape={audio.shape} (padded from {original_length}), min={audio.min():.4f}, max={audio.max():.4f}"
        )

        # Determine output dimension and layer based on model version
        # v1 models: layer 9, 256-dim features
        # v2 models: layer 12, 768-dim features (original RVC specification)
        if self.synthesizer.version == 1:
            output_dim = 256
            output_layer = 9
        else:
            output_dim = 768
            output_layer = 12

        # Parallel extraction: HuBERT features + F0 (if enabled)
        # Use ThreadPoolExecutor for ~10% speedup (more stable than GPU streams)
        features = None
        f0_raw = None

        if use_parallel_extraction and self.has_f0:

            def extract_hubert():
                with torch.autocast(device_type=self.device, dtype=self.dtype):
                    return self.hubert.extract(
                        audio, output_layer=output_layer, output_dim=output_dim
                    )

            def extract_f0():
                if f0_method == "fcpe" and self.fcpe is not None:
                    with torch.autocast(device_type=self.device, dtype=self.dtype):
                        return self.fcpe.infer(audio, threshold=0.006)
                elif self.rmvpe is not None:
                    with torch.autocast(device_type=self.device, dtype=self.dtype):
                        return self.rmvpe.infer(audio)
                return None

            # Run HuBERT and F0 extraction in parallel threads
            with ThreadPoolExecutor(max_workers=2) as executor:
                hubert_future = executor.submit(extract_hubert)
                f0_future = executor.submit(extract_f0)

                features = hubert_future.result()
                f0_raw = f0_future.result()

            logger.debug("Parallel extraction complete (HuBERT + F0, ThreadPool)")

        # Fallback: sequential extraction
        if features is None:
            with torch.autocast(device_type=self.device, dtype=self.dtype):
                features = self.hubert.extract(
                    audio, output_layer=output_layer, output_dim=output_dim
                )

        logger.info(
            f"HuBERT features: shape={features.shape}, min={features.min():.4f}, max={features.max():.4f}"
        )

        # Apply FAISS index retrieval if enabled (before interpolation, like original RVC)
        if index_rate > 0 and self.faiss_index is not None:
            logger.info(
                f"Applying index retrieval: index_rate={index_rate}, k={index_k}, features_before={features.shape} mean={features.mean():.4f} std={features.std():.4f}"
            )
            features = self._search_index(features, index_rate, k=index_k)
            logger.info(
                f"Index retrieval applied: index_rate={index_rate}, features_after={features.shape} mean={features.mean():.4f} std={features.std():.4f}"
            )

        # Apply feature cache blending for chunk continuity
        if use_feature_cache and self._cached_features is not None:
            cache_frames = min(
                self._feature_cache_frames, features.shape[1], self._cached_features.shape[1]
            )
            if cache_frames > 0:
                # Blend cached features with current features at the beginning
                # Use cosine crossfade for smoother transition: alpha goes from 1 (use cached) to 0 (use current)
                t = torch.linspace(0, 1, cache_frames, device=features.device, dtype=features.dtype)
                alpha = 0.5 * (1 + torch.cos(torch.pi * t))  # Cosine fade: 1 -> 0
                alpha = alpha.view(1, cache_frames, 1)  # [1, T, 1] for broadcasting
                cached = self._cached_features[:, -cache_frames:, :]  # Last N frames of cache
                current = features[:, :cache_frames, :]  # First N frames of current
                blended = alpha * cached + (1 - alpha) * current
                features = torch.cat([blended, features[:, cache_frames:, :]], dim=1)
                logger.debug(f"Feature cache blended: {cache_frames} frames (cosine)")

        # Cache features for next chunk (before interpolation, at 50fps)
        if use_feature_cache:
            self._cached_features = features[:, -self._feature_cache_frames :, :].clone()

        # Interpolate features to match synthesizer expectation
        # RVC uses bilinear (linear) interpolation for 2x upscale
        # HuBERT hop=320 @ 16kHz (50fps) -> Synthesizer needs 100fps
        original_frames = features.shape[1]
        features = torch.nn.functional.interpolate(
            features.permute(0, 2, 1),  # [B, T, C] -> [B, C, T]
            scale_factor=2,  # Fixed 2x upscale (matches original RVC)
            mode="linear",  # RVC uses bilinear interpolation
            align_corners=False,
        ).permute(0, 2, 1)  # [B, C, T] -> [B, T, C]
        logger.info(
            f"Interpolated features: {original_frames} -> {features.shape[1]} frames (2x linear)"
        )

        # Feature length
        feature_lengths = torch.tensor([features.shape[1]], dtype=torch.long, device=self.device)

        # Extract F0 if using F0 model
        pitch = None
        pitchf = None
        if self.has_f0:
            # Use F0 from parallel extraction if available
            f0 = f0_raw if f0_raw is not None else None

            # Otherwise extract sequentially (fallback or CPU mode)
            if f0 is None:
                # Use FCPE if requested and available
                if f0_method == "fcpe" and self.fcpe is not None:
                    with torch.autocast(device_type=self.device, dtype=self.dtype):
                        f0 = self.fcpe.infer(audio, threshold=0.006)
                    logger.debug("F0 extracted with FCPE (sequential)")

                # Use RMVPE if requested and available (or fallback if FCPE failed)
                elif self.rmvpe is not None and (f0_method == "rmvpe" or f0 is None):
                    with torch.autocast(device_type=self.device, dtype=self.dtype):
                        f0 = self.rmvpe.infer(audio)
                    logger.debug("F0 extracted with RMVPE (sequential)")
            else:
                logger.debug(f"F0 from parallel extraction ({f0_method})")

            if f0 is not None:
                # Apply pitch shift (only to voiced regions where f0 > 0)
                if pitch_shift != 0:
                    f0 = torch.where(f0 > 0, f0 * (2 ** (pitch_shift / 12)), f0)

                # Align F0 length with features
                # RMVPE: hop=160 (100 frames/sec), HuBERT: hop=320 (50 frames/sec)
                # F0 length is approximately 2x feature length
                if f0.shape[1] != features.shape[1]:
                    f0 = torch.nn.functional.interpolate(
                        f0.unsqueeze(1),
                        size=features.shape[1],
                        mode="linear",
                        align_corners=False,
                    ).squeeze(1)

                # Apply F0 cache blending for chunk continuity
                f0_cache_frames = (
                    self._feature_cache_frames * 2
                )  # 2x because features were interpolated
                if use_feature_cache and self._cached_f0 is not None:
                    cache_frames = min(f0_cache_frames, f0.shape[1], self._cached_f0.shape[1])
                    if cache_frames > 0:
                        # Blend F0: smooth transition from cached to current
                        # Use cosine crossfade for smoother transition
                        t = torch.linspace(0, 1, cache_frames, device=f0.device, dtype=f0.dtype)
                        alpha = 0.5 * (1 + torch.cos(torch.pi * t))  # Cosine fade: 1 -> 0
                        alpha = alpha.view(1, cache_frames)
                        cached_f0 = self._cached_f0[:, -cache_frames:]
                        current_f0 = f0[:, :cache_frames]

                        # Blend strategy: more aggressive blending for better continuity
                        # Blend if EITHER has valid F0, not just both
                        either_voiced = (cached_f0 > 0) | (current_f0 > 0)
                        both_voiced = (cached_f0 > 0) & (current_f0 > 0)

                        # If both voiced: blend normally
                        # If only one voiced: use the voiced one with slight fade
                        blended_f0 = torch.where(
                            both_voiced,
                            alpha * cached_f0 + (1 - alpha) * current_f0,
                            torch.where(
                                cached_f0 > 0,
                                cached_f0 * alpha,  # Fade out cached if current is unvoiced
                                current_f0 * (1 - alpha),  # Fade in current if cached is unvoiced
                            )
                        )

                        f0 = torch.cat([blended_f0, f0[:, cache_frames:]], dim=1)
                        logger.debug(f"F0 cache blended: {cache_frames} frames (cosine, aggressive)")

                # Cache F0 for next chunk
                if use_feature_cache:
                    self._cached_f0 = f0[:, -f0_cache_frames:].clone()

                # pitchf: continuous F0 values for NSF decoder
                pitchf = f0.to(self.dtype)

                # pitch: quantized F0 for pitch embedding (256 bins)
                # Match original RVC WebUI pitch quantization exactly:
                # 1. Convert F0 to mel scale
                # 2. Normalize to 1-255 range (for voiced frames with f0_mel > 0)
                # 3. Set unvoiced/low values to 1 (NOT 0 - original RVC convention)
                f0_mel_min = 1127 * math.log(1 + 50 / 700)  # ~69.07 (50Hz)
                f0_mel_max = 1127 * math.log(1 + 1100 / 700)  # ~942.46 (1100Hz)

                # Convert F0 to mel scale (f0=0 -> f0_mel=0)
                f0_mel = 1127 * torch.log(1 + f0 / 700)

                # Only normalize voiced frames (f0_mel > 0)
                # Original: f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
                voiced_mask = f0_mel > 0
                f0_mel_normalized = torch.where(
                    voiced_mask,
                    (f0_mel - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1,
                    f0_mel,  # Keep 0 for unvoiced
                )

                # Clamp to valid range and set low values to 1
                # Original: f0_mel[f0_mel <= 1] = 1; f0_mel[f0_mel > 255] = 255
                pitch = torch.clamp(f0_mel_normalized, 1, 255).round().long()
                logger.info(
                    f"F0: shape={f0.shape}, min={f0.min():.1f}, max={f0.max():.1f}, voiced={voiced_mask.sum().item()}/{f0.numel()}, pitch_range=[{pitch.min().item()}, {pitch.max().item()}]"
                )

                # Store voiced mask for gating (will be used after synthesis)
                voiced_mask_for_gate = voiced_mask.float()  # [B, T]
            else:
                # No F0 model available - use pitch=1 (unvoiced marker per RVC convention)
                pitch = torch.ones(
                    features.shape[0], features.shape[1], dtype=torch.long, device=self.device
                )
                pitchf = torch.zeros(
                    features.shape[0], features.shape[1], dtype=self.dtype, device=self.device
                )
                voiced_mask_for_gate = None
                logger.info("F0: using unvoiced pitch=1 (no F0 model)")
        else:
            voiced_mask_for_gate = None

        # Pad features if too short for synthesizer decoder
        # The decoder's upsampling convolutions require minimum input length
        synth_pad_frames = 0
        if features.shape[1] < MIN_SYNTH_FEATURE_FRAMES:
            synth_pad_frames = MIN_SYNTH_FEATURE_FRAMES - features.shape[1]
            # Reflection pad to avoid edge artifacts
            pad_left = synth_pad_frames // 2
            pad_right = synth_pad_frames - pad_left
            features = torch.nn.functional.pad(
                features.permute(0, 2, 1),  # [B, T, C] -> [B, C, T]
                (pad_left, pad_right),
                mode="reflect",
            ).permute(0, 2, 1)  # [B, C, T] -> [B, T, C]
            # Update feature lengths
            feature_lengths = torch.tensor(
                [features.shape[1]], dtype=torch.long, device=self.device
            )
            # Pad pitch/pitchf if present
            if pitch is not None:
                pitch = torch.nn.functional.pad(pitch, (pad_left, pad_right), mode="reflect")
            if pitchf is not None:
                pitchf = torch.nn.functional.pad(pitchf, (pad_left, pad_right), mode="reflect")
            logger.info(
                f"Padded short input: {features.shape[1] - synth_pad_frames} -> {features.shape[1]} frames (min={MIN_SYNTH_FEATURE_FRAMES})"
            )

        # Run synthesizer
        logger.info(
            f"Synthesizer input: features={features.shape}, pitch={pitch.shape if pitch is not None else None}"
        )
        with torch.autocast(device_type=self.device, dtype=self.dtype):
            output = self.synthesizer.infer(
                features,
                feature_lengths,
                pitch=pitch,
                pitchf=pitchf,
            )

        logger.info(
            f"Synthesizer output: shape={output.shape}, min={output.min():.4f}, max={output.max():.4f}"
        )

        # Trim synthesizer padding if we added it for short inputs
        if synth_pad_frames > 0:
            # Calculate output samples to trim based on feature-to-audio ratio
            # Each feature frame = samples_per_frame samples at synthesizer sample rate
            samples_per_frame = self.sample_rate // 100  # 100fps features -> samples/frame
            trim_left = (synth_pad_frames // 2) * samples_per_frame
            trim_right = (synth_pad_frames - synth_pad_frames // 2) * samples_per_frame

            # Debug: check synth output before trimming
            synth_total = output.shape[-1]
            synth_tail_rms = (
                float(
                    torch.sqrt(
                        torch.mean(output[..., -trim_right - samples_per_frame : -trim_right] ** 2)
                    )
                )
                if trim_right > 0
                else 0
            )

            if output.shape[-1] > trim_left + trim_right:
                output = (
                    output[..., trim_left:-trim_right]
                    if trim_right > 0
                    else output[..., trim_left:]
                )
                logger.info(
                    f"Trimmed synth padding: {trim_left} + {trim_right} samples (synth_total={synth_total}, synth_tail_rms={synth_tail_rms:.4f})"
                )

        # Apply voice gating based on mode
        if voice_gate_mode != "off" and voiced_mask_for_gate is not None:
            output_len = output.shape[-1]
            gate_mask = voiced_mask_for_gate.clone()  # [B, T_feat]

            # Mode: expand - dilate voiced regions to include adjacent plosives
            if voice_gate_mode == "expand":
                # Expand by ~30ms on each side (covers most plosives)
                # At feature rate (~50fps), that's about 1-2 frames
                expand_frames = 2
                # Use max pooling to dilate the mask
                gate_mask = torch.nn.functional.max_pool1d(
                    gate_mask.unsqueeze(1),
                    kernel_size=expand_frames * 2 + 1,
                    stride=1,
                    padding=expand_frames,
                ).squeeze(1)

            # Mode: energy - combine F0 with energy-based detection
            elif voice_gate_mode == "energy":
                # Compute frame-level energy from output (already synthesized)
                # Use short-time energy at feature frame rate
                frame_size = output_len // gate_mask.shape[-1]
                if frame_size > 0:
                    output_frames = output.unfold(
                        -1, frame_size, frame_size
                    )  # [B, num_frames, frame_size]
                    frame_energy = (output_frames**2).mean(dim=-1)  # [B, num_frames]
                    # Normalize energy to 0-1
                    energy_max = frame_energy.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
                    energy_mask = frame_energy / energy_max
                    # Threshold: keep frames with energy above threshold
                    energy_mask = (energy_mask > energy_threshold).float()
                    # Combine: voiced OR has energy
                    if energy_mask.shape[-1] == gate_mask.shape[-1]:
                        gate_mask = torch.maximum(gate_mask, energy_mask)

            # Upsample mask to match output length
            gate_mask = torch.nn.functional.interpolate(
                gate_mask.unsqueeze(1),  # [B, 1, T_feat]
                size=output_len,
                mode="linear",
                align_corners=False,
            ).squeeze(1)  # [B, T_out]

            # Apply smooth attack/release to avoid clicks (5ms)
            smooth_samples = int(self.sample_rate * 0.005)
            if smooth_samples > 1:
                kernel = torch.ones(1, 1, smooth_samples, device=gate_mask.device) / smooth_samples
                gate_mask = torch.nn.functional.conv1d(
                    gate_mask.unsqueeze(1),
                    kernel,
                    padding=smooth_samples // 2,
                ).squeeze(1)
                # Ensure exact size match after convolution
                if gate_mask.shape[-1] != output_len:
                    gate_mask = gate_mask[..., :output_len]
                gate_mask = torch.clamp(gate_mask, 0, 1)

            # Apply gate
            output = output * gate_mask
            voiced_ratio = gate_mask.mean().item()
            logger.info(f"Voice gate ({voice_gate_mode}): {voiced_ratio * 100:.1f}% passed")

        # Convert to numpy
        output = output.cpu().float().numpy()

        if output.ndim == 2:
            output = output[0]

        # Trim padding from output (match the input padding ratio)
        # t_pad_tgt was calculated earlier based on actual t_pad used
        # Also account for extra_pad added for HuBERT alignment
        extra_pad_tgt = int(extra_pad * self.sample_rate / 16000)
        trim_start = t_pad_tgt
        trim_end = t_pad_tgt + extra_pad_tgt

        # Debug: check output before trimming
        pre_trim_tail_rms = (
            np.sqrt(np.mean(output[-trim_end - 480 : -trim_end] ** 2))
            if trim_end > 0 and len(output) > trim_end + 480
            else 0
        )
        post_trim_tail_start = -trim_end if trim_end > 0 else len(output)

        if len(output) > trim_start + trim_end:
            output = output[trim_start:-trim_end] if trim_end > 0 else output[trim_start:]
            logger.info(
                f"Trimmed {trim_start} from start, {trim_end} from end (pre-trim tail rms={pre_trim_tail_rms:.4f})"
            )

        # Note: HuBERT frame quantization causes output to be slightly shorter than ideally expected.
        # We intentionally do NOT pad/extend to match expected length, as artificial waveform
        # repetition creates artifacts at chunk boundaries. Instead:
        # - Accept the actual output length (crossfade handles variable-length chunks)
        # - Only trim if output is too long (rare edge case)
        expected_output_samples = int(original_length * self.sample_rate / 16000)
        length_diff = len(output) - expected_output_samples
        logger.info(
            f"Length check: got {len(output)}, expected {expected_output_samples}, diff={length_diff}"
        )

        if length_diff > 0:
            # Output too long - trim from end
            output = output[:expected_output_samples]
            logger.debug(f"Trimmed {length_diff} extra samples from end")
        elif length_diff < 0 and abs(length_diff) > 100:
            # Output too short - resample to stretch to expected length
            # This maintains timing alignment between chunks
            output = resample(output, len(output), expected_output_samples)
            logger.debug(
                f"Resampled output from {expected_output_samples + length_diff} to {expected_output_samples} samples"
            )

        logger.info(
            f"Final output: shape={output.shape}, min={output.min():.4f}, max={output.max():.4f}"
        )
        return output

    def get_info(self) -> dict:
        """Get information about the loaded pipeline."""
        if not self._loaded:
            return {"loaded": False}

        return {
            "loaded": True,
            "device": self.device,
            "dtype": str(self.dtype),
            "model_path": str(self.model_path),
            "index_path": str(self.index_path) if self.index_path else None,
            "has_index": self.faiss_index is not None,
            "index_vectors": self.faiss_index.ntotal if self.faiss_index else 0,
            "version": self.synthesizer.version if self.synthesizer else None,
            "has_f0": self.has_f0,
            "sample_rate": self.sample_rate,
            "use_compile": self.use_compile,
        }
