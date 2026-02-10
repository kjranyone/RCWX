"""RVC inference pipeline."""

from __future__ import annotations

import logging
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt, medfilt, resample_poly

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

# Maximum audio context for HuBERT in streaming mode (16kHz samples).
# Longer context = better feature quality (HuBERT is a transformer that
# benefits from seeing more audio).  F0 is extracted on the current chunk
# only (pitch detection is local), so this doesn't affect F0 processing time.
# 2.0s (32000 samples) gives cosine similarity ~0.60 vs batch features.
MAX_HUBERT_CONTEXT_16K = 32000  # 2.0 seconds @ 16kHz
# Keep within practical CLI/GUI pitch range.
# GUI pitch slider is -24..+24 semitones.
MAX_PRE_HUBERT_SHIFT_ST = 24.0
MAX_MOE_BOOST = 1.0


def compute_pre_hubert_shift(pitch_shift: int, ratio: float) -> float:
    """Compute bounded pre-HuBERT shift in semitones."""
    ratio = max(0.0, min(1.0, float(ratio)))
    shift = float(pitch_shift) * ratio
    if shift > MAX_PRE_HUBERT_SHIFT_ST:
        return MAX_PRE_HUBERT_SHIFT_ST
    if shift < -MAX_PRE_HUBERT_SHIFT_ST:
        return -MAX_PRE_HUBERT_SHIFT_ST
    return shift


def compute_post_f0_shift(pitch_shift: int, pre_hubert_shift: float) -> float:
    """Compute residual F0 shift after pre-HuBERT audio shift."""
    return float(pitch_shift) - float(pre_hubert_shift)


def _fit_length_center(x: np.ndarray, target_len: int) -> np.ndarray:
    """Center-crop or symmetric-pad 1D array to target length."""
    cur_len = len(x)
    if cur_len == target_len:
        return x
    if cur_len > target_len:
        start = (cur_len - target_len) // 2
        return x[start : start + target_len]

    pad_total = target_len - cur_len
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    pad_mode = "reflect" if cur_len > 1 else "edge"
    return np.pad(x, (pad_left, pad_right), mode=pad_mode)


def pitch_shift_resample(
    audio_t: torch.Tensor, sample_rate: int, semitones: float
) -> torch.Tensor:
    """Pitch-shift audio by resampling with boundary context and fixed length."""
    if abs(semitones) < 0.01:
        return audio_t

    ratio = 2.0 ** (semitones / 12.0)

    device = audio_t.device
    dtype = audio_t.dtype
    shape = audio_t.shape  # e.g. (1, N) or (N,)
    audio_np = audio_t.detach().cpu().numpy().squeeze()
    orig_len = len(audio_np)
    if orig_len < 8:
        return audio_t

    # Symmetric context reduces edge artifacts when chunks are shifted independently.
    max_reflect = max(0, orig_len - 1)
    context = min(int(sample_rate * 0.04), max_reflect, max(16, orig_len // 4))
    if context > 0:
        audio_ctx = np.pad(audio_np, (context, context), mode="reflect")
    else:
        audio_ctx = audio_np

    # resample_poly(x, up, down) => len(x) * up / down.
    # We need output_len ~= len / ratio, so up/down ~= 1/ratio.
    frac = Fraction(1.0 / ratio).limit_denominator(256)
    up = frac.numerator
    down = frac.denominator

    shifted_ctx = resample_poly(
        audio_ctx, up, down, window=("kaiser", 8.0)
    ).astype(np.float32)
    shifted_ctx = _fit_length_center(shifted_ctx, len(audio_ctx))

    if context > 0:
        shifted = shifted_ctx[context : context + orig_len]
    else:
        shifted = _fit_length_center(shifted_ctx, orig_len)
    shifted = _fit_length_center(shifted, orig_len).astype(np.float32, copy=False)

    # Anchor chunk edges to original audio to avoid boundary discontinuities
    # when processing chunk-by-chunk in realtime.
    edge = min(int(sample_rate * 0.01), orig_len // 8)
    if edge >= 4:
        ramp = np.linspace(0.0, 1.0, edge, dtype=np.float32)
        shifted[:edge] = audio_np[:edge] * (1.0 - ramp) + shifted[:edge] * ramp
        shifted[-edge:] = shifted[-edge:] * (1.0 - ramp) + audio_np[-edge:] * ramp

    result = torch.from_numpy(shifted)
    if len(shape) > 1:
        result = result.unsqueeze(0)
    return result.to(device=device, dtype=dtype)


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


def sigmoid_blend_weights(steps: int, steepness: float = 4.0) -> np.ndarray:
    """Generate sigmoid-shaped blend weights for smoother transitions.

    Args:
        steps: Number of blend steps
        steepness: Controls the sharpness of the transition (4.0 = smooth S-curve)

    Returns:
        Array of weights from ~1.0 to ~0.0 (sigmoid curve)
    """
    x = np.linspace(-steepness, steepness, steps)
    return 1.0 / (1.0 + np.exp(x))


def smooth_f0_spikes(f0: torch.Tensor, window: int = 3) -> torch.Tensor:
    """Remove F0 spikes using median filter on voiced regions.

    Args:
        f0: F0 tensor [B, T]
        window: Median filter window size (odd number, default 3)

    Returns:
        Smoothed F0 tensor with spikes removed
    """
    if f0.shape[1] < window:
        return f0

    # Convert to numpy for scipy median filter (must be float32 for medfilt)
    f0_np = f0.cpu().to(torch.float32).numpy()
    result = np.zeros_like(f0_np)

    for b in range(f0_np.shape[0]):
        # Get voiced mask (f0 > 0)
        voiced = f0_np[b] > 0

        # Apply median filter only to voiced regions
        if np.any(voiced):
            # medfilt preserves array length
            filtered = medfilt(f0_np[b].astype(np.float64), window).astype(np.float32)

            # Only apply to voiced regions (keep unvoiced as 0)
            result[b] = np.where(voiced, filtered, f0_np[b])
        else:
            result[b] = f0_np[b]

    return torch.from_numpy(result).to(f0.device, dtype=f0.dtype)


def lowpass_f0(f0: torch.Tensor, cutoff_hz: float = 16.0, sample_rate: float = 100.0) -> torch.Tensor:
    """Apply lowpass filter to F0 to remove high-frequency jitter.

    Phase 6: Butterworth lowpass filter for smoother F0 contour.
    Only applies to voiced regions to preserve unvoiced detection.

    Args:
        f0: F0 tensor [B, T] in Hz
        cutoff_hz: Cutoff frequency in Hz (default 8Hz - removes jitter above ~8Hz)
        sample_rate: F0 sample rate in Hz (default 100fps)

    Returns:
        Lowpass filtered F0 tensor
    """
    if f0.shape[1] < 10:  # Too short to filter
        return f0

    # Convert to numpy
    f0_np = f0.cpu().to(torch.float32).numpy()
    result = np.zeros_like(f0_np)

    # Design lowpass filter
    nyquist = sample_rate / 2
    normalized_cutoff = cutoff_hz / nyquist
    # Use order 2 for gentler filtering
    b, a = butter(2, normalized_cutoff, btype="low")

    for batch in range(f0_np.shape[0]):
        voiced = f0_np[batch] > 0

        if np.sum(voiced) > 10:  # Need enough voiced samples
            # Extract voiced regions and interpolate gaps for filtering
            f0_interp = f0_np[batch].copy()

            # Simple linear interpolation for unvoiced gaps
            voiced_indices = np.where(voiced)[0]
            if len(voiced_indices) > 1:
                # Interpolate between voiced regions
                for i in range(len(voiced_indices) - 1):
                    start = voiced_indices[i]
                    end = voiced_indices[i + 1]
                    if end - start > 1:  # There's a gap
                        f0_interp[start:end + 1] = np.linspace(
                            f0_np[batch, start], f0_np[batch, end], end - start + 1
                        )

            # Apply lowpass filter
            try:
                filtered = filtfilt(b, a, f0_interp).astype(np.float32)
                # Only keep filtered values in voiced regions
                result[batch] = np.where(voiced, filtered, 0.0)
            except ValueError:
                # Filter failed, return original
                result[batch] = f0_np[batch]
        else:
            result[batch] = f0_np[batch]

    return torch.from_numpy(result).to(f0.device, dtype=f0.dtype)


def _smooth_fcpe_f0(f0: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Smooth FCPE F0 with avg_pool1d, preserving unvoiced (f0=0) regions."""
    f0_smooth = torch.nn.functional.avg_pool1d(
        f0.unsqueeze(1), kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
    ).squeeze(1)
    return torch.where(f0 > 0, f0_smooth, f0)


def apply_moe_f0_style(f0: torch.Tensor, strength: float) -> torch.Tensor:
    """Apply a moe F0 style: clearer accents + reduced breathy dropouts."""
    strength = max(0.0, min(MAX_MOE_BOOST, float(strength)))
    if strength <= 0.0 or f0.numel() == 0:
        return f0

    def _fill_short_unvoiced_gaps(row: torch.Tensor, max_gap_frames: int) -> torch.Tensor:
        """Interpolate very short unvoiced gaps to reduce F0 flicker."""
        if max_gap_frames <= 0 or row.numel() == 0:
            return row
        filled = row.clone()
        voiced = filled > 0
        n = int(filled.shape[0])
        i = 0
        while i < n:
            if voiced[i]:
                i += 1
                continue
            start = i
            while i < n and not voiced[i]:
                i += 1
            end = i
            gap = end - start
            if (
                gap > 0
                and gap <= max_gap_frames
                and start > 0
                and end < n
                and voiced[start - 1]
                and voiced[end]
            ):
                left = filled[start - 1]
                right = filled[end]
                w = torch.linspace(
                    0.0, 1.0, gap + 2, device=filled.device, dtype=filled.dtype
                )[1:-1]
                filled[start:end] = left * (1.0 - w) + right * w
        return filled

    max_gap_frames = int(2 + 3 * strength)  # 20-50ms @ 100fps
    accent_up = 1.0 + 0.85 * strength
    accent_down = 1.0 + 0.15 * strength
    bias_st = 0.9 * strength
    max_dev_st = 18.0

    styled = f0.clone()
    for b in range(styled.shape[0]):
        row = _fill_short_unvoiced_gaps(styled[b], max_gap_frames=max_gap_frames)
        voiced = row > 0
        if voiced.sum().item() < 4:
            styled[b] = row
            continue

        voiced_f0 = torch.clamp(row[voiced], min=1e-5, max=1400.0)
        log2_f0 = torch.log2(voiced_f0)
        window = max(5, int(5 + 12 * strength))
        if window % 2 == 0:
            window += 1
        trend_input = torch.nn.functional.pad(
            log2_f0.view(1, 1, -1), (window // 2, window // 2), mode="replicate"
        )
        trend = torch.nn.functional.avg_pool1d(
            trend_input, kernel_size=window, stride=1
        ).view(-1)

        dev_st = (log2_f0 - trend) * 12.0
        dev_st = torch.where(dev_st >= 0, dev_st * accent_up, dev_st * accent_down)
        dev_st = torch.clamp(dev_st + bias_st, -max_dev_st, max_dev_st)
        shaped = torch.exp2(trend + dev_st / 12.0)

        # Keep voiced floor above very low dips to avoid breathy/raspy artifacts.
        median_hz = torch.exp2(torch.median(log2_f0))
        floor_hz = torch.clamp(
            median_hz * (0.48 + 0.12 * strength), min=80.0, max=320.0
        )
        shaped = torch.maximum(shaped, floor_hz)

        row[voiced] = shaped
        styled[b] = row

    return styled


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

        # Feature cache for chunk continuity (HuBERT/F0 boundary blending)
        self._feature_cache: Optional[torch.Tensor] = None  # [1, T_cache, C]
        self._f0_cache: Optional[torch.Tensor] = None  # [1, T_cache]
        self._f0_voiced_cache: Optional[torch.Tensor] = None  # [1, T_cache] bool
        # Cache lengths (frames). Can be tuned by realtime controller.
        self._feature_cache_frames: int = 20  # HuBERT frames @ 50fps
        self._f0_cache_frames: int = 40  # F0 frames @ 100fps

        # Phase 5: Audio-level overlap cache for F0/HuBERT extraction
        # Store the tail of each audio chunk to prepend to the next chunk
        # This allows F0/HuBERT to see continuous audio across boundaries
        self._audio_cache: Optional[np.ndarray] = None  # [T_audio] at 16kHz
        self._audio_cache_len: int = 3200  # 200ms at 16kHz (covers F0/HuBERT receptive fields)
        # Streaming history buffer for HuBERT context
        self._stream_history: Optional[np.ndarray] = None  # [T_audio] at 16kHz

        # Phase 8: Output overlap cache for smooth chunk boundaries
        # Store the tail of synthesizer output for crossfade with next chunk
        # This enables overlap-add blending at the audio output level
        self._output_cache: Optional[np.ndarray] = None  # [T_output] at model sample rate
        self._output_overlap_len: int = 0  # Set dynamically based on crossfade_sec

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
        self._feature_cache = None
        self._f0_cache = None
        self._f0_voiced_cache = None
        self._audio_cache = None
        self._output_cache = None
        self._stream_history = None
        # Streaming feature cache (used by infer_streaming to avoid synth padding)
        self._streaming_feat_cache = None
        # HuBERT audio context buffer (accumulates 16kHz audio for richer context)
        self._streaming_audio_history = None

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
        pad_mode: str = "chunk",
        synth_min_frames: int | None = MIN_SYNTH_FEATURE_FRAMES,
        history_sec: float = 0.0,
        noise_scale: float = 0.66666,
        pre_hubert_pitch_ratio: float = 0.0,
        moe_boost: float = 0.0,
        f0_lowpass_cutoff_hz: float = 16.0,
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
            pad_mode: Audio padding mode:
                - "chunk": per-chunk padding (default, legacy behavior)
                - "batch": batch-style padding (reflection at stream boundaries)
                - "none": no audio-level padding (use for chunked streaming)
            synth_min_frames: Minimum feature frames for synthesizer decoder.
                Set to 0 or None to disable short-input padding (test-only).
            history_sec: Prepend this many seconds of past audio for HuBERT context.
            pre_hubert_pitch_ratio: Ratio of pitch_shift to apply before
                HuBERT (0.0=disabled, 1.0=full). HuBERT/F0 are extracted
                from the same shifted audio for alignment, and only the
                residual shift is applied on F0.
            moe_boost: Moe voice style strength for F0 contour shaping
                (0.0=off, 1.0=strong).

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

        # Keep a copy of current chunk for history/output length tracking
        chunk_audio_np = audio_np
        chunk_length = len(chunk_audio_np)

        # Optional: prepend past audio for HuBERT context
        history_samples = int(max(0.0, history_sec) * 16000)
        history_len_used = 0
        if history_samples > 0:
            if self._stream_history is None:
                self._stream_history = np.zeros(0, dtype=chunk_audio_np.dtype)
            if len(self._stream_history) > 0:
                history_len_used = min(len(self._stream_history), history_samples)
                history_slice = self._stream_history[-history_len_used:]
                audio_np = np.concatenate([history_slice, chunk_audio_np])
            # Update history buffer with current chunk
            self._stream_history = np.concatenate([self._stream_history, chunk_audio_np])[-history_samples:]

        # Apply high-pass filter to remove DC offset and low-frequency noise
        # Original RVC uses 48Hz cutoff Butterworth filter
        audio_np = highpass_filter(audio_np, sr=16000, cutoff=48)

        # Add reflection padding for edge handling
        # Base padding: 50ms (800 samples @ 16kHz)
        # For short inputs, increase padding to ensure we get MIN_SYNTH_FEATURE_FRAMES
        # without needing feature-level padding (which causes length mismatch issues)
        input_length = len(audio_np)
        hubert_hop = 320
        f0_hop = 160  # F0 models use 160 sample hop (100fps at 16kHz)

        # Phase 5: Audio-level overlap for F0/HuBERT extraction continuity
        # Re-enabled: prepend cached audio and discard overlap frames from features/F0.
        audio_overlap_samples = 0
        hubert_overlap_frames = 0
        f0_overlap_frames = 0
        raw_audio_np = chunk_audio_np
        if history_samples <= 0 and use_feature_cache and self._audio_cache is not None and len(self._audio_cache) > 0:
            audio_overlap_samples = len(self._audio_cache)
            audio_np = np.concatenate([self._audio_cache, audio_np])
            hubert_overlap_frames = audio_overlap_samples // hubert_hop
            f0_overlap_frames = audio_overlap_samples // f0_hop
            logger.debug(
                f"Audio overlap: prepended {audio_overlap_samples} samples "
                f"({audio_overlap_samples/16:.1f}ms), "
                f"hubert_overlap_frames={hubert_overlap_frames}, "
                f"f0_overlap_frames={f0_overlap_frames}"
            )

        # Update audio cache with tail of current chunk (for future use)
        # Currently disabled but keeping cache updated for potential re-enabling
        if use_feature_cache:
            cache_len = min(self._audio_cache_len, chunk_length)
            self._audio_cache = raw_audio_np[-cache_len:].copy()

        pad_mode = pad_mode.lower()
        valid_pad_modes = {"chunk", "batch", "none"}
        if pad_mode not in valid_pad_modes:
            raise ValueError(f"Invalid pad_mode: {pad_mode} (valid: {sorted(valid_pad_modes)})")

        # Base padding: 50ms (800 samples) for batch processing
        # Chunk processing uses reduced padding (1 HuBERT hop) below
        base_pad = int(16000 * 0.05)  # 800 samples (50ms @ 16kHz)

        # For chunk processing, use minimal padding to avoid excessive padding artifacts
        # Optimal: 1 HuBERT hop (320 samples = 20ms) for batch/chunk consistency
        # This reduces padding accumulation at chunk boundaries while maintaining edge quality
        if pad_mode == "none":
            t_pad = 0
        elif pad_mode == "chunk" and allow_short_input:
            t_pad = hubert_hop  # 320 samples (20ms @ 16kHz, 1 HuBERT hop) - optimal
        else:
            # Calculate minimum input samples needed for MIN_SYNTH_FEATURE_FRAMES features
            # MIN_SYNTH_FEATURE_FRAMES is at 100fps, HuBERT produces 50fps, so /2
            # HuBERT produces approximately (samples / hop) - 1 frames due to its internal handling,
            # so we add 2 extra hops to ensure we get enough frames
            min_hubert_frames = MIN_SYNTH_FEATURE_FRAMES // 2  # 32 frames
            min_input_samples = (min_hubert_frames + 2) * hubert_hop  # 10880 samples (with buffer)

            # Check if we need extra padding to meet minimum
            total_with_base = base_pad + input_length + base_pad
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

        if pad_mode == "none":
            extra_pad = 0
        else:
            # Pad to multiple of HuBERT hop size (320) to avoid frame truncation
            padded_for_hubert = t_pad + input_length + t_pad
            remainder = padded_for_hubert % hubert_hop
            if remainder != 0:
                extra_pad = hubert_hop - remainder
            else:
                extra_pad = 0

        if t_pad > 0 or extra_pad > 0:
            audio_np = np.pad(audio_np, (t_pad, t_pad + extra_pad), mode="reflect")
        logger.info(
            f"Padding: chunk={chunk_length}, input={input_length}, "
            f"t_pad={t_pad}, extra_pad={extra_pad}, "
            f"final={len(audio_np)} (mode={pad_mode})"
        )

        audio = torch.from_numpy(audio_np).float()

        # Ensure 2D for batch processing
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        audio = audio.to(self.device)
        audio_for_hubert = audio
        effective_shift = compute_pre_hubert_shift(pitch_shift, pre_hubert_pitch_ratio)
        if abs(effective_shift) > 0.01:
            audio_for_hubert = pitch_shift_resample(
                audio, sample_rate=16000, semitones=effective_shift
            )
        f0_source_audio = audio_for_hubert if abs(effective_shift) > 0.01 else audio
        residual_f0_shift = compute_post_f0_shift(pitch_shift, effective_shift)
        moe_strength = max(0.0, min(MAX_MOE_BOOST, float(moe_boost)))

        # Debug: input audio stats
        logger.info(
            f"Input audio: shape={audio.shape} (chunk={chunk_length}, input={input_length}), min={audio.min():.4f}, max={audio.max():.4f}"
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
                        audio_for_hubert, output_layer=output_layer, output_dim=output_dim
                    )

            def extract_f0():
                if f0_method == "fcpe" and self.fcpe is not None:
                    with torch.autocast(device_type=self.device, dtype=self.dtype):
                        return self.fcpe.infer(f0_source_audio, threshold=0.006)
                elif self.rmvpe is not None:
                    with torch.autocast(device_type=self.device, dtype=self.dtype):
                        return self.rmvpe.infer(f0_source_audio)
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
                    audio_for_hubert, output_layer=output_layer, output_dim=output_dim
                )

        logger.info(
            f"HuBERT features: shape={features.shape}, min={features.min():.4f}, max={features.max():.4f}"
        )

        # Phase 5: Trim overlap frames from HuBERT features
        # These frames came from the prepended audio cache and should be discarded
        if hubert_overlap_frames > 0 and features.shape[1] > hubert_overlap_frames:
            features = features[:, hubert_overlap_frames:, :]
            logger.debug(
                f"HuBERT overlap trim: removed {hubert_overlap_frames} frames, new shape={features.shape}"
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

        # Feature cache blending for chunk continuity (50fps HuBERT features)
        # Phase 2 improvement: Adaptive blending based on cosine similarity
        if use_feature_cache and self._feature_cache is not None:
            max_cache_len = max(1, int(self._feature_cache_frames))
            cache_avail = min(max_cache_len, self._feature_cache.shape[1], features.shape[1])

            if cache_avail > 0:
                prev_tail = self._feature_cache[:, -cache_avail:, :]
                curr_head = features[:, :cache_avail, :].clone()

                # Calculate cosine similarity at boundary
                prev_last = prev_tail[:, -1:, :]  # [B, 1, C]
                curr_first = curr_head[:, :1, :]  # [B, 1, C]
                cos_sim = torch.nn.functional.cosine_similarity(
                    prev_last.squeeze(1), curr_first.squeeze(1), dim=-1
                )  # [B]

                # Adaptive blend length: lower similarity = longer blend
                # similarity 0.9+ -> 5 frames, similarity 0.5 -> 15 frames
                adaptive_blend = int((1.0 - cos_sim.mean().item()) * max_cache_len) + 5
                blend_len = min(adaptive_blend, cache_avail)

                if blend_len > 0:
                    # Sigmoid blending for smoother transitions
                    alpha_np = sigmoid_blend_weights(blend_len, steepness=4.0)
                    alpha = torch.from_numpy(alpha_np).to(
                        device=features.device, dtype=features.dtype
                    ).view(1, -1, 1)

                    # Blend features
                    blended = prev_tail[:, -blend_len:, :] * alpha + curr_head[:, :blend_len, :] * (1.0 - alpha)

                    # Preserve original norm (feature magnitude)
                    orig_norms = torch.norm(curr_head[:, :blend_len, :], dim=-1, keepdim=True)
                    blended_norms = torch.norm(blended, dim=-1, keepdim=True) + 1e-8
                    blended = blended * (orig_norms / blended_norms)

                    features[:, :blend_len, :] = blended

                    logger.debug(
                        f"HuBERT adaptive blend: cos_sim={cos_sim.mean().item():.3f}, blend_len={blend_len}"
                    )

        # Update HuBERT feature cache (store tail for next chunk)
        if use_feature_cache:
            cache_len = max(1, int(self._feature_cache_frames))
            if features.shape[1] > 0:
                self._feature_cache = features[:, -cache_len:, :].detach()

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
                        f0 = self.fcpe.infer(f0_source_audio, threshold=0.006)
                    logger.debug("F0 extracted with FCPE (sequential)")

                # Use RMVPE if requested and available (or fallback if FCPE failed)
                elif self.rmvpe is not None and (f0_method == "rmvpe" or f0 is None):
                    with torch.autocast(device_type=self.device, dtype=self.dtype):
                        f0 = self.rmvpe.infer(f0_source_audio)
                    logger.debug("F0 extracted with RMVPE (sequential)")
            else:
                logger.debug(f"F0 from parallel extraction ({f0_method})")

            # Phase 5: Trim overlap frames from F0
            # These frames came from the prepended audio cache and should be discarded
            if f0 is not None and f0_overlap_frames > 0 and f0.shape[1] > f0_overlap_frames:
                f0 = f0[:, f0_overlap_frames:]
                logger.debug(
                    f"F0 overlap trim: removed {f0_overlap_frames} frames, new shape={f0.shape}"
                )

            if f0 is not None and f0.numel() > 0:  # Check for non-empty F0
                # FCPE smoothing for stability (reduce frame-to-frame jitter)
                if f0_method == "fcpe":
                    # Light smoothing to reduce jitter without adding artifacts
                    f0 = _smooth_fcpe_f0(f0)

                # Apply pitch shift (only to voiced regions where f0 > 0)
                if abs(residual_f0_shift) > 0.01:
                    f0 = torch.where(
                        f0 > 0, f0 * (2 ** (residual_f0_shift / 12)), f0
                    )
                if moe_strength > 0.0:
                    f0 = apply_moe_f0_style(f0, moe_strength)

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

                # F0 cache blending for chunk continuity (100fps F0)
                # Phase 1 improvement: Extended cache, sigmoid blending, jump detection
                if use_feature_cache and self._f0_cache is not None:
                    cache_len = max(1, int(self._f0_cache_frames))
                    blend_len = min(cache_len, self._f0_cache.shape[1], f0.shape[1])
                    if blend_len > 0:
                        prev_tail = self._f0_cache[:, -blend_len:]
                        prev_voiced = (
                            self._f0_voiced_cache[:, -blend_len:]
                            if self._f0_voiced_cache is not None
                            else prev_tail > 0
                        )
                        cur_head = f0[:, :blend_len].clone()
                        cur_voiced = cur_head > 0
                        blend_mask = prev_voiced & cur_voiced

                        # Sigmoid blending for smoother transitions
                        alpha_np = sigmoid_blend_weights(blend_len, steepness=4.0)
                        alpha = torch.from_numpy(alpha_np).to(
                            device=f0.device, dtype=f0.dtype
                        ).view(1, -1)

                        # Detect large jumps and apply linear interpolation
                        # Check the boundary between cache and current
                        prev_last_f0 = prev_tail[:, -1]
                        cur_first_f0 = cur_head[:, 0]
                        both_voiced = (prev_last_f0 > 0) & (cur_first_f0 > 0)

                        if both_voiced.any():
                            jump = torch.abs(prev_last_f0 - cur_first_f0)
                            # If jump > 50Hz, apply linear interpolation at boundary
                            if jump.item() > 50:
                                # Linear interpolation for first few frames
                                interp_len = min(10, blend_len)
                                interp_weights = torch.linspace(
                                    0.0, 1.0, interp_len, device=f0.device, dtype=f0.dtype
                                ).view(1, -1)
                                # Interpolate from prev_last_f0 to values further in cur_head
                                target_f0 = cur_head[:, interp_len - 1 : interp_len]
                                if target_f0.numel() > 0:
                                    interp_f0 = prev_last_f0.view(-1, 1) * (1 - interp_weights) + \
                                                target_f0 * interp_weights
                                    # Apply interpolation only where both are voiced
                                    for i in range(interp_len):
                                        if cur_head[:, i].item() > 0:
                                            cur_head[:, i] = interp_f0[:, i]
                                logger.debug(
                                    f"F0 jump {jump.item():.1f}Hz detected, applied linear interpolation"
                                )

                        # Apply sigmoid blending
                        blended = prev_tail * alpha + cur_head * (1.0 - alpha)
                        f0[:, :blend_len] = torch.where(blend_mask, blended, f0[:, :blend_len])

                # Apply median filter to remove F0 spikes
                f0 = smooth_f0_spikes(f0, window=3)

                # Phase 6: Apply lowpass filter to remove high-frequency jitter
                f0 = lowpass_f0(f0, cutoff_hz=f0_lowpass_cutoff_hz, sample_rate=100.0)

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

                # Update F0 cache (store tail for next chunk)
                # Extended cache length for better boundary blending
                if use_feature_cache:
                    cache_len = max(1, int(self._f0_cache_frames))
                    self._f0_cache = f0[:, -cache_len:].detach()
                    self._f0_voiced_cache = (f0 > 0)[:, -cache_len:].detach()
            elif f0 is not None and f0.numel() == 0:
                # F0 extraction returned empty array (input too short)
                logger.warning(
                    f"F0 extraction returned empty array (input too short: {len(audio)/16000:.3f}s). "
                    f"Minimum recommended: FCPE=0.10s, RMVPE=0.32s. Falling back to unvoiced."
                )
                f0 = None  # Fall through to unvoiced handling

            if f0 is None:
                # No F0 available - use pitch=1 (unvoiced marker per RVC convention)
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
        min_synth_frames = (
            MIN_SYNTH_FEATURE_FRAMES if synth_min_frames is None else synth_min_frames
        )
        if min_synth_frames > 0 and features.shape[1] < min_synth_frames:
            synth_pad_frames = min_synth_frames - features.shape[1]
            pad_left = synth_pad_frames // 2
            pad_right = synth_pad_frames - pad_left

            # Choose padding mode based on input size
            # reflect mode requires padding < input size, use replicate for very short inputs
            current_size = features.shape[1]
            if max(pad_left, pad_right) >= current_size:
                pad_mode = "replicate"  # Edge replication for very short inputs
                logger.warning(
                    f"Input too short ({current_size} frames) for reflection padding ({pad_left}+{pad_right}). "
                    f"Using replicate mode. Consider increasing chunk_sec to >= 0.15s"
                )
            else:
                pad_mode = "reflect"  # Preferred for normal inputs

            features = torch.nn.functional.pad(
                features.permute(0, 2, 1),  # [B, T, C] -> [B, C, T]
                (pad_left, pad_right),
                mode=pad_mode,
            ).permute(0, 2, 1)  # [B, C, T] -> [B, T, C]
            # Update feature lengths
            feature_lengths = torch.tensor(
                [features.shape[1]], dtype=torch.long, device=self.device
            )
            # Pad pitch/pitchf if present
            if pitch is not None:
                pitch = torch.nn.functional.pad(pitch, (pad_left, pad_right), mode=pad_mode)
            if pitchf is not None:
                pitchf = torch.nn.functional.pad(pitchf, (pad_left, pad_right), mode=pad_mode)
            logger.info(
                f"Padded short input: {features.shape[1] - synth_pad_frames} -> {features.shape[1]} frames (min={min_synth_frames})"
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
                noise_scale=noise_scale,
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
        # HuBERT produces (input / 320) - 1 frames, so synthesizer output
        # is 1 HuBERT frame shorter than naive expectation.  Reduce trim_end
        # to compensate (same fix as infer_streaming).
        hubert_deficit = 2 * (self.sample_rate // 100)
        trim_start = t_pad_tgt
        trim_end = max(0, t_pad_tgt + extra_pad_tgt - hubert_deficit)

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

        if history_len_used > 0:
            history_output_samples = int(history_len_used * self.sample_rate / 16000)
            if len(output) > history_output_samples:
                output = output[history_output_samples:]
                logger.info(
                    f"Trimmed history from start: {history_output_samples} samples (history_sec={history_sec:.3f})"
                )

        # Note: HuBERT frame quantization causes output to be slightly shorter than ideally expected.
        # We intentionally do NOT pad/extend to match expected length, as artificial waveform
        # repetition creates artifacts at chunk boundaries. Instead:
        # - Accept the actual output length (crossfade handles variable-length chunks)
        # - Only trim if output is too long (rare edge case)
        expected_output_samples = int(chunk_length * self.sample_rate / 16000)
        length_diff = len(output) - expected_output_samples
        logger.info(
            f"Length check: got {len(output)}, expected {expected_output_samples}, diff={length_diff}"
        )

        if length_diff > 0:
            # Output too long - trim from end
            output = output[:expected_output_samples]
            logger.debug(f"Trimmed {length_diff} extra samples from end")
        elif length_diff < 0 and abs(length_diff) > 100:
            if allow_short_input and pad_mode == "chunk":
                # Output too short (chunk processing) - pad zeros to expected length
                # This prevents cumulative drift across chunks while avoiding resampling artifacts.
                output = np.pad(output, (0, -length_diff))
                logger.debug(
                    f"Padded output from {expected_output_samples + length_diff} to {expected_output_samples} samples"
                )
            else:
                # Output too short - resample to stretch to expected length
                # Used for batch, and for streaming with pad_mode="none" to avoid zero-padding artifacts.
                output = resample(output, len(output), expected_output_samples)
                logger.debug(
                    f"Resampled output from {expected_output_samples + length_diff} to {expected_output_samples} samples"
                )

        logger.info(
            f"Final output: shape={output.shape}, min={output.min():.4f}, max={output.max():.4f}"
        )
        return output

    @torch.no_grad()
    def infer_streaming(
        self,
        audio_16k: np.ndarray,
        overlap_samples: int,
        pitch_shift: int = 0,
        f0_method: str = "fcpe",
        index_rate: float = 0.0,
        index_k: int = 4,
        voice_gate_mode: str = "expand",
        energy_threshold: float = 0.05,
        use_parallel_extraction: bool = True,
        noise_scale: float = 0.66666,
        sola_extra_samples: int = 0,
        pre_hubert_pitch_ratio: float = 0.0,
        moe_boost: float = 0.0,
        f0_lowpass_cutoff_hz: float = 16.0,
    ) -> np.ndarray:
        """Streaming inference with audio-level overlap.

        Processes [overlap | new_hop] audio through the full pipeline
        (HuBERT -> synthesizer), then trims the synthesizer OUTPUT to
        keep only the new_hop portion (plus optional SOLA extra).

        Args:
            audio_16k: Input audio at 16kHz, shape [overlap + new_hop].
                       Length MUST be a multiple of 320 (HuBERT hop).
                       overlap_samples MUST also be a multiple of 320.
            overlap_samples: Number of overlap samples from previous chunk.
            pitch_shift: Pitch shift in semitones.
            f0_method: F0 extraction method ("fcpe", "rmvpe", "none").
            index_rate: FAISS index blending ratio (0=off).
            index_k: Number of FAISS neighbors.
            voice_gate_mode: Voice gate mode (off/strict/expand/energy).
            energy_threshold: Energy threshold for "energy" gate mode.
            use_parallel_extraction: Enable parallel HuBERT+F0 extraction.
            sola_extra_samples: Extra samples to keep from overlap region
                (at model sample rate) to compensate for SOLA crossfade
                deficit. When SOLA is enabled, consecutive outputs overlap
                by this amount so SOLA can crossfade without losing samples.
            pre_hubert_pitch_ratio: Ratio of pitch_shift to apply before
                HuBERT (0.0=disabled, 1.0=full). HuBERT/F0 are extracted
                from the same shifted audio for alignment, and only the
                residual shift is applied on F0.
            moe_boost: Moe voice style strength for F0 contour shaping
                (0.0=off, 1.0=strong).

        Returns:
            Converted audio at model sample rate (usually 40kHz).
            Output length = new_hop + sola_extra_samples.
        """
        if not self._loaded:
            self.load()

        hubert_hop = 320

        # Validate alignment
        total_samples = len(audio_16k)
        assert total_samples % hubert_hop == 0, (
            f"audio_16k length {total_samples} is not a multiple of {hubert_hop}"
        )
        assert overlap_samples % hubert_hop == 0, (
            f"overlap_samples {overlap_samples} is not a multiple of {hubert_hop}"
        )

        new_samples = total_samples - overlap_samples
        assert new_samples > 0, "No new audio samples after overlap"

        # Expected output length (at model sample rate)
        # Include sola_extra so consecutive outputs overlap by that amount,
        # allowing SOLA crossfade to consume those samples without deficit.
        expected_output = int(new_samples * self.sample_rate / 16000) + sola_extra_samples

        # --- HuBERT audio context buffer ---
        # Accumulate real 16kHz audio so HuBERT processes a larger context
        # window each chunk.  Up to MAX_HUBERT_CONTEXT_16K (2.0s) of audio
        # is kept, producing coherent features from a single forward pass.
        # F0 is extracted on the current chunk only (pitch is local).
        t_pad = 800

        new_hop_16k = audio_16k[overlap_samples:]

        # Minimum 16kHz audio (before padding) to produce MIN_SYNTH_FEATURE_FRAMES
        # from a single HuBERT forward pass.
        # HuBERT: (input + 2*t_pad) / 320 - 1 frames @ 50fps, interpolated 2x → 100fps
        # 64 features → 32 HuBERT frames → 33*320 - 2*t_pad = 8960 samples
        min_audio_for_full_features = (
            (MIN_SYNTH_FEATURE_FRAMES // 2 + 1) * hubert_hop - 2 * t_pad
        )

        # --- Audio history for HuBERT (capped for fast synthesis) ---
        if self._streaming_audio_history is None:
            self._streaming_audio_history = audio_16k.copy()
        else:
            self._streaming_audio_history = np.concatenate([
                self._streaming_audio_history, new_hop_16k
            ])
        if len(self._streaming_audio_history) > min_audio_for_full_features:
            self._streaming_audio_history = (
                self._streaming_audio_history[-min_audio_for_full_features:]
            )

        # Extend audio_16k with HuBERT history (capped ~8960 for synthesis).
        pre_context_samples = 0
        if len(self._streaming_audio_history) > total_samples:
            pre_context_samples = len(self._streaming_audio_history) - total_samples
            # Treat pre-context as additional overlap (trimmed from output)
            audio_16k = self._streaming_audio_history.copy()
            total_samples = len(audio_16k)
            overlap_samples = overlap_samples + pre_context_samples

        # Apply high-pass filter
        audio_filtered = highpass_filter(audio_16k, sr=16000, cutoff=48)

        # Reflection padding for edge handling (fixed 50ms like batch infer)
        input_length = len(audio_filtered)

        # Ensure padded audio is multiple of hubert_hop
        padded_for_hubert = t_pad + input_length + t_pad
        remainder = padded_for_hubert % hubert_hop
        extra_pad = (hubert_hop - remainder) if remainder != 0 else 0

        audio_padded = np.pad(
            audio_filtered, (t_pad, t_pad + extra_pad), mode="reflect"
        )

        # Pad HuBERT input to a fixed size
        # As the audio history grows, the input size changes every chunk,
        # triggering expensive kernel compilations on Intel XPU.  By padding
        # to the maximum expected size, we get a single compilation on the
        # first chunk and stable performance thereafter.  The extra synthesis
        # output from the end padding is exactly consumed by the increased
        # trim_right, so net output length is unchanged.
        fixed_hubert_input = (
            (min_audio_for_full_features + 2 * t_pad + hubert_hop - 1)
            // hubert_hop * hubert_hop
        )
        if len(audio_padded) < fixed_hubert_input:
            end_pad = fixed_hubert_input - len(audio_padded)
            audio_padded = np.pad(audio_padded, (0, end_pad), mode="reflect")
            extra_pad += end_pad

        # Output-level trim amounts (at model sample rate)
        t_pad_tgt = int(t_pad * self.sample_rate / 16000)
        extra_pad_tgt = int(extra_pad * self.sample_rate / 16000)
        overlap_tgt = int(overlap_samples * self.sample_rate / 16000)
        # HuBERT produces (input / 320) - 1 frames, so the synthesizer output
        # is 1 HuBERT frame shorter than the naive expectation.  In output
        # space this is 2 feature-frames * samples_per_frame.
        hubert_deficit = 2 * (self.sample_rate // 100)
        # Left trim: padding + overlap (minus sola_extra to keep extra overlap
        # in the output for SOLA crossfade compensation).
        # Right trim: padding + extra - deficit.
        trim_left = max(0, t_pad_tgt + overlap_tgt - sola_extra_samples)
        trim_right = max(0, t_pad_tgt + extra_pad_tgt - hubert_deficit)

        # Convert to tensors
        audio_t = (
            torch.from_numpy(audio_padded).float().unsqueeze(0).to(self.device)
        )
        audio_t_for_hubert = audio_t
        effective_shift = compute_pre_hubert_shift(pitch_shift, pre_hubert_pitch_ratio)
        if abs(effective_shift) > 0.01:
            audio_t_for_hubert = pitch_shift_resample(
                audio_t, sample_rate=16000, semitones=effective_shift
            )
        f0_source_audio = audio_t_for_hubert if abs(effective_shift) > 0.01 else audio_t
        residual_f0_shift = compute_post_f0_shift(pitch_shift, effective_shift)
        moe_strength = max(0.0, min(MAX_MOE_BOOST, float(moe_boost)))
        # Determine HuBERT output params
        if self.synthesizer.version == 1:
            output_dim = 256
            output_layer = 9
        else:
            output_dim = 768
            output_layer = 12

        # Parallel HuBERT + F0 extraction
        # HuBERT/F0 use the same source audio to preserve alignment.
        features = None
        f0_raw = None
        use_f0 = self.has_f0 and f0_method != "none"

        if use_parallel_extraction and use_f0:
            def extract_hubert():
                with torch.autocast(device_type=self.device, dtype=self.dtype):
                    return self.hubert.extract(
                        audio_t_for_hubert, output_layer=output_layer, output_dim=output_dim
                    )

            def extract_f0():
                if f0_method == "fcpe" and self.fcpe is not None:
                    with torch.autocast(device_type=self.device, dtype=self.dtype):
                        return self.fcpe.infer(f0_source_audio, threshold=0.006)
                elif self.rmvpe is not None:
                    with torch.autocast(device_type=self.device, dtype=self.dtype):
                        return self.rmvpe.infer(f0_source_audio)
                logger.warning(
                    "[INFER] F0 parallel extraction returned None "
                    f"(method={f0_method}, fcpe={'loaded' if self.fcpe else 'None'}, "
                    f"rmvpe={'loaded' if self.rmvpe else 'None'})"
                )
                return None

            with ThreadPoolExecutor(max_workers=2) as executor:
                hubert_future = executor.submit(extract_hubert)
                f0_future = executor.submit(extract_f0)
                features = hubert_future.result()
                f0_raw = f0_future.result()
        else:
            with torch.autocast(device_type=self.device, dtype=self.dtype):
                features = self.hubert.extract(
                    audio_t_for_hubert, output_layer=output_layer, output_dim=output_dim
                )

        # FAISS index retrieval
        if index_rate > 0 and self.faiss_index is not None:
            features = self._search_index(features, index_rate, k=index_k)

        # Interpolate features 2x (50fps -> 100fps for synthesizer)
        features = torch.nn.functional.interpolate(
            features.permute(0, 2, 1),
            scale_factor=2,
            mode="linear",
            align_corners=False,
        ).permute(0, 2, 1)

        feature_lengths = torch.tensor(
            [features.shape[1]], dtype=torch.long, device=self.device
        )

        # F0 processing (same audio_t as HuBERT for perfect frame alignment)
        pitch = None
        pitchf = None
        voiced_mask_for_gate = None

        if use_f0:
            f0 = f0_raw
            if f0 is None:
                logger.debug(
                    "[INFER] F0 parallel result was None, trying sequential fallback "
                    f"(method={f0_method})"
                )
                if f0_method == "fcpe" and self.fcpe is not None:
                    with torch.autocast(device_type=self.device, dtype=self.dtype):
                        f0 = self.fcpe.infer(f0_source_audio, threshold=0.006)
                elif self.rmvpe is not None:
                    with torch.autocast(device_type=self.device, dtype=self.dtype):
                        f0 = self.rmvpe.infer(f0_source_audio)

            if f0 is not None and f0.numel() > 0:
                # FCPE smoothing
                if f0_method == "fcpe":
                    f0 = _smooth_fcpe_f0(f0)

                # Pitch shift
                if abs(residual_f0_shift) > 0.01:
                    f0 = torch.where(
                        f0 > 0, f0 * (2 ** (residual_f0_shift / 12)), f0
                    )
                if moe_strength > 0.0:
                    f0 = apply_moe_f0_style(f0, moe_strength)

                # Align F0 to feature length (F0 from same audio_t, minor hop mismatch)
                if f0.shape[1] != features.shape[1]:
                    f0 = torch.nn.functional.interpolate(
                        f0.unsqueeze(1),
                        size=features.shape[1],
                        mode="linear",
                        align_corners=False,
                    ).squeeze(1)

                # Median filter for spike removal
                f0 = smooth_f0_spikes(f0, window=3)

                # Lowpass filter for jitter removal
                f0 = lowpass_f0(f0, cutoff_hz=f0_lowpass_cutoff_hz, sample_rate=100.0)

                pitchf = f0.to(self.dtype)

                # Quantized pitch for embedding
                f0_mel_min = 1127 * math.log(1 + 50 / 700)
                f0_mel_max = 1127 * math.log(1 + 1100 / 700)
                f0_mel = 1127 * torch.log(1 + f0 / 700)
                voiced_mask = f0_mel > 0
                f0_mel_normalized = torch.where(
                    voiced_mask,
                    (f0_mel - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1,
                    f0_mel,
                )
                pitch = torch.clamp(f0_mel_normalized, 1, 255).round().long()
                voiced_mask_for_gate = voiced_mask.float()

            if pitch is None:
                logger.warning(
                    "[INFER] F0 extraction failed — using flat pitch "
                    f"(method={f0_method}, f0_raw={'OK' if f0_raw is not None else 'None'}, "
                    f"audio_samples={audio_t.shape[-1]})"
                )
                pitch = torch.ones(
                    features.shape[0], features.shape[1],
                    dtype=torch.long, device=self.device,
                )
                pitchf = torch.zeros(
                    features.shape[0], features.shape[1],
                    dtype=self.dtype, device=self.device,
                )

        # Fallback: F0 model but f0_method="none" (e.g. overload protection)
        if self.has_f0 and pitch is None:
            logger.warning(
                "[INFER] F0 skipped — flat pitch fallback "
                f"(f0_method={f0_method}, use_f0={use_f0})"
            )
            pitch = torch.ones(
                features.shape[0], features.shape[1],
                dtype=torch.long, device=self.device,
            )
            pitchf = torch.zeros(
                features.shape[0], features.shape[1],
                dtype=self.dtype, device=self.device,
            )

        # --- Feature cache: use real features from previous chunk instead
        # of reflect-padding when the current chunk is too short for the
        # synthesizer decoder.  This gives the decoder real context from
        # the preceding audio, dramatically improving output quality for
        # low-latency chunk sizes.
        cache_prepend_frames = 0
        new_features_for_cache = features.clone()
        new_pitch_for_cache = pitch.clone() if pitch is not None else None
        new_pitchf_for_cache = pitchf.clone() if pitchf is not None else None

        if (
            self._streaming_feat_cache is not None
            and features.shape[1] < MIN_SYNTH_FEATURE_FRAMES
        ):
            c_feat, c_pitch, c_pitchf = self._streaming_feat_cache
            need = MIN_SYNTH_FEATURE_FRAMES - features.shape[1]
            avail = c_feat.shape[1]
            cache_prepend_frames = min(need, avail)

            features = torch.cat(
                [c_feat[:, -cache_prepend_frames:], features], dim=1
            )
            feature_lengths = torch.tensor(
                [features.shape[1]], dtype=torch.long, device=self.device,
            )
            if pitch is not None and c_pitch is not None:
                pitch = torch.cat(
                    [c_pitch[:, -cache_prepend_frames:], pitch], dim=1
                )
            if pitchf is not None and c_pitchf is not None:
                pitchf = torch.cat(
                    [c_pitchf[:, -cache_prepend_frames:], pitchf], dim=1
                )
            # Extend voice gate mask to cover cached prefix (all-voiced,
            # since the cached portion is trimmed from the output anyway).
            if voiced_mask_for_gate is not None:
                cache_mask = torch.ones(
                    1, cache_prepend_frames,
                    dtype=voiced_mask_for_gate.dtype,
                    device=voiced_mask_for_gate.device,
                )
                voiced_mask_for_gate = torch.cat(
                    [cache_mask, voiced_mask_for_gate], dim=1
                )
            trim_left += cache_prepend_frames * (self.sample_rate // 100)

        # Fallback: reflect-pad if still too short (first chunk, no cache)
        synth_pad_frames = 0
        if features.shape[1] < MIN_SYNTH_FEATURE_FRAMES:
            synth_pad_frames = MIN_SYNTH_FEATURE_FRAMES - features.shape[1]
            pad_left = synth_pad_frames // 2
            pad_right = synth_pad_frames - pad_left

            current_size = features.shape[1]
            pad_mode = "replicate" if max(pad_left, pad_right) >= current_size else "reflect"

            features = torch.nn.functional.pad(
                features.permute(0, 2, 1), (pad_left, pad_right), mode=pad_mode,
            ).permute(0, 2, 1)
            feature_lengths = torch.tensor(
                [features.shape[1]], dtype=torch.long, device=self.device,
            )
            if pitch is not None:
                pitch = torch.nn.functional.pad(pitch, (pad_left, pad_right), mode=pad_mode)
            if pitchf is not None:
                pitchf = torch.nn.functional.pad(pitchf, (pad_left, pad_right), mode=pad_mode)

        # Save current chunk's features for next call's cache
        self._streaming_feat_cache = (
            new_features_for_cache,
            new_pitch_for_cache,
            new_pitchf_for_cache,
        )

        # --- Compute skip_head/return_length for streaming synthesis ---
        # TextEncoder processes ALL features for rich context, but Flow +
        # Decoder only synthesize the output region (new_hop + sola_extra).
        # This prevents SineGen phase accumulation through varying context,
        # which causes "a-na-na-" artifacts at chunk boundaries.
        samples_per_frame = self.sample_rate // 100

        # Total left context to skip (in model_sr samples)
        # trim_left already includes: t_pad_tgt + overlap_tgt - sola_extra + cache_prepend
        total_left_samples = trim_left
        if synth_pad_frames > 0:
            total_left_samples += (synth_pad_frames // 2) * samples_per_frame

        # Total right padding to skip (in model_sr samples)
        total_right_samples = trim_right
        if synth_pad_frames > 0:
            total_right_samples += (synth_pad_frames - synth_pad_frames // 2) * samples_per_frame

        skip_head_feat = total_left_samples // samples_per_frame
        trim_right_feat = total_right_samples // samples_per_frame

        # Clamp to valid range
        skip_head_feat = min(skip_head_feat, features.shape[1] - 1)
        return_length_feat = max(
            1, features.shape[1] - skip_head_feat - trim_right_feat
        )

        # Residual samples not covered by feature-level skip (sub-frame trim)
        residual_left = total_left_samples - skip_head_feat * samples_per_frame

        # Run synthesizer with skip_head/return_length
        with torch.autocast(device_type=self.device, dtype=self.dtype):
            output = self.synthesizer.infer(
                features, feature_lengths, pitch=pitch, pitchf=pitchf,
                noise_scale=noise_scale,
                skip_head=skip_head_feat,
                return_length=return_length_feat,
                return_length2=return_length_feat,
            )

        # Voice gating (output is already the target region)
        if voice_gate_mode != "off" and voiced_mask_for_gate is not None:
            # Slice mask to match the synthesized output region
            gate_mask_src = voiced_mask_for_gate[
                :, skip_head_feat:skip_head_feat + return_length_feat
            ]
            output_len = output.shape[-1]
            gate_mask = gate_mask_src.clone()

            if voice_gate_mode == "expand":
                expand_frames = 2
                gate_mask = torch.nn.functional.max_pool1d(
                    gate_mask.unsqueeze(1),
                    kernel_size=expand_frames * 2 + 1,
                    stride=1,
                    padding=expand_frames,
                ).squeeze(1)
            elif voice_gate_mode == "energy":
                frame_size = output_len // gate_mask.shape[-1]
                if frame_size > 0:
                    output_frames = output.unfold(-1, frame_size, frame_size)
                    frame_energy = (output_frames ** 2).mean(dim=-1)
                    energy_max = frame_energy.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
                    energy_mask = (frame_energy / energy_max > energy_threshold).float()
                    if energy_mask.shape[-1] == gate_mask.shape[-1]:
                        gate_mask = torch.maximum(gate_mask, energy_mask)

            gate_mask = torch.nn.functional.interpolate(
                gate_mask.unsqueeze(1), size=output_len,
                mode="linear", align_corners=False,
            ).squeeze(1)

            smooth_samples = int(self.sample_rate * 0.005)
            if smooth_samples > 1:
                kernel = torch.ones(1, 1, smooth_samples, device=gate_mask.device) / smooth_samples
                gate_mask = torch.nn.functional.conv1d(
                    gate_mask.unsqueeze(1), kernel, padding=smooth_samples // 2,
                ).squeeze(1)
                if gate_mask.shape[-1] != output_len:
                    gate_mask = gate_mask[..., :output_len]
                gate_mask = torch.clamp(gate_mask, 0, 1)

            output = output * gate_mask

        # Convert to numpy
        output = output.cpu().float().numpy()
        if output.ndim == 2:
            output = output[0]

        # Trim sub-frame residual from left (skip_head rounds to feature boundaries)
        if residual_left > 0 and len(output) > residual_left:
            output = output[residual_left:]

        # Adjust to exact expected length
        length_diff = len(output) - expected_output
        if length_diff > 0:
            output = output[:expected_output]
        elif length_diff < 0 and abs(length_diff) > 100:
            output = resample(output, len(output), expected_output)

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
