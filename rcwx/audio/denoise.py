"""Real-time noise reduction.

Supports multiple backends:
- Facebook Denoiser: ML-based, real-time capable, preserves human voice
- Spectral Gate: Traditional DSP, lower latency fallback

Facebook Denoiser is a PyTorch-based speech enhancement model that
removes background noise while preserving human voice.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class DenoiseConfig:
    """Configuration for noise reduction."""

    # FFT parameters
    n_fft: int = 2048
    hop_length: int = 512
    win_length: Optional[int] = None  # Defaults to n_fft

    # Noise estimation
    noise_frames: int = 10  # Number of frames to estimate noise
    noise_floor: float = 0.001  # Minimum noise floor

    # Spectral gate parameters
    threshold_db: float = -20.0  # Gate threshold relative to noise (dB)
    reduction_db: float = -30.0  # Amount of noise reduction (dB)
    attack_time: float = 0.01  # Attack time in seconds
    release_time: float = 0.05  # Release time in seconds

    # Smoothing
    freq_smoothing: int = 3  # Frequency smoothing window size

    def __post_init__(self):
        if self.win_length is None:
            self.win_length = self.n_fft


# =============================================================================
# Facebook Denoiser (ML-based, recommended)
# =============================================================================

class MLDenoiser:
    """ML-based noise suppression using Facebook's Denoiser.

    Facebook Denoiser is trained to preserve human voice while removing
    background noise - similar to modern microphone noise cancellation.

    Supports CPU, CUDA, and XPU. Falls back to CPU if the requested
    device fails during model loading.

    Requires: pip install denoiser

    Usage:
        denoiser = MLDenoiser(device="xpu")
        clean_audio = denoiser.process(noisy_audio, sample_rate=16000)
    """

    def __init__(self, device: str = "cpu"):
        """Initialize Facebook Denoiser model.

        Args:
            device: Requested device ("cpu", "cuda", "xpu", "auto")
                    Falls back to CPU if the requested device fails.
        """
        self._requested_device = device
        self._actual_device = device if device != "auto" else "cpu"
        self._model = None
        self._loaded = False

    def _load_model(self):
        """Lazy-load the Denoiser model."""
        if self._loaded:
            return

        try:
            from denoiser import pretrained
            import torch

            # Load pre-trained DNS64 model (best quality)
            self._model = pretrained.dns64()
            self._model.eval()

            # Move to requested device (XPU, CUDA, etc.) with CPU fallback
            if self._actual_device != "cpu":
                try:
                    self._model = self._model.to(self._actual_device)
                    # Verify with a small test inference
                    test_input = torch.zeros(1, 1, 16000, device=self._actual_device)
                    with torch.no_grad():
                        self._model(test_input)
                    logger.info(f"Facebook Denoiser model loaded on {self._actual_device}")
                except Exception as e:
                    logger.warning(f"Could not run model on {self._actual_device}, using CPU: {e}")
                    self._actual_device = "cpu"
                    self._model = self._model.cpu()
            else:
                logger.info("Facebook Denoiser model loaded on CPU")

            self._loaded = True

        except ImportError:
            raise ImportError(
                "Facebook Denoiser not installed. Install with: pip install denoiser"
            )

    def process(
        self,
        audio: NDArray[np.float32],
        sample_rate: int = 16000,
    ) -> NDArray[np.float32]:
        """Process audio through Facebook Denoiser.

        Args:
            audio: Input audio (mono, float32, -1 to 1 range)
            sample_rate: Input sample rate

        Returns:
            Denoised audio at original sample rate
        """
        self._load_model()

        import torch

        # Denoiser expects 16kHz
        target_sr = 16000

        # Resample if needed
        if sample_rate != target_sr:
            from scipy import signal
            gcd = np.gcd(sample_rate, target_sr)
            up = target_sr // gcd
            down = sample_rate // gcd
            audio_16k = signal.resample_poly(audio, up, down).astype(np.float32)
        else:
            audio_16k = audio

        # Convert to tensor [batch, channels, samples]
        audio_tensor = torch.from_numpy(audio_16k).unsqueeze(0).unsqueeze(0)

        if self._actual_device != "cpu":
            audio_tensor = audio_tensor.to(self._actual_device)

        # Enhance
        with torch.no_grad():
            enhanced = self._model(audio_tensor)

        # Back to numpy
        enhanced_np = enhanced.squeeze().cpu().numpy().astype(np.float32)

        # Resample back to original rate
        if sample_rate != target_sr:
            from scipy import signal
            gcd = np.gcd(target_sr, sample_rate)
            up = sample_rate // gcd
            down = target_sr // gcd
            enhanced_np = signal.resample_poly(enhanced_np, up, down).astype(np.float32)

        # Match original length
        if len(enhanced_np) > len(audio):
            enhanced_np = enhanced_np[:len(audio)]
        elif len(enhanced_np) < len(audio):
            enhanced_np = np.pad(enhanced_np, (0, len(audio) - len(enhanced_np)))

        return enhanced_np


# =============================================================================
# Spectral Gate Denoiser (Traditional DSP fallback)
# =============================================================================

class SpectralGateDenoiser:
    """Real-time spectral gate noise reducer.

    Uses STFT-based spectral gating with proper overlap-add reconstruction.

    Usage:
        denoiser = SpectralGateDenoiser(sample_rate=16000)

        # Option 1: Learn noise from reference
        denoiser.learn_noise(noise_audio)

        # Option 2: Auto-learn from first N silent frames
        denoiser.enable_auto_learn()

        # Process audio chunks
        clean_audio = denoiser.process(noisy_chunk)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        config: Optional[DenoiseConfig] = None,
    ):
        self.sample_rate = sample_rate
        self.config = config or DenoiseConfig()

        # FFT parameters
        self.n_fft = self.config.n_fft
        self.hop_length = self.config.hop_length
        self.win_length = self.config.win_length
        self.n_bins = self.n_fft // 2 + 1

        # Use sqrt-Hann window for analysis and synthesis (perfect reconstruction)
        hann = np.hanning(self.win_length).astype(np.float32)
        self.window = np.sqrt(hann)

        # Noise profile (magnitude spectrum)
        self.noise_profile: Optional[NDArray[np.float32]] = None
        self.noise_frames_collected = 0
        self.auto_learn_enabled = False

        # Smoothing state for gain
        self.prev_gain = np.ones(self.n_bins, dtype=np.float32)
        self._compute_smoothing_coeffs()

        # Convert thresholds to linear scale
        self.threshold_mult = 10 ** (self.config.threshold_db / 20)
        self.reduction_mult = 10 ** (self.config.reduction_db / 20)

    def _compute_smoothing_coeffs(self):
        """Compute attack/release smoothing coefficients."""
        frames_per_sec = self.sample_rate / self.hop_length
        self.attack_coeff = np.exp(-1.0 / (self.config.attack_time * frames_per_sec))
        self.release_coeff = np.exp(-1.0 / (self.config.release_time * frames_per_sec))

    def learn_noise(self, noise_audio: NDArray[np.float32]) -> None:
        """Learn noise profile from reference audio.

        Args:
            noise_audio: Audio containing only noise (no speech)
        """
        if len(noise_audio) < self.n_fft:
            logger.warning("Noise audio too short for learning")
            return

        # Compute STFT of noise
        n_frames = (len(noise_audio) - self.n_fft) // self.hop_length + 1
        noise_mag = np.zeros(self.n_bins, dtype=np.float32)

        for i in range(n_frames):
            start = i * self.hop_length
            frame = noise_audio[start : start + self.n_fft]
            if len(frame) < self.n_fft:
                frame = np.pad(frame, (0, self.n_fft - len(frame)))
            windowed = frame * self.window
            spectrum = np.fft.rfft(windowed)
            noise_mag += np.abs(spectrum)

        # Average and apply floor
        self.noise_profile = np.maximum(noise_mag / n_frames, self.config.noise_floor)
        self.noise_frames_collected = n_frames
        logger.info(f"Learned noise profile from {n_frames} frames")

    def enable_auto_learn(self, enabled: bool = True) -> None:
        """Enable automatic noise learning from initial frames."""
        self.auto_learn_enabled = enabled
        if enabled:
            self.noise_profile = None
            self.noise_frames_collected = 0
            logger.info("Auto noise learning enabled")

    def reset(self) -> None:
        """Reset internal state."""
        self.input_buffer.fill(0)
        self.output_buffer.fill(0)
        self.prev_gain.fill(1)
        if self.auto_learn_enabled:
            self.noise_profile = None
            self.noise_frames_collected = 0

    def _update_noise_profile(self, magnitude: NDArray[np.float32]) -> None:
        """Update noise profile with new frame (for auto-learning)."""
        if self.noise_profile is None:
            self.noise_profile = magnitude.copy()
        else:
            # Exponential moving average
            alpha = 1.0 / (self.noise_frames_collected + 1)
            self.noise_profile = (1 - alpha) * self.noise_profile + alpha * magnitude
        self.noise_frames_collected += 1

    def _compute_gain(self, magnitude: NDArray[np.float32]) -> NDArray[np.float32]:
        """Compute spectral gate gain for each frequency bin.

        Uses soft spectral gating:
        - Above threshold: gain approaches 1.0
        - Below threshold: gain approaches reduction_mult
        - Smooth transition around threshold
        """
        if self.noise_profile is None:
            return np.ones_like(magnitude)

        # Threshold = noise * multiplier (threshold_db converts to multiplier)
        threshold = self.noise_profile * self.threshold_mult

        # Soft gate: sigmoid-like transition
        # When magnitude >> threshold: gain -> 1.0
        # When magnitude << threshold: gain -> reduction_mult
        # Smooth transition width controlled by threshold
        ratio = magnitude / (threshold + 1e-10)

        # Soft knee using tanh for smooth transition
        # Map ratio: 0->reduction_mult, 1->~0.5, 2->~0.9, inf->1.0
        soft_gain = 0.5 * (1 + np.tanh(2 * (ratio - 1)))  # 0 to 1

        # Scale to range [reduction_mult, 1.0]
        gain = self.reduction_mult + soft_gain * (1.0 - self.reduction_mult)

        # Frequency smoothing (optional)
        if self.config.freq_smoothing > 1:
            kernel = np.ones(self.config.freq_smoothing) / self.config.freq_smoothing
            gain = np.convolve(gain, kernel, mode="same")

        return gain.astype(np.float32)

    def _smooth_gain(self, gain: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply temporal smoothing to gain (attack/release)."""
        # Attack: fast response when gain increases (signal present)
        # Release: slow response when gain decreases (signal absent)
        smooth_gain = np.where(
            gain > self.prev_gain,
            self.attack_coeff * self.prev_gain + (1 - self.attack_coeff) * gain,
            self.release_coeff * self.prev_gain + (1 - self.release_coeff) * gain,
        )
        self.prev_gain = smooth_gain
        return smooth_gain

    def process_frame(self, frame: NDArray[np.float32]) -> NDArray[np.float32]:
        """Process a single frame (n_fft samples).

        Args:
            frame: Input frame of n_fft samples

        Returns:
            Denoised frame (windowed for overlap-add)
        """
        # Apply window and compute FFT
        windowed = frame * self.window
        spectrum = np.fft.rfft(windowed)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)

        # Auto-learn noise from initial frames
        if self.auto_learn_enabled and self.noise_frames_collected < self.config.noise_frames:
            self._update_noise_profile(magnitude)
            # During learning, pass through with synthesis window for correct reconstruction
            if self.noise_frames_collected < self.config.noise_frames:
                passthrough = np.fft.irfft(spectrum, n=self.n_fft)
                return (passthrough * self.window).astype(np.float32)

        # Compute and smooth gain
        gain = self._compute_gain(magnitude)
        gain = self._smooth_gain(gain)

        # Apply gain to spectrum
        filtered_spectrum = magnitude * gain * np.exp(1j * phase)

        # Inverse FFT with synthesis window (sqrt-Hann for WOLA reconstruction)
        filtered = np.fft.irfft(filtered_spectrum, n=self.n_fft)

        # Apply synthesis window (analysis * synthesis = Hann)
        return (filtered * self.window).astype(np.float32)

    def process(self, audio: NDArray[np.float32]) -> NDArray[np.float32]:
        """Process audio with overlap-add.

        Args:
            audio: Input audio (any length)

        Returns:
            Denoised audio (same length as input)
        """
        if len(audio) == 0:
            return audio

        # Pad input to multiple of hop_length
        pad_len = (self.hop_length - len(audio) % self.hop_length) % self.hop_length
        if pad_len > 0:
            audio = np.pad(audio, (0, pad_len))

        output = np.zeros(len(audio), dtype=np.float32)
        window_sum = np.zeros(len(audio), dtype=np.float32)

        # Process with overlap-add
        n_frames = (len(audio) - self.n_fft) // self.hop_length + 1
        for i in range(n_frames):
            start = i * self.hop_length
            end = start + self.n_fft

            if end > len(audio):
                break

            frame = audio[start:end]
            processed = self.process_frame(frame)

            output[start:end] += processed
            # With sqrt-Hann analysis and synthesis, effective window is Hann
            window_sum[start:end] += self.window ** 2  # = Hann window

        # Normalize by window sum (avoid division by zero)
        window_sum = np.maximum(window_sum, 1e-8)
        output /= window_sum

        # Remove padding
        if pad_len > 0:
            output = output[:-pad_len]

        return output


# Global cache for MLDenoiser (avoid reloading model every call)
_ml_denoiser_cache: Optional[MLDenoiser] = None


def _get_cached_ml_denoiser(device: str = "cpu") -> MLDenoiser:
    """Get or create cached MLDenoiser instance."""
    global _ml_denoiser_cache
    if _ml_denoiser_cache is None:
        _ml_denoiser_cache = MLDenoiser(device=device)
    return _ml_denoiser_cache


def denoise(
    audio: NDArray[np.float32],
    sample_rate: int = 16000,
    method: str = "auto",
    noise_reference: Optional[NDArray[np.float32]] = None,
    threshold_db: float = 6.0,
    reduction_db: float = -24.0,
    device: str = "cpu",
) -> NDArray[np.float32]:
    """Denoise audio using the best available method.

    Args:
        audio: Input audio (mono, float32)
        sample_rate: Sample rate
        method: "auto" (try ML denoiser, fallback to spectral gate),
                "ml" (Facebook Denoiser, ML-based),
                "spectral" (traditional DSP)
        noise_reference: Noise reference for spectral gate (ignored for ML)
        threshold_db: Spectral gate threshold (ignored for ML)
        reduction_db: Spectral gate reduction (ignored for ML)
        device: Device for ML denoiser ("cpu", "cuda", "xpu")

    Returns:
        Denoised audio
    """
    # Select method
    use_ml = False

    if method == "ml" or method == "deepfilter":  # Accept both names for compatibility
        use_ml = True
    elif method == "auto":
        # Try ML denoiser first
        if is_ml_denoiser_available():
            use_ml = True
        else:
            logger.info("ML denoiser not available, using spectral gate")
            use_ml = False
    # else: method == "spectral"

    if use_ml:
        # Use cached denoiser to avoid reloading model every call
        denoiser = _get_cached_ml_denoiser(device=device)
        return denoiser.process(audio, sample_rate)
    else:
        config = DenoiseConfig(threshold_db=threshold_db, reduction_db=reduction_db)
        denoiser = SpectralGateDenoiser(sample_rate, config)

        if noise_reference is not None:
            denoiser.learn_noise(noise_reference)
        else:
            denoiser.enable_auto_learn()

        return denoiser.process(audio)


def is_ml_denoiser_available() -> bool:
    """Check if Facebook Denoiser is installed."""
    try:
        import denoiser
        return True
    except ImportError:
        return False


# Backwards compatibility alias (deprecated)
def is_deepfilter_available() -> bool:
    """Check if ML denoiser is available.

    .. deprecated::
        Use :func:`is_ml_denoiser_available` instead.
        This alias will be removed in a future version.
    """
    import warnings
    warnings.warn(
        "is_deepfilter_available() is deprecated, use is_ml_denoiser_available() instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return is_ml_denoiser_available()
