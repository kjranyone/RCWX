"""Real-time noise reduction.

Supports multiple backends:
- Facebook Denoiser: ML-based, real-time capable, preserves human voice
- Spectral Gate: Traditional DSP, lower latency fallback

Facebook Denoiser is a PyTorch-based speech enhancement model that
removes background noise while preserving human voice.
"""

from __future__ import annotations

import importlib.util
import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from rcwx.accelerator_graph import accelerator_graph_enabled, run_accelerator_graph
from rcwx.audio.gtcrn import get_cached_gtcrn, is_gtcrn_available

logger = logging.getLogger(__name__)

# Throttle for the short-hop spectral warning (fired per hop otherwise).
_SHORT_HOP_WARN_INTERVAL = 5.0
_last_short_hop_warning = 0.0


def _warn_short_spectral_hop(length: int, n_fft: int) -> None:
    global _last_short_hop_warning
    now = time.time()
    if now - _last_short_hop_warning < _SHORT_HOP_WARN_INTERVAL:
        return
    _last_short_hop_warning = now
    logger.warning(
        "Spectral gate needs at least n_fft (%d) samples, got %d — passing "
        "audio through unchanged (use denoise method 'gtcrn' for short hops)",
        n_fft,
        length,
    )


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
        self._graph_resample_patched = False
        self._graph_upsample2 = None
        self._graph_downsample2 = None

    def _configure_graph_safe_resampling(self) -> None:
        """Prebuild Demucs sinc kernels that its upstream forward allocates."""
        if self._graph_resample_patched or self._actual_device == "cpu":
            return

        try:
            import torch
            import torch.nn.functional as functional
            from denoiser.resample import kernel_downsample2, kernel_upsample2

            up_kernel = kernel_upsample2().to(self._actual_device)
            down_kernel = kernel_downsample2().to(self._actual_device)

            def upsample2_static(x, zeros: int = 56):
                if zeros != 56:
                    raise ValueError("Graph-safe denoiser supports zeros=56")
                *other, sample_count = x.shape
                out = functional.conv1d(
                    x.reshape(-1, 1, sample_count),
                    up_kernel,
                    padding=zeros,
                )[..., 1:].reshape(*other, sample_count)
                return torch.stack([x, out], dim=-1).reshape(*other, -1)

            def downsample2_static(x, zeros: int = 56):
                if zeros != 56:
                    raise ValueError("Graph-safe denoiser supports zeros=56")
                if x.shape[-1] % 2:
                    x = functional.pad(x, (0, 1))
                even = x[..., ::2]
                odd = x[..., 1::2]
                *other, sample_count = odd.shape
                filtered = functional.conv1d(
                    odd.reshape(-1, 1, sample_count),
                    down_kernel,
                    padding=zeros,
                )[..., :-1].reshape(*other, sample_count)
                return (even + filtered).reshape(*other, -1).mul(0.5)

            self._graph_upsample2 = upsample2_static
            self._graph_downsample2 = downsample2_static
            self._graph_resample_patched = True
        except Exception as exc:
            logger.warning("Could not prebuild graph-safe denoiser kernels: %s", exc)

    def _graph_forward(self, audio_tensor):
        """Run Demucs with static sinc kernels only for this invocation."""
        if not self._graph_resample_patched:
            return self._model(audio_tensor)

        import denoiser.demucs as demucs_module

        original_up = demucs_module.upsample2
        original_down = demucs_module.downsample2
        demucs_module.upsample2 = self._graph_upsample2
        demucs_module.downsample2 = self._graph_downsample2
        try:
            return self._model(audio_tensor)
        finally:
            demucs_module.upsample2 = original_up
            demucs_module.downsample2 = original_down

    def _load_model(self):
        """Lazy-load the Denoiser model."""
        if self._loaded:
            return

        try:
            import torch
            from denoiser import pretrained

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

            if accelerator_graph_enabled(self._actual_device):
                self._configure_graph_safe_resampling()
            self._loaded = True

        except ImportError:
            raise ImportError(
                "Facebook Denoiser not installed. Install with: pip install denoiser"
            )

    def process(
        self,
        audio: NDArray[np.float32],
        sample_rate: int = 16000,
        strength: float = 1.0,
    ) -> NDArray[np.float32]:
        """Process audio through Facebook Denoiser.

        Args:
            audio: Input audio (mono, float32, -1 to 1 range)
            sample_rate: Input sample rate
            strength: Suppression strength from 0.5 to 2.0. Up to 1.0
                crossfades dry/clean audio; above 1.0 crossfades into a
                second DNS64 pass for stronger residual-noise suppression.

        Returns:
            Denoised audio at original sample rate
        """
        self._load_model()
        strength = max(0.5, min(2.0, float(strength)))

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

        # Enhance. The denoiser is part of the live critical path, so replay
        # fixed micro-hop shapes through the same accelerator graph cache as
        # HuBERT/Synthesizer. Unsupported devices or operators fall back to
        # eager execution inside run_accelerator_graph().
        def enhance(value):
            if accelerator_graph_enabled(self._actual_device):
                return run_accelerator_graph(
                    self,
                    "ml-denoiser",
                    lambda graph_input: self._graph_forward(graph_input),
                    value,
                )
            return self._model(value)

        with torch.no_grad():
            enhanced = enhance(audio_tensor)
            if strength < 1.0:
                enhanced = torch.lerp(audio_tensor, enhanced, strength)
            elif strength > 1.0:
                enhanced_twice = enhance(enhanced)
                enhanced = torch.lerp(enhanced, enhanced_twice, strength - 1.0)

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
        """Reset internal state.

        ``process()`` performs a self-contained overlap-add per call, so the
        only state carried across calls is the temporal gain smoothing and the
        (optional) auto-learned noise profile.
        """
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


class StreamingSpectralGate:
    """Persistent STFT spectral gate for the realtime path.

    ``SpectralGateDenoiser`` is self-contained per call: rebuilt every hop,
    it can never finish auto-learning (10 frames = 416ms of audio), so per-hop
    use degenerated to a windowed passthrough — spectral "denoising" did no
    denoising at all for chunks below ~416ms.  This class instead carries the
    STFT state across hops:

    * sqrt-Hann analysis+synthesis at n_fft=512 / hop=256 (GTCRN geometry):
      adjacent window pairs satisfy COLA exactly (w²(n) + w²(n+H) = 1), so
      finalized samples need no per-sample normalization — the chunk-edge
      amplification of the offline overlap-add cannot occur.
    * the noise profile is tracked continuously with asymmetric one-pole
      smoothing (fast toward quieter frames, slow toward louder ones), so it
      settles to the noise floor during speech pauses instead of freezing
      on whatever the first 416ms contained.
    * zero added latency: everything except the newest ``n_fft - hop``
      samples is exact WOLA output; the newest region is crossfaded toward
      the dry signal (the same bounded approximation GTCRN makes for its
      newest 16ms).
    """

    N_FFT = 512
    HOP = 256

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold_db: float = 6.0,
        reduction_db: float = -24.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.window = np.sqrt(np.hanning(self.N_FFT).astype(np.float32))
        self.n_bins = self.N_FFT // 2 + 1

        hop_sec = self.HOP / sample_rate
        # Per-bin magnitudes of stationary noise are Rayleigh-distributed
        # (high variance); track a short symmetric average so the estimate is
        # low-variance, then apply minimum statistics: drop to minima
        # instantly (noise floor appears during speech pauses / spectral
        # valleys) and rise at a capped +6dB/s so sustained speech cannot
        # drag the floor estimate up.
        self._mag_smooth_coeff = float(np.exp(-hop_sec / 0.25))
        self._rise_factor = float(10 ** (6.0 / 20.0 * hop_sec))
        # Pass the first ~250ms at unity while mag_smooth converges: gating
        # on a single-frame profile is erratic per-bin.
        self._bootstrap_frames = max(1, int(round(0.25 / hop_sec)))
        # Per-bin gain temporal smoothing (fast open, slow close).
        self._g_attack = float(np.exp(-hop_sec / 0.01))
        self._g_release = float(np.exp(-hop_sec / 0.05))
        self._freq_kernel = np.ones(3) / 3.0
        self._noise_floor = 1e-4

        self._threshold_db = 0.0
        self._reduction_db = 0.0
        self.configure(threshold_db, reduction_db)
        self.reset()

    # -- configuration -------------------------------------------------

    def configure(self, threshold_db: float, reduction_db: float) -> None:
        """Update gate parameters in place (strength slider, live)."""
        self._threshold_mult = 10 ** (float(threshold_db) / 20)
        # Cap the floor at -60dB; deeper suppression just trades residual
        # musical noise for pumping.
        reduction_db = max(-60.0, min(-3.0, float(reduction_db)))
        self._reduction_mult = 10 ** (reduction_db / 20)

    def reset(self) -> None:
        """Reset stream state (new session / stream restart)."""
        self.noise_profile: Optional[NDArray[np.float32]] = None
        self._mag_smooth: Optional[NDArray[np.float32]] = None
        self._frames_seen = 0
        self._prev_gain = np.ones(self.n_bins, dtype=np.float32)
        # Absolute timeline: 256 zero samples precede the stream so the first
        # input samples land in full-coverage window positions.  ``_buf``
        # starts at absolute position ``_pos``; ``_ola``/``_win`` accumulate
        # from ``_ola_start``; everything below ``_next_out`` is emitted.
        self._buf = np.zeros(self.HOP, dtype=np.float32)
        self._pos = 0
        self._ola_start = 0
        self._ola = np.zeros(0, dtype=np.float32)
        self._win = np.zeros(0, dtype=np.float32)
        self._next_out = self.HOP

    # -- frame pipeline ------------------------------------------------

    def _update_profile(self, magnitude: NDArray[np.float32]) -> None:
        self._frames_seen += 1
        if self._mag_smooth is None:
            self._mag_smooth = magnitude.copy()
        else:
            b = self._mag_smooth_coeff
            self._mag_smooth = b * self._mag_smooth + (1.0 - b) * magnitude
        if self.noise_profile is None:
            self.noise_profile = np.maximum(self._mag_smooth, self._noise_floor)
            return
        risen = self.noise_profile * self._rise_factor
        self.noise_profile = np.maximum(
            np.minimum(self._mag_smooth, risen), self._noise_floor
        )

    def _compute_gain(self, magnitude: NDArray[np.float32]) -> NDArray[np.float32]:
        if self.noise_profile is None or self._frames_seen < self._bootstrap_frames:
            return np.ones(self.n_bins, dtype=np.float32)
        threshold = self.noise_profile * self._threshold_mult
        ratio = magnitude / (threshold + 1e-10)
        soft_gain = 0.5 * (1.0 + np.tanh(2.0 * (ratio - 1.0)))
        gain = self._reduction_mult + soft_gain * (1.0 - self._reduction_mult)
        return np.convolve(gain, self._freq_kernel, mode="same").astype(np.float32)

    def _gate_frame(self, frame: NDArray[np.float32], learn: bool) -> NDArray[np.float32]:
        """Gate one n_fft frame; returns the windowed output frame."""
        spectrum = np.fft.rfft(frame * self.window)
        magnitude = np.abs(spectrum)
        if learn:
            self._update_profile(magnitude)
        gain = self._compute_gain(magnitude)
        # Temporal smoothing from the finalized gain sequence; speculative
        # frames read prev_gain without advancing it, so their smoothing
        # matches what the finalized recomputation will apply.
        prev = self._prev_gain
        smoothed = np.where(
            gain > prev,
            self._g_attack * prev + (1.0 - self._g_attack) * gain,
            self._g_release * prev + (1.0 - self._g_release) * gain,
        ).astype(np.float32)
        if learn:
            self._prev_gain = smoothed
        return np.fft.irfft(spectrum * smoothed, n=self.N_FFT) * self.window

    @staticmethod
    def _grow(array: NDArray[np.float32], needed: int) -> NDArray[np.float32]:
        if needed <= len(array):
            return array
        return np.concatenate(
            [array, np.zeros(needed - len(array), dtype=np.float32)]
        )

    # -- streaming API -------------------------------------------------

    def process(self, audio: NDArray[np.float32]) -> NDArray[np.float32]:
        """Denoise one hop; returns exactly ``len(audio)`` samples."""
        audio = audio.astype(np.float32, copy=False)
        n = len(audio)
        if n == 0:
            return audio

        self._buf = np.concatenate([self._buf, audio])
        s_end = self._pos + len(self._buf)
        emit_start = self._next_out

        # Finalized frames: frames advance by hop, so every sample below the
        # next frame start (_pos) has all of its covering frames processed.
        while len(self._buf) >= self.N_FFT:
            enhanced = self._gate_frame(self._buf[: self.N_FFT], learn=True)
            lo = self._pos - self._ola_start
            needed = lo + self.N_FFT
            self._ola = self._grow(self._ola, needed)
            self._win = self._grow(self._win, needed)
            self._ola[lo : lo + self.N_FFT] += enhanced
            self._win[lo : lo + self.N_FFT] += self.window**2
            self._buf = self._buf[self.HOP :]
            self._pos += self.HOP

        exact_end = min(self._pos, s_end)
        if exact_end > emit_start:
            lo = emit_start - self._ola_start
            exact = self._ola[lo : lo + (exact_end - emit_start)].copy()
        else:
            exact = np.zeros(0, dtype=np.float32)

        # Speculative tail: frames beyond _pos need future input.  Run them
        # zero-padded without touching persistent state, then crossfade to
        # dry by window coverage so the approximation neither attenuates nor
        # amplifies the newest audio.
        spec_lo = max(self._pos, emit_start)
        spec_len = s_end - spec_lo
        spec = np.zeros(max(0, spec_len), dtype=np.float32)
        if spec_len > 0:
            base = self._pos - self._ola_start
            spec_ola = self._ola[base:].copy()
            spec_win = self._win[base:].copy()
            gain_snapshot = self._prev_gain.copy()
            spec_pos = self._pos
            while spec_pos < s_end:
                offset = spec_pos - self._pos
                frame = self._buf[offset:]
                if len(frame) < self.N_FFT:
                    frame = np.pad(frame, (0, self.N_FFT - len(frame)))
                enhanced = self._gate_frame(frame, learn=False)
                lo = spec_pos - self._pos
                needed = lo + self.N_FFT
                spec_ola = self._grow(spec_ola, needed)
                spec_win = self._grow(spec_win, needed)
                spec_ola[lo : lo + self.N_FFT] += enhanced
                spec_win[lo : lo + self.N_FFT] += self.window**2
                spec_pos += self.HOP
            self._prev_gain = gain_snapshot
            lo = spec_lo - self._pos
            coverage = np.clip(spec_win[lo : lo + spec_len], 0.0, 1.0)
            dry = self._buf[lo : lo + spec_len]
            spec = (spec_ola[lo : lo + spec_len] * coverage + dry * (1.0 - coverage)).astype(
                np.float32
            )

        # Retire emitted samples; keep the tail that overlaps future frames.
        keep = self._pos - self._ola_start
        self._ola = self._ola[keep:].copy()
        self._win = self._win[keep:].copy()
        self._ola_start = self._pos
        self._next_out = s_end

        return np.concatenate([exact, spec])


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
    strength: float = 1.0,
    device: str = "cpu",
    streaming_state: Optional[StreamingSpectralGate] = None,
) -> NDArray[np.float32]:
    """Denoise audio using the best available method.

    Args:
        audio: Input audio (mono, float32)
        sample_rate: Sample rate
        method: "auto" (try ML denoiser, fallback to spectral gate),
                "ml" (Facebook Denoiser, ML-based, GPU),
                "gtcrn" (GTCRN, MIT license, CPU ONNX, 16kHz only),
                "spectral" (traditional DSP)
        noise_reference: Noise reference for spectral gate (ignored for ML)
        threshold_db: Spectral gate threshold (ignored for ML)
        reduction_db: Spectral gate reduction (ignored for ML)
        strength: Suppression strength from 0.5 to 2.0. ML uses a blended
            second pass above 1.0; spectral scales its threshold/reduction.
        device: Device for ML denoiser ("cpu", "cuda", "xpu")
        streaming_state: Persistent ``StreamingSpectralGate`` for the
            realtime path.  Without it the spectral fallback is stateless
            and rebuilds per call (fine for whole files, useless per hop:
            it never finishes learning its noise profile).  ``noise_reference``
            is ignored when this is given.

    Returns:
        Denoised audio
    """
    strength = max(0.5, min(2.0, float(strength)))

    # Select method
    use_ml = False

    if method == "gtcrn":
        # MIT-licensed streaming denoiser on CPU (no GPU contention).
        # Degrades to spectral gate when onnxruntime is unavailable or the
        # model cannot be loaded (e.g. first-use download fails offline).
        if is_gtcrn_available():
            try:
                return get_cached_gtcrn().process(
                    audio, sample_rate, strength=strength
                )
            except Exception as e:
                logger.warning(
                    f"GTCRN denoiser failed ({e}); falling back to spectral gate"
                )
        else:
            logger.warning(
                "GTCRN denoiser requested but onnxruntime is not installed; "
                "falling back to spectral gate"
            )

    if method == "ml":
        # Explicit ML request still degrades gracefully when the optional
        # 'denoiser' package is missing, so a stale config (or direct call)
        # never crashes the real-time inference thread with ImportError.
        if is_ml_denoiser_available():
            use_ml = True
        else:
            logger.warning(
                "ML denoiser requested but 'denoiser' is not installed; "
                "falling back to spectral gate. Install with: "
                "uv sync --extra ml-denoise"
            )
            use_ml = False
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
        return denoiser.process(audio, sample_rate, strength=strength)
    else:
        if streaming_state is not None:
            # Persistent streaming gate: a fresh SpectralGateDenoiser per hop
            # never finishes auto-learning (10 frames = 416ms) and degenerates
            # to a windowed passthrough for typical realtime chunks.
            streaming_state.configure(
                threshold_db=threshold_db * strength,
                reduction_db=reduction_db * strength,
            )
            return streaming_state.process(audio)
        config = DenoiseConfig(
            threshold_db=threshold_db * strength,
            reduction_db=reduction_db * strength,
        )
        if len(audio) < config.n_fft:
            # The spectral gate needs one full analysis window (n_fft = 2048
            # samples = 128ms @16k).  Shorter realtime hops (all Aggressive
            # chunks, and Normal chunks below 128ms) yield no analysis frame
            # and the overlap-add loop would return pure silence — pass the
            # audio through unchanged instead of muting the stream.
            _warn_short_spectral_hop(len(audio), config.n_fft)
            return audio.astype(np.float32, copy=False)
        denoiser = SpectralGateDenoiser(sample_rate, config)

        if noise_reference is not None:
            denoiser.learn_noise(noise_reference)
        else:
            denoiser.enable_auto_learn()

        return denoiser.process(audio)


def is_ml_denoiser_available() -> bool:
    """Check if Facebook Denoiser is installed."""
    return importlib.util.find_spec("denoiser") is not None
