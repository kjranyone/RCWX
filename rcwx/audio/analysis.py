"""Audio analysis utilities for adaptive parameter adjustment."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_energy(audio: NDArray[np.float32]) -> float:
    """
    Compute RMS energy of audio.

    Args:
        audio: Audio samples [T]

    Returns:
        RMS energy (0-1 range typically)
    """
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio**2)))


def compute_spectral_flux(audio: NDArray[np.float32], sample_rate: int = 48000) -> float:
    """
    Compute spectral flux (rate of spectral change).

    Higher values indicate transient/percussive content.
    Lower values indicate stable/sustained content.

    Args:
        audio: Audio samples [T]
        sample_rate: Audio sample rate

    Returns:
        Spectral flux (normalized, higher = more transient)
    """
    if len(audio) < 512:
        return 0.0

    # Simple spectral flux using frame-to-frame difference
    frame_size = 512
    hop_size = 256

    num_frames = (len(audio) - frame_size) // hop_size
    if num_frames < 2:
        return 0.0

    flux_values = []
    prev_spectrum = None

    for i in range(num_frames):
        start = i * hop_size
        frame = audio[start:start + frame_size]

        # Apply window
        frame = frame * np.hanning(frame_size)

        # Compute magnitude spectrum
        spectrum = np.abs(np.fft.rfft(frame))

        if prev_spectrum is not None:
            # Compute flux (sum of positive differences)
            diff = spectrum - prev_spectrum
            flux = np.sum(np.maximum(diff, 0))
            flux_values.append(flux)

        prev_spectrum = spectrum

    if not flux_values:
        return 0.0

    # Normalize to 0-1 range (roughly)
    mean_flux = np.mean(flux_values)
    # Typical flux values are in range 0-10000, normalize to 0-1
    normalized_flux = min(mean_flux / 5000.0, 1.0)

    return float(normalized_flux)


def compute_pitch_stability(
    f0: NDArray[np.float32] | None,
    threshold: float = 0.1,
) -> float:
    """
    Compute pitch stability metric.

    Measures how stable the F0 contour is (less variation = more stable).

    Args:
        f0: F0 contour [T] or None
        threshold: Threshold for considering pitch as voiced

    Returns:
        Stability metric (0-1, higher = more stable)
    """
    if f0 is None or len(f0) < 2:
        return 0.5  # Neutral value

    # Extract voiced regions (f0 > threshold)
    voiced_mask = f0 > threshold
    voiced_f0 = f0[voiced_mask]

    if len(voiced_f0) < 2:
        return 0.5  # Not enough data

    # Compute coefficient of variation (normalized std dev)
    mean_f0 = np.mean(voiced_f0)
    std_f0 = np.std(voiced_f0)

    if mean_f0 < 1e-6:
        return 0.5

    cv = std_f0 / mean_f0

    # Convert CV to stability (0-1, higher = more stable)
    # CV of 0.1 (10% variation) = moderately stable
    # CV of 0.3 (30% variation) = unstable
    stability = np.exp(-cv * 5)  # Exponential decay

    return float(np.clip(stability, 0.0, 1.0))


class AdaptiveParameterCalculator:
    """
    Calculates adaptive parameters based on audio characteristics.

    This implements the adaptive parameter adjustment strategy from
    architecture_analysis.md to improve chunk boundary quality.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        # Thresholds
        energy_threshold_low: float = 0.02,
        energy_threshold_high: float = 0.15,
        flux_threshold_low: float = 0.2,
        flux_threshold_high: float = 0.6,
        stability_threshold_low: float = 0.4,
        stability_threshold_high: float = 0.7,
        # Base parameters
        base_crossfade_sec: float = 0.119,
        base_context_sec: float = 0.119,
        base_sola_search_ms: float = 30.0,
    ):
        """
        Initialize adaptive parameter calculator.

        Args:
            sample_rate: Audio sample rate
            energy_threshold_low: Low energy threshold
            energy_threshold_high: High energy threshold
            flux_threshold_low: Low flux threshold (stable)
            flux_threshold_high: High flux threshold (transient)
            stability_threshold_low: Low stability threshold
            stability_threshold_high: High stability threshold
            base_crossfade_sec: Base crossfade duration
            base_context_sec: Base context duration
            base_sola_search_ms: Base SOLA search range (ms)
        """
        self.sample_rate = sample_rate

        # Thresholds
        self.energy_threshold_low = energy_threshold_low
        self.energy_threshold_high = energy_threshold_high
        self.flux_threshold_low = flux_threshold_low
        self.flux_threshold_high = flux_threshold_high
        self.stability_threshold_low = stability_threshold_low
        self.stability_threshold_high = stability_threshold_high

        # Base parameters
        self.base_crossfade_sec = base_crossfade_sec
        self.base_context_sec = base_context_sec
        self.base_sola_search_ms = base_sola_search_ms

    def calculate_crossfade_sec(
        self,
        energy: float,
        flux: float,
    ) -> float:
        """
        Calculate adaptive crossfade duration.

        Strategy:
        - Low energy (silence): larger crossfade (more smoothing)
        - High flux (transient): shorter crossfade (preserve transients)
        - Normal: base value

        Args:
            energy: Energy level (0-1)
            flux: Spectral flux (0-1)

        Returns:
            Crossfade duration in seconds
        """
        # Start with base value
        crossfade_sec = self.base_crossfade_sec

        # Low energy: increase crossfade (1.5x)
        if energy < self.energy_threshold_low:
            crossfade_sec = self.base_crossfade_sec * 1.5

        # High flux (transient): decrease crossfade (0.7x)
        elif flux > self.flux_threshold_high:
            crossfade_sec = self.base_crossfade_sec * 0.7

        # Normal: use base value
        else:
            crossfade_sec = self.base_crossfade_sec

        # Clamp to reasonable range (50ms - 200ms)
        crossfade_sec = np.clip(crossfade_sec, 0.05, 0.20)

        return float(crossfade_sec)

    def calculate_context_sec(
        self,
        energy: float,
    ) -> float:
        """
        Calculate adaptive context duration.

        Strategy:
        - Low energy: smaller context (less processing needed)
        - High energy: larger context (more stability)

        Args:
            energy: Energy level (0-1)

        Returns:
            Context duration in seconds
        """
        # Start with base value
        context_sec = self.base_context_sec

        # Low energy: reduce context (0.7x)
        if energy < self.energy_threshold_low:
            context_sec = self.base_context_sec * 0.7

        # High energy: increase context (1.2x)
        elif energy > self.energy_threshold_high:
            context_sec = self.base_context_sec * 1.2

        # Normal: use base value
        else:
            context_sec = self.base_context_sec

        # Clamp to reasonable range (50ms - 200ms)
        context_sec = np.clip(context_sec, 0.05, 0.20)

        return float(context_sec)

    def calculate_sola_search_ms(
        self,
        stability: float,
    ) -> float:
        """
        Calculate adaptive SOLA search range.

        Strategy:
        - Low stability (varying pitch): larger search range
        - High stability (steady pitch): smaller search range

        Args:
            stability: Pitch stability (0-1, higher = more stable)

        Returns:
            SOLA search range in milliseconds
        """
        # Start with base value
        sola_search_ms = self.base_sola_search_ms

        # Low stability: increase search range (1.5x)
        if stability < self.stability_threshold_low:
            sola_search_ms = self.base_sola_search_ms * 1.5

        # High stability: decrease search range (0.7x)
        elif stability > self.stability_threshold_high:
            sola_search_ms = self.base_sola_search_ms * 0.7

        # Normal: use base value
        else:
            sola_search_ms = self.base_sola_search_ms

        # Clamp to reasonable range (20ms - 60ms)
        sola_search_ms = np.clip(sola_search_ms, 20.0, 60.0)

        return float(sola_search_ms)

    def analyze_and_adjust(
        self,
        audio: NDArray[np.float32],
        f0: NDArray[np.float32] | None = None,
    ) -> dict[str, float]:
        """
        Analyze audio and calculate adaptive parameters.

        Args:
            audio: Audio samples [T] at sample_rate
            f0: Optional F0 contour for pitch stability analysis

        Returns:
            Dictionary with adjusted parameters:
            - crossfade_sec
            - context_sec
            - sola_search_ms
            - energy (for debugging)
            - flux (for debugging)
            - stability (for debugging)
        """
        # Analyze audio characteristics
        energy = compute_energy(audio)
        flux = compute_spectral_flux(audio, self.sample_rate)
        stability = compute_pitch_stability(f0)

        # Calculate adaptive parameters
        crossfade_sec = self.calculate_crossfade_sec(energy, flux)
        context_sec = self.calculate_context_sec(energy)
        sola_search_ms = self.calculate_sola_search_ms(stability)

        return {
            'crossfade_sec': crossfade_sec,
            'context_sec': context_sec,
            'sola_search_ms': sola_search_ms,
            'energy': energy,
            'flux': flux,
            'stability': stability,
        }
