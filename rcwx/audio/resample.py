"""Audio resampling utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample_poly


def resample(
    audio: NDArray[np.float32],
    orig_sr: int,
    target_sr: int,
    method: str = "poly",
) -> NDArray[np.float32]:
    """
    Resample audio to target sample rate.

    Args:
        audio: Input audio array (1D)
        orig_sr: Original sample rate
        target_sr: Target sample rate
        method: Resampling method
            - "poly": scipy polyphase (high quality, slow)
            - "linear": linear interpolation (fast, lower quality)

    Returns:
        Resampled audio array
    """
    if orig_sr == target_sr:
        return audio

    if method == "linear":
        # Fast linear interpolation (good enough for real-time)
        ratio = target_sr / orig_sr
        target_len = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, target_len)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
    else:
        # High-quality polyphase resampling (default)
        from math import gcd

        g = gcd(orig_sr, target_sr)
        up = target_sr // g
        down = orig_sr // g

        return resample_poly(audio, up, down).astype(np.float32)
