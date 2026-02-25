"""Audio input stream using sounddevice with robust fallback."""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
from numpy.typing import NDArray

from rcwx.audio.stream_base import AudioStreamBase, AudioStreamError, list_devices, get_default_device

logger = logging.getLogger(__name__)


def _auto_select_channel(indata: NDArray[np.float32]) -> NDArray[np.float32]:
    """Auto-detect active channel for multi-channel input.

    For 2-channel (stereo) input:
        Many USB microphones report as stereo but only output on one channel.
        Compares energy of L/R channels and uses the active one if the other
        is near-silent (< 1% energy ratio). Otherwise averages both.

    For >2-channel input (e.g. ASIO with loopback):
        Picks the single loudest channel to avoid mixing in loopback or
        unrelated channels.
    """
    n_channels = indata.shape[1]

    if n_channels == 2:
        # Original stereo logic — average when both active
        left = indata[:, 0]
        right = indata[:, 1]
        energy_l = np.dot(left, left)
        energy_r = np.dot(right, right)
        energy_max = max(energy_l, energy_r)

        if energy_max < 1e-10:
            return left.astype(np.float32)

        ratio = min(energy_l, energy_r) / energy_max
        if ratio < 0.01:
            if energy_l >= energy_r:
                return left.astype(np.float32)
            else:
                return right.astype(np.float32)

        return np.mean(indata, axis=1).astype(np.float32)

    # >2 channels: pick the single loudest channel
    energies = [float(np.dot(indata[:, i], indata[:, i])) for i in range(n_channels)]
    energy_max = max(energies)

    if energy_max < 1e-10:
        return indata[:, 0].astype(np.float32)

    loudest = int(np.argmax(energies))
    return indata[:, loudest].astype(np.float32)


def select_channel(
    indata: NDArray[np.float32],
    channel_selection: str,
) -> NDArray[np.float32]:
    """Extract mono audio from multi-channel input.

    Parameters
    ----------
    indata:
        Audio data, shape ``(frames,)`` or ``(frames, channels)``.
    channel_selection:
        ``"auto"`` — auto-detect active channel.
        ``"left"`` — channel 0 (legacy alias for ``"0"``).
        ``"right"`` — channel 1 (legacy alias for ``"1"``).
        ``"average"`` — average all channels.
        ``"0"``, ``"1"``, ``"2"``, … — specific channel index.
    """
    if indata.ndim == 1:
        return indata.astype(np.float32)
    if indata.shape[1] == 1:
        return indata[:, 0].astype(np.float32)

    if channel_selection == "auto":
        return _auto_select_channel(indata)
    elif channel_selection == "left":
        return indata[:, 0].astype(np.float32)
    elif channel_selection == "right":
        return indata[:, min(1, indata.shape[1] - 1)].astype(np.float32)
    elif channel_selection == "average":
        return np.mean(indata, axis=1).astype(np.float32)
    else:
        try:
            idx = int(channel_selection)
            if 0 <= idx < indata.shape[1]:
                return indata[:, idx].astype(np.float32)
            else:
                logger.warning(
                    "Channel index %d out of range (max %d), falling back to ch 0",
                    idx, indata.shape[1] - 1,
                )
                return indata[:, 0].astype(np.float32)
        except ValueError:
            return _auto_select_channel(indata)


class AudioInputError(AudioStreamError):
    """Exception raised when audio input cannot be opened."""

    pass


class AudioInput(AudioStreamBase):
    """
    Audio input stream manager with robust fallback.

    Captures audio from microphone using sounddevice.
    """

    STREAM_TYPE = "input"
    STREAM_CLASS = sd.InputStream

    def __init__(
        self,
        device: Optional[int] = None,
        sample_rate: int = 16000,
        channels: int = 1,
        blocksize: int = 1024,
        callback: Optional[Callable[[NDArray[np.float32]], None]] = None,
        channel_selection: str = "average",
    ):
        super().__init__(device, sample_rate, channels, blocksize)
        self._callback = callback
        self._channel_selection = channel_selection  # "left", "right", "average"

    def _audio_callback(
        self,
        indata: NDArray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        """Internal sounddevice callback."""
        if status:
            logger.warning(f"Input stream status: {status}")

        if self._callback is not None:
            audio = select_channel(indata, self._channel_selection)
            self._callback(audio)

def list_input_devices(wasapi_only: bool = False) -> list[dict]:
    """List available audio input devices (all drivers by default)."""
    return list_devices("input", wasapi_only)
