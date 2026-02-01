"""Audio input stream using sounddevice with robust fallback."""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
from numpy.typing import NDArray

from rcwx.audio.stream_base import AudioStreamBase, AudioStreamError, list_devices, get_default_device

logger = logging.getLogger(__name__)


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
            # Convert to mono based on channel selection
            if indata.ndim > 1 and indata.shape[1] > 1:
                # Stereo input
                if self._channel_selection == "left":
                    audio = indata[:, 0].astype(np.float32)
                elif self._channel_selection == "right":
                    audio = indata[:, 1].astype(np.float32)
                else:  # "average"
                    audio = np.mean(indata, axis=1).astype(np.float32)
            else:
                # Mono input
                audio = indata[:, 0].astype(np.float32) if indata.ndim > 1 else indata.astype(np.float32)

            self._callback(audio)

    def set_callback(self, callback: Callable[[NDArray[np.float32]], None]) -> None:
        """Set the audio callback function."""
        self._callback = callback


def list_input_devices(wasapi_only: bool = False) -> list[dict]:
    """List available audio input devices (all drivers by default)."""
    return list_devices("input", wasapi_only)


def get_default_input_device() -> Optional[int]:
    """Get the default input device index."""
    return get_default_device("input")
