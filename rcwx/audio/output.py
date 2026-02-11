"""Audio output stream using sounddevice with robust fallback."""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
from numpy.typing import NDArray

from rcwx.audio.stream_base import AudioStreamBase, AudioStreamError, list_devices, get_default_device

logger = logging.getLogger(__name__)


class AudioOutputError(AudioStreamError):
    """Exception raised when audio output cannot be opened."""

    pass


class AudioOutput(AudioStreamBase):
    """
    Audio output stream manager with robust fallback.

    Plays processed audio using sounddevice.
    """

    STREAM_TYPE = "output"
    STREAM_CLASS = sd.OutputStream

    def __init__(
        self,
        device: Optional[int] = None,
        sample_rate: int = 48000,
        channels: int = 1,
        blocksize: int = 1024,
        callback: Optional[Callable[[int], NDArray[np.float32]]] = None,
    ):
        super().__init__(device, sample_rate, channels, blocksize)
        self._callback = callback

    def _audio_callback(
        self,
        outdata: NDArray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        """Internal sounddevice callback."""
        if status:
            logger.warning(f"Output stream status: {status}")

        if self._callback is not None:
            audio = self._callback(frames)
            if len(audio) >= frames:
                mono = audio[:frames]
            else:
                mono = np.zeros(frames, dtype=np.float32)
                if len(audio) > 0:
                    mono[: len(audio)] = audio

            # Write mono to all output channels
            if outdata.ndim > 1 and outdata.shape[1] > 1:
                outdata[:] = mono[:, np.newaxis]
            else:
                outdata[:, 0] = mono
        else:
            outdata.fill(0)

def list_output_devices(wasapi_only: bool = False) -> list[dict]:
    """List available audio output devices (all drivers by default)."""
    return list_devices("output", wasapi_only)
