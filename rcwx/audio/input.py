"""Audio input stream using sounddevice with robust fallback."""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
from numpy.typing import NDArray

from rcwx.audio.stream_base import AudioStreamBase, AudioStreamError, list_devices, get_default_device

logger = logging.getLogger(__name__)


def _auto_select_channel(
    indata: NDArray[np.float32],
    exclude_channels: Optional[frozenset[int]] = None,
) -> NDArray[np.float32]:
    """Auto-detect active channel for multi-channel input.

    For 2-channel (stereo) input:
        Many USB microphones report as stereo but only output on one channel.
        Compares energy of L/R channels and uses the active one if the other
        is near-silent (< 1% energy ratio). Otherwise averages both.

    For >2-channel input (e.g. ASIO with loopback):
        Picks the single loudest channel among the candidates.
        ``exclude_channels`` removes channels from consideration — used to
        keep ASIO LOOPBACK channels (which carry the device's own playback,
        i.e. the voice changer output) from being auto-selected, which would
        close a feedback loop.
    """
    n_channels = indata.shape[1]
    candidates = [
        i for i in range(n_channels)
        if not exclude_channels or i not in exclude_channels
    ]
    if not candidates:
        candidates = list(range(n_channels))

    if len(candidates) == 2:
        # Stereo logic — average when both active
        left = indata[:, candidates[0]]
        right = indata[:, candidates[1]]
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

        return np.mean(indata[:, candidates], axis=1).astype(np.float32)

    if len(candidates) == 1:
        return indata[:, candidates[0]].astype(np.float32)

    # >2 candidates: pick the single loudest channel
    energies = [float(np.dot(indata[:, i], indata[:, i])) for i in candidates]
    energy_max = max(energies)

    if energy_max < 1e-10:
        return indata[:, candidates[0]].astype(np.float32)

    loudest = candidates[int(np.argmax(energies))]
    return indata[:, loudest].astype(np.float32)


# Selections already warned about (avoid per-audio-block log spam)
_warned_channel_selections: set[str] = set()


def select_channel(
    indata: NDArray[np.float32],
    channel_selection: str,
    exclude_channels: Optional[frozenset[int]] = None,
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
    exclude_channels:
        Channels excluded from ``"auto"`` / ``"average"`` (e.g. ASIO
        LOOPBACK channels).  An explicit index selection always wins.
    """
    if indata.ndim == 1:
        return indata.astype(np.float32)
    if indata.shape[1] == 1:
        return indata[:, 0].astype(np.float32)

    if channel_selection == "auto":
        return _auto_select_channel(indata, exclude_channels)
    elif channel_selection == "left":
        return indata[:, 0].astype(np.float32)
    elif channel_selection == "right":
        return indata[:, min(1, indata.shape[1] - 1)].astype(np.float32)
    elif channel_selection == "average":
        candidates = [
            i for i in range(indata.shape[1])
            if not exclude_channels or i not in exclude_channels
        ]
        if not candidates:
            candidates = list(range(indata.shape[1]))
        return np.mean(indata[:, candidates], axis=1).astype(np.float32)
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
            # NEVER silently auto-select: an unparseable selection reaching
            # this point means an upstream wiring bug (this previously turned
            # an explicit mic-channel choice into loudest-channel auto pick,
            # feeding ASIO LOOPBACK output back into the pipeline).
            if channel_selection not in _warned_channel_selections:
                _warned_channel_selections.add(channel_selection)
                logger.warning(
                    "Unrecognized channel_selection %r — falling back to auto "
                    "channel detection. Fix the caller to pass a canonical "
                    "value ('auto', 'average', 'left', 'right', or an index).",
                    channel_selection,
                )
            return _auto_select_channel(indata, exclude_channels)


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
