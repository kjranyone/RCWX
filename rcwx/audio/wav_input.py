"""WAV file loop input that mimics AudioInput interface."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class WavFileInput:
    """Play a WAV file in a loop, delivering chunks via callback.

    Provides the same interface as AudioInput (start/stop/actual_sample_rate)
    so it can be used as a drop-in replacement in RealtimeVoiceChangerUnified.
    """

    def __init__(
        self,
        path: str,
        sample_rate: int,
        blocksize: int,
        callback: Optional[Callable[[NDArray[np.float32]], None]] = None,
        loop: bool = True,
    ) -> None:
        self._path = Path(path)
        self._sample_rate = sample_rate
        self._blocksize = blocksize
        self._callback = callback
        self._loop = loop

        self._audio: np.ndarray = self._load_wav()
        self._pos = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None

        logger.info(
            "[WavInput] Loaded %s: %.2fs, %d samples, sr=%d, blocksize=%d",
            self._path.name,
            len(self._audio) / self._sample_rate,
            len(self._audio),
            self._sample_rate,
            self._blocksize,
        )

    def _load_wav(self) -> np.ndarray:
        """Load WAV file and convert to float32 mono at target sample rate."""
        import scipy.io.wavfile as wavfile
        from rcwx.audio.resample import resample

        file_sr, data = wavfile.read(str(self._path))

        # Convert to float32
        if data.dtype == np.int16:
            audio = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            audio = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.float32:
            audio = data.copy()
        elif data.dtype == np.float64:
            audio = data.astype(np.float32)
        else:
            audio = data.astype(np.float32)

        # Convert to mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1).astype(np.float32)

        # Resample to target sample rate
        if file_sr != self._sample_rate:
            audio = resample(audio, file_sr, self._sample_rate)

        return audio

    @property
    def actual_sample_rate(self) -> int:
        return self._sample_rate

    def start(self) -> None:
        if self._running:
            return
        self._pos = 0
        self._running = True
        self._thread = threading.Thread(
            target=self._playback_thread,
            daemon=True,
            name="RCWX-WavInput",
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _playback_thread(self) -> None:
        interval = self._blocksize / self._sample_rate
        total = len(self._audio)

        logger.info("[WavInput] Playback thread started (interval=%.3fs)", interval)

        while self._running:
            t0 = time.perf_counter()

            end = self._pos + self._blocksize
            if end <= total:
                chunk = self._audio[self._pos:end].copy()
                self._pos = end
            else:
                # Wrap around
                remaining = total - self._pos
                chunk = np.empty(self._blocksize, dtype=np.float32)
                chunk[:remaining] = self._audio[self._pos:]
                if self._loop:
                    self._pos = 0
                    filled = remaining
                    while filled < self._blocksize:
                        take = min(self._blocksize - filled, total)
                        chunk[filled:filled + take] = self._audio[:take]
                        self._pos = take
                        filled += take
                else:
                    chunk[remaining:] = 0.0
                    self._running = False

            if self._callback is not None:
                self._callback(chunk)

            elapsed = time.perf_counter() - t0
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.info("[WavInput] Playback thread stopped")
