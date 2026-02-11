"""Ring output buffer for real-time audio playback."""

from __future__ import annotations

import logging
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class RingOutputBuffer:
    """Pre-allocated circular output buffer with O(1) read/write.

    Uses a fixed-size numpy array to avoid memory allocation during
    real-time audio playback. Supports fade-in on underrun recovery
    and fade-out when buffer empties.
    """

    def __init__(self, capacity_samples: int, fade_samples: int = 256):
        """Initialize ring output buffer.

        Args:
            capacity_samples: Total buffer capacity in samples (pre-allocated).
            fade_samples: Fade length for underrun recovery (default 256).
        """
        self.capacity = capacity_samples
        self.fade_samples = fade_samples

        self._buf = np.zeros(capacity_samples, dtype=np.float32)
        self._read_pos = 0
        self._write_pos = 0
        self._count = 0  # samples currently in buffer

        self._last_was_underrun = False
        self._samples_dropped = 0
        self._underrun_count = 0

    def add(self, audio: NDArray[np.float32]) -> int:
        """Write audio into the ring buffer.

        If buffer is full, drops oldest samples to make room.

        Args:
            audio: Audio samples to write.

        Returns:
            Number of old samples dropped (0 if none).
        """
        n = len(audio)
        if n == 0:
            return 0

        dropped = 0
        # Drop oldest if we'd overflow
        if self._count + n > self.capacity:
            overflow = self._count + n - self.capacity
            self._read_pos = (self._read_pos + overflow) % self.capacity
            self._count -= overflow
            dropped = overflow
            self._samples_dropped += dropped

        # Write into ring buffer (may wrap around)
        end = self._write_pos + n
        if end <= self.capacity:
            self._buf[self._write_pos:end] = audio
        else:
            first = self.capacity - self._write_pos
            self._buf[self._write_pos:] = audio[:first]
            self._buf[:n - first] = audio[first:]

        self._write_pos = (self._write_pos + n) % self.capacity
        self._count += n

        return dropped

    def get(self, samples: int) -> NDArray[np.float32]:
        """Read samples from the ring buffer.

        If not enough samples available, returns what's available
        with fade-out, padded with zeros (underrun).

        Args:
            samples: Number of samples requested.

        Returns:
            Audio array of exactly `samples` length.
        """
        if self._count >= samples:
            result = self._read(samples)

            # Apply fade-in if recovering from underrun
            if self._last_was_underrun and self.fade_samples > 0:
                fade_len = min(self.fade_samples, len(result))
                fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
                result[:fade_len] *= fade_in
                self._last_was_underrun = False

            return result
        else:
            # Underrun — return available audio with fade-out + silence
            result = np.zeros(samples, dtype=np.float32)
            avail = self._count
            if avail > 0:
                chunk = self._read(avail)
                # Apply fade-out to prevent click
                if self.fade_samples > 0 and avail > self.fade_samples:
                    fade_out = np.linspace(1.0, 0.0, self.fade_samples, dtype=np.float32)
                    chunk[-self.fade_samples:] *= fade_out
                result[:avail] = chunk
            self._last_was_underrun = True
            self._underrun_count += 1
            return result

    def _read(self, n: int) -> NDArray[np.float32]:
        """Read n samples from ring buffer (internal, assumes n <= _count)."""
        end = self._read_pos + n
        if end <= self.capacity:
            result = self._buf[self._read_pos:end].copy()
        else:
            first = self.capacity - self._read_pos
            result = np.concatenate([
                self._buf[self._read_pos:],
                self._buf[:n - first],
            ])
        self._read_pos = (self._read_pos + n) % self.capacity
        self._count -= n
        return result

    def clear(self) -> None:
        """Reset buffer to empty state (no reallocation)."""
        self._read_pos = 0
        self._write_pos = 0
        self._count = 0
        self._last_was_underrun = False
        self._samples_dropped = 0
        self._underrun_count = 0

    @property
    def available(self) -> int:
        """Samples currently available for reading."""
        return self._count

    @property
    def free(self) -> int:
        """Free space in samples."""
        return self.capacity - self._count

    @property
    def samples_dropped(self) -> int:
        """Total samples dropped due to overflow."""
        return self._samples_dropped

    @property
    def underrun_count(self) -> int:
        """Total underrun events."""
        return self._underrun_count
