"""Chunk buffer with crossfade for seamless audio processing."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class ChunkBuffer:
    """
    Audio chunk buffer with crossfade support.

    Manages input buffering and output crossfading for real-time processing.

    w-okada style processing:
    - Each chunk includes left_context (from previous) and right_context (lookahead)
    - Structure: [left_context | main | right_context]
    - Model processes with both contexts for higher quality output
    - Output is trimmed to keep only the "main" portion
    """

    def __init__(
        self,
        chunk_samples: int,
        crossfade_samples: int,
        context_samples: int = 0,
        lookahead_samples: int = 0,
    ):
        """
        Initialize the chunk buffer.

        Args:
            chunk_samples: Number of samples per processing chunk (main + left_context)
            crossfade_samples: Number of samples for crossfade overlap
            context_samples: Left context samples for inference (overlap with previous chunk)
            lookahead_samples: Right context samples for inference (requires future samples)
        """
        self.chunk_samples = chunk_samples
        self.crossfade_samples = crossfade_samples
        self.context_samples = context_samples
        self.lookahead_samples = lookahead_samples

        # Input buffer for accumulating samples
        self._input_buffer: NDArray[np.float32] = np.array([], dtype=np.float32)

        # Previous chunk output for crossfading
        self._prev_output: NDArray[np.float32] | None = None

        # Track if this is the first chunk (no left context for first chunk)
        self._is_first_chunk: bool = True

        # Create crossfade windows
        if crossfade_samples > 0:
            self._fade_in = np.linspace(0, 1, crossfade_samples, dtype=np.float32)
            self._fade_out = np.linspace(1, 0, crossfade_samples, dtype=np.float32)
        else:
            self._fade_in = np.array([], dtype=np.float32)
            self._fade_out = np.array([], dtype=np.float32)

    def add_input(self, audio: NDArray[np.float32]) -> None:
        """
        Add audio samples to the input buffer.

        Args:
            audio: Audio samples to add
        """
        self._input_buffer = np.concatenate([self._input_buffer, audio])

    def has_chunk(self) -> bool:
        """Check if a full chunk is available for processing.

        First chunk starts immediately with just main portion (no context).
        Subsequent chunks require context from previous chunk.
        """
        if self._is_first_chunk:
            # First chunk: start immediately with just main + lookahead
            required = self.chunk_samples + self.lookahead_samples
        else:
            # Subsequent chunks: need context + main + lookahead
            required = self.chunk_samples + self.context_samples + self.lookahead_samples
        return len(self._input_buffer) >= required

    def get_chunk(self) -> NDArray[np.float32] | None:
        """
        Get a chunk for processing (w-okada style).

        Returns:
            Audio chunk for inference
            First chunk: [main] (no left context)
            Subsequent chunks: [context | main] (context from previous chunk)

        Note:
            Always advances by chunk_samples (uniform progression)
            w-okada style: extraConvertSize (context) is processed but trimmed from output
        """
        # Determine required samples based on whether this is first chunk
        if self._is_first_chunk:
            # First chunk: just main (+ optional lookahead)
            required = self.chunk_samples + self.lookahead_samples
        else:
            # Subsequent chunks: context + main (+ optional lookahead)
            required = self.chunk_samples + self.context_samples + self.lookahead_samples

        if len(self._input_buffer) < required:
            return None

        # Extract chunk
        chunk = self._input_buffer[:required].copy()

        # Always advance by chunk_samples (w-okada style: uniform progression)
        # This is the key difference from overlap-based chunking
        self._input_buffer = self._input_buffer[self.chunk_samples :]

        # Update flag after first chunk processed
        if self._is_first_chunk:
            self._is_first_chunk = False

        return chunk

    def apply_crossfade(self, output: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Apply crossfade to output for seamless playback.

        Args:
            output: Processed audio chunk

        Returns:
            Crossfaded audio chunk
        """
        if self.crossfade_samples == 0 or self._prev_output is None:
            self._prev_output = output
            return output

        result = output.copy()

        # Apply crossfade at the beginning
        if len(self._prev_output) >= self.crossfade_samples:
            prev_tail = self._prev_output[-self.crossfade_samples :]
            curr_head = result[: self.crossfade_samples]

            # Crossfade: fade out previous + fade in current
            crossfaded = prev_tail * self._fade_out + curr_head * self._fade_in
            result[: self.crossfade_samples] = crossfaded

        self._prev_output = output
        return result

    def clear(self) -> None:
        """Clear all buffers."""
        self._input_buffer = np.array([], dtype=np.float32)
        self._prev_output = None
        self._is_first_chunk = True

    @property
    def buffered_samples(self) -> int:
        """Return the number of samples currently buffered."""
        return len(self._input_buffer)


class OutputBuffer:
    """
    Output buffer for managing processed audio playback.

    Handles buffering and smooth output when processing is slower than real-time.
    """

    def __init__(self, max_latency_samples: int, fade_samples: int = 256):
        """
        Initialize output buffer.

        Args:
            max_latency_samples: Maximum samples to buffer before dropping old samples
            fade_samples: Number of samples for fade in/out on underrun
        """
        self.max_latency_samples = max_latency_samples
        self.fade_samples = fade_samples
        self._buffer: NDArray[np.float32] = np.array([], dtype=np.float32)
        self._last_was_underrun = False
        self._samples_dropped = 0

    def add(self, audio: NDArray[np.float32]) -> int:
        """Add processed audio to the buffer.

        Returns:
            Number of old samples dropped to maintain max latency (0 if none)
        """
        self._buffer = np.concatenate([self._buffer, audio])

        # Drop OLD samples if buffer exceeds max latency
        # This keeps playback close to real-time by catching up
        dropped = 0
        if len(self._buffer) > self.max_latency_samples:
            dropped = len(self._buffer) - self.max_latency_samples
            self._buffer = self._buffer[dropped:]
            self._samples_dropped += dropped

        return dropped

    def get(self, samples: int) -> NDArray[np.float32]:
        """
        Get samples for playback.

        Args:
            samples: Number of samples to get

        Returns:
            Audio samples (zero-padded if not enough available)
        """
        if len(self._buffer) >= samples:
            result = self._buffer[:samples].copy()
            self._buffer = self._buffer[samples:]

            # Apply fade-in if recovering from underrun
            if self._last_was_underrun and self.fade_samples > 0:
                fade_len = min(self.fade_samples, len(result))
                fade_in = np.linspace(0, 1, fade_len, dtype=np.float32)
                result[:fade_len] *= fade_in
                self._last_was_underrun = False

            return result
        else:
            # Not enough samples - return what we have + silence
            result = np.zeros(samples, dtype=np.float32)
            available = len(self._buffer)
            if available > 0:
                # Apply fade-out to available samples to prevent click
                if self.fade_samples > 0 and available > self.fade_samples:
                    fade_out = np.linspace(1, 0, self.fade_samples, dtype=np.float32)
                    self._buffer[-self.fade_samples :] *= fade_out
                result[:available] = self._buffer
                self._buffer = np.array([], dtype=np.float32)
            self._last_was_underrun = True
            return result

    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer = np.array([], dtype=np.float32)
        self._last_was_underrun = False
        self._samples_dropped = 0

    @property
    def available(self) -> int:
        """Return available samples in buffer."""
        return len(self._buffer)

    @property
    def samples_dropped(self) -> int:
        """Return total samples dropped to maintain max latency."""
        return self._samples_dropped

    def set_max_latency(self, max_latency_samples: int) -> None:
        """Update the maximum latency buffer size.

        Args:
            max_latency_samples: New maximum samples to buffer
        """
        self.max_latency_samples = max_latency_samples
        # Immediately trim buffer if it exceeds new max
        if len(self._buffer) > self.max_latency_samples:
            dropped = len(self._buffer) - self.max_latency_samples
            self._buffer = self._buffer[dropped:]
            self._samples_dropped += dropped
