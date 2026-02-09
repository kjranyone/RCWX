"""Chunk buffer with crossfade for seamless audio processing."""

from __future__ import annotations

import logging
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


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

        logger.info(
            f"[ChunkBuffer] INIT: chunk={chunk_samples}, crossfade={crossfade_samples}, "
            f"context={context_samples}, lookahead={lookahead_samples}"
        )

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
            First chunk: [reflection_padding | main] (reflection padding as left context)
            Subsequent chunks: [context | main] (context from previous chunk)

        Note:
            Always advances by chunk_samples (uniform progression)
            w-okada style: extraConvertSize (context) is processed but trimmed from output
            First chunk uses reflection padding to provide consistent context structure
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

        # Extract chunk with reflection padding for first chunk
        if self._is_first_chunk and self.context_samples > 0:
            # First chunk: use reflection padding for left context
            main_chunk = self._input_buffer[:required].copy()
            # Create reflection padding from the beginning of the main chunk
            reflect_len = min(self.context_samples, len(main_chunk))
            reflection = main_chunk[:reflect_len][::-1].copy()  # Reverse the first samples
            if len(reflection) < self.context_samples:
                # Pad with zeros if not enough samples for reflection
                reflection = np.concatenate([
                    np.zeros(self.context_samples - len(reflection), dtype=np.float32),
                    reflection
                ])
            chunk = np.concatenate([reflection, main_chunk])
            logger.info(
                f"[ChunkBuffer] First chunk WITH context: required={required}, main={len(main_chunk)}, "
                f"reflection={len(reflection)}, total={len(chunk)}"
            )
        else:
            chunk = self._input_buffer[:required].copy()
            if self._is_first_chunk:
                # First chunk but context_samples=0
                logger.info(
                    f"[ChunkBuffer] First chunk NO context: context_samples={self.context_samples}, "
                    f"required={required}, total={len(chunk)}"
                )

        # Advance input buffer
        # w-okada style: For first chunk, advance by (chunk - context) so that
        # the next chunk's context overlaps with this chunk's main ending.
        # For subsequent chunks, advance by chunk_samples (full main portion).
        if self._is_first_chunk:
            # First chunk: advance less to create overlap for next chunk's context
            # This ensures second chunk's context = first chunk's main ending
            advance = self.chunk_samples - self.context_samples
            if advance <= 0:
                logger.warning(
                    "[ChunkBuffer] context_samples >= chunk_samples; "
                    "forcing advance to chunk_samples to avoid repeated chunks"
                )
                advance = self.chunk_samples
            self._input_buffer = self._input_buffer[advance:]
            logger.debug(
                f"[ChunkBuffer] First chunk advance: {advance} samples "
                f"(chunk={self.chunk_samples} - context={self.context_samples})"
            )
            self._is_first_chunk = False
        else:
            # Subsequent chunks: uniform progression by chunk_samples
            self._input_buffer = self._input_buffer[self.chunk_samples:]

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
        if len(audio) > 0:
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
