"""Protocol definitions for chunking strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@dataclass
class ChunkConfig:
    """Configuration for chunking strategies.

    All sample rates are in Hz, durations in seconds.
    """

    # Sample rate of microphone input
    mic_sample_rate: int = 48000

    # Main chunk duration in seconds
    chunk_sec: float = 0.10

    # Context duration in seconds (w-okada style: left context for quality)
    context_sec: float = 0.10

    # Lookahead duration in seconds (right context, adds latency)
    lookahead_sec: float = 0.0

    # RVC WebUI mode: overlap duration in seconds
    rvc_overlap_sec: float = 0.22

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.chunk_sec <= 0:
            raise ValueError(f"chunk_sec must be positive, got {self.chunk_sec}")
        if self.context_sec < 0:
            raise ValueError(f"context_sec must be non-negative, got {self.context_sec}")
        if self.lookahead_sec < 0:
            raise ValueError(f"lookahead_sec must be non-negative, got {self.lookahead_sec}")
        if self.rvc_overlap_sec < 0:
            raise ValueError(f"rvc_overlap_sec must be non-negative, got {self.rvc_overlap_sec}")

    @property
    def chunk_samples(self) -> int:
        """Main chunk size in samples."""
        return int(self.mic_sample_rate * self.chunk_sec)

    @property
    def context_samples(self) -> int:
        """Context size in samples."""
        return int(self.mic_sample_rate * self.context_sec)

    @property
    def lookahead_samples(self) -> int:
        """Lookahead size in samples."""
        return int(self.mic_sample_rate * self.lookahead_sec)

    @property
    def overlap_samples(self) -> int:
        """RVC WebUI overlap size in samples."""
        return int(self.mic_sample_rate * self.rvc_overlap_sec)


@dataclass
class ChunkResult:
    """Result of getting a chunk from a chunking strategy.

    Attributes:
        chunk: The audio chunk for processing
        is_first_chunk: Whether this is the first chunk (no left context from previous)
        expected_input_samples: Expected number of samples in the chunk
                                (useful for validation in inference)
    """

    chunk: NDArray[np.float32]
    is_first_chunk: bool = False
    expected_input_samples: int = 0


@runtime_checkable
class ChunkingStrategy(Protocol):
    """Protocol for chunking strategies.

    A chunking strategy manages:
    1. Input buffering (accumulating audio samples)
    2. Chunk extraction (getting chunks for inference)
    3. Hop calculation (how much to advance between chunks)
    """

    @property
    def mode_name(self) -> str:
        """Return the name of this chunking mode."""
        ...

    @property
    def chunk_samples(self) -> int:
        """Return the main chunk size in samples."""
        ...

    @property
    def hop_samples(self) -> int:
        """Return the hop size (advance per chunk) in samples."""
        ...

    @property
    def buffered_samples(self) -> int:
        """Return the number of samples currently buffered."""
        ...

    def add_input(self, audio: NDArray[np.float32]) -> None:
        """Add audio samples to the input buffer.

        Args:
            audio: Audio samples to add (at mic sample rate)
        """
        ...

    def has_chunk(self) -> bool:
        """Check if a full chunk is available for processing.

        Returns:
            True if a chunk can be extracted
        """
        ...

    def get_chunk(self) -> Optional[ChunkResult]:
        """Get the next chunk for processing.

        Returns:
            ChunkResult with the audio chunk, or None if not enough samples
        """
        ...

    def clear(self) -> None:
        """Clear all internal buffers and reset state."""
        ...

    def get_expected_input_samples(self, is_first: bool) -> int:
        """Get the expected number of input samples for a chunk.

        This is useful for validation in the inference thread.

        Args:
            is_first: Whether this is the first chunk

        Returns:
            Expected number of samples
        """
        ...
