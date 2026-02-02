"""Base class for chunking strategies with common utilities."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .protocol import ChunkConfig, ChunkResult

logger = logging.getLogger(__name__)


# HuBERT hop size at 16kHz (320 samples = 20ms)
HUBERT_HOP_16K = 320


def round_to_hubert_hop(samples: int, mic_sample_rate: int) -> int:
    """Round sample count to align with HuBERT hop for stable output length.

    HuBERT operates at 16kHz with a hop of 320 samples (20ms).
    For proper frame alignment, chunk sizes should be multiples of this hop
    when converted to the processing rate.

    Args:
        samples: Number of samples at mic sample rate
        mic_sample_rate: Microphone sample rate (e.g., 48000)

    Returns:
        Rounded sample count that aligns with HuBERT hop boundaries
    """
    # HuBERT hop at mic sample rate
    hubert_hop_mic = int(HUBERT_HOP_16K * mic_sample_rate / 16000)

    # Round up to nearest hop multiple
    rounded = ((samples + hubert_hop_mic - 1) // hubert_hop_mic) * hubert_hop_mic

    if rounded != samples:
        logger.debug(
            f"Chunk size adjusted: {samples} -> {rounded} samples "
            f"to align with HuBERT hop ({hubert_hop_mic} samples)"
        )

    return rounded


class BaseChunkingStrategy(ABC):
    """Base class for chunking strategies.

    Provides common functionality:
    - Configuration management
    - HuBERT hop rounding
    - Logging utilities

    Subclasses must implement:
    - add_input()
    - has_chunk()
    - get_chunk()
    - clear()
    """

    def __init__(self, config: ChunkConfig, align_to_hubert: bool = True):
        """Initialize base chunking strategy.

        Args:
            config: Chunking configuration
            align_to_hubert: Whether to round chunk size to HuBERT hop alignment
        """
        self._config = config

        # Calculate chunk samples with optional HuBERT alignment
        self._chunk_samples = config.chunk_samples
        if align_to_hubert:
            self._chunk_samples = round_to_hubert_hop(
                self._chunk_samples, config.mic_sample_rate
            )

        self._context_samples = config.context_samples
        self._lookahead_samples = config.lookahead_samples

        # Track chunk count for first-chunk detection
        self._chunks_processed = 0

        logger.debug(
            f"[{self.mode_name}] Initialized: "
            f"chunk={self._chunk_samples}, context={self._context_samples}, "
            f"lookahead={self._lookahead_samples}"
        )

    @property
    @abstractmethod
    def mode_name(self) -> str:
        """Return the name of this chunking mode."""
        ...

    @property
    def config(self) -> ChunkConfig:
        """Return the chunking configuration."""
        return self._config

    @property
    def chunk_samples(self) -> int:
        """Return the main chunk size in samples (may be HuBERT-aligned)."""
        return self._chunk_samples

    @property
    def context_samples(self) -> int:
        """Return the context size in samples."""
        return self._context_samples

    @property
    def lookahead_samples(self) -> int:
        """Return the lookahead size in samples."""
        return self._lookahead_samples

    @property
    @abstractmethod
    def hop_samples(self) -> int:
        """Return the hop size (advance per chunk) in samples."""
        ...

    @property
    @abstractmethod
    def buffered_samples(self) -> int:
        """Return the number of samples currently buffered."""
        ...

    @abstractmethod
    def add_input(self, audio: NDArray[np.float32]) -> None:
        """Add audio samples to the input buffer."""
        ...

    @abstractmethod
    def has_chunk(self) -> bool:
        """Check if a full chunk is available for processing."""
        ...

    @abstractmethod
    def get_chunk(self) -> Optional[ChunkResult]:
        """Get the next chunk for processing."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clear all internal buffers and reset state."""
        ...

    @abstractmethod
    def get_expected_input_samples(self, is_first: bool) -> int:
        """Get the expected number of input samples for a chunk."""
        ...

    def _is_first_chunk(self) -> bool:
        """Check if the next chunk would be the first one."""
        return self._chunks_processed == 0

    def _increment_chunk_count(self) -> None:
        """Increment the processed chunk counter."""
        self._chunks_processed += 1

    def _reset_chunk_count(self) -> None:
        """Reset the processed chunk counter."""
        self._chunks_processed = 0
