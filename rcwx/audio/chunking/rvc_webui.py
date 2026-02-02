"""RVC WebUI style chunking strategy.

RVC WebUI style processing:
- Simple overlap-based chunking
- Each chunk overlaps with previous by overlap_samples
- hop = chunk_samples - overlap_samples
- SOLA finds optimal crossfade position in overlap region
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .base import BaseChunkingStrategy
from .protocol import ChunkConfig, ChunkResult

logger = logging.getLogger(__name__)


class RVCWebUIChunkingStrategy(BaseChunkingStrategy):
    """RVC WebUI style chunking with overlap-based processing.

    Key characteristics:
    - Uses simple numpy array buffer (no ChunkBuffer)
    - Expected input: chunk_samples only (no context structure)
    - hop = chunk_samples - overlap_samples
    - Overlap provides continuity, SOLA finds optimal crossfade
    """

    def __init__(self, config: ChunkConfig, align_to_hubert: bool = False):
        """Initialize RVC WebUI chunking strategy.

        Args:
            config: Chunking configuration
            align_to_hubert: Whether to round chunk size to HuBERT hop alignment
                            (disabled by default for RVC WebUI mode)
        """
        super().__init__(config, align_to_hubert)

        # Calculate overlap and hop
        self._overlap_samples = config.overlap_samples
        self._hop_samples = self._chunk_samples - self._overlap_samples

        if self._hop_samples <= 0:
            raise ValueError(
                f"Invalid overlap: overlap_samples ({self._overlap_samples}) "
                f">= chunk_samples ({self._chunk_samples})"
            )

        # Simple buffer for accumulating input
        self._buffer: NDArray[np.float32] = np.array([], dtype=np.float32)

        logger.debug(
            f"[RVCWebUIChunkingStrategy] Created: "
            f"chunk={self._chunk_samples}, overlap={self._overlap_samples}, "
            f"hop={self._hop_samples}"
        )

    @property
    def mode_name(self) -> str:
        """Return the name of this chunking mode."""
        return "rvc_webui"

    @property
    def hop_samples(self) -> int:
        """Return the hop size (advance per chunk) in samples.

        In RVC WebUI mode, hop = chunk - overlap.
        """
        return self._hop_samples

    @property
    def overlap_samples(self) -> int:
        """Return the overlap size in samples."""
        return self._overlap_samples

    @property
    def buffered_samples(self) -> int:
        """Return the number of samples currently buffered."""
        return len(self._buffer)

    def add_input(self, audio: NDArray[np.float32]) -> None:
        """Add audio samples to the input buffer.

        Args:
            audio: Audio samples to add (at mic sample rate)
        """
        self._buffer = np.concatenate([self._buffer, audio])

    def has_chunk(self) -> bool:
        """Check if a full chunk is available for processing.

        Returns:
            True if a chunk can be extracted
        """
        return len(self._buffer) >= self._chunk_samples

    def get_chunk(self) -> Optional[ChunkResult]:
        """Get the next chunk for processing.

        Returns:
            ChunkResult with the audio chunk, or None if not enough samples

        Note:
            Each chunk is exactly chunk_samples.
            After extraction, buffer advances by hop_samples (keeps overlap).
        """
        if len(self._buffer) < self._chunk_samples:
            return None

        is_first = self._is_first_chunk()

        # Extract chunk
        chunk = self._buffer[: self._chunk_samples].copy()

        # Advance by hop_samples (keeps overlap for next chunk)
        self._buffer = self._buffer[self._hop_samples:]

        self._increment_chunk_count()

        return ChunkResult(
            chunk=chunk,
            is_first_chunk=is_first,
            expected_input_samples=self._chunk_samples,
        )

    def clear(self) -> None:
        """Clear all internal buffers and reset state."""
        self._buffer = np.array([], dtype=np.float32)
        self._reset_chunk_count()
        logger.debug("[RVCWebUIChunkingStrategy] Cleared")

    def get_expected_input_samples(self, is_first: bool) -> int:
        """Get the expected number of input samples for a chunk.

        Args:
            is_first: Whether this is the first chunk (unused in RVC WebUI mode)

        Returns:
            Expected number of samples (always chunk_samples)
        """
        # RVC WebUI mode: always chunk_samples, no context structure
        return self._chunk_samples
