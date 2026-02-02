"""Context-based chunking strategy base class.

Shared implementation for w-okada and hybrid chunking modes.
Both use ChunkBuffer with context structure [left_context | main | right_context].
"""

from __future__ import annotations

import logging
from typing import Optional

from numpy.typing import NDArray
import numpy as np

from rcwx.audio.buffer import ChunkBuffer

from .base import BaseChunkingStrategy
from .protocol import ChunkConfig, ChunkResult

logger = logging.getLogger(__name__)


class ContextBasedChunkingStrategy(BaseChunkingStrategy):
    """Base class for context-based chunking strategies (wokada/hybrid).

    Key characteristics:
    - Uses ChunkBuffer internally
    - Expected input: context + main + lookahead (first chunk: main + lookahead with reflection)
    - hop = chunk_samples (no overlap between main portions)
    - Context provides overlap for quality, but is trimmed from output

    Subclasses only need to override mode_name property.
    """

    def __init__(self, config: ChunkConfig, align_to_hubert: bool = True):
        """Initialize context-based chunking strategy.

        Args:
            config: Chunking configuration
            align_to_hubert: Whether to round chunk size to HuBERT hop alignment
        """
        super().__init__(config, align_to_hubert)

        # Create internal ChunkBuffer
        self._buffer = ChunkBuffer(
            chunk_samples=self._chunk_samples,
            crossfade_samples=0,  # Crossfade handled by CrossfadeStrategy
            context_samples=self._context_samples,
            lookahead_samples=self._lookahead_samples,
        )

        logger.debug(
            f"[{self.mode_name}] Created with ChunkBuffer: "
            f"chunk={self._chunk_samples}, context={self._context_samples}, "
            f"lookahead={self._lookahead_samples}"
        )

    @property
    def hop_samples(self) -> int:
        """Return the hop size (advance per chunk) in samples.

        In context-based modes, hop equals chunk_samples (no overlap).
        Context provides overlap for quality but is trimmed from output.
        """
        return self._chunk_samples

    @property
    def buffered_samples(self) -> int:
        """Return the number of samples currently buffered."""
        return self._buffer.buffered_samples

    def add_input(self, audio: NDArray[np.float32]) -> None:
        """Add audio samples to the input buffer.

        Args:
            audio: Audio samples to add (at mic sample rate)
        """
        self._buffer.add_input(audio)

    def has_chunk(self) -> bool:
        """Check if a full chunk is available for processing.

        Returns:
            True if a chunk can be extracted
        """
        return self._buffer.has_chunk()

    def get_chunk(self) -> Optional[ChunkResult]:
        """Get the next chunk for processing.

        Returns:
            ChunkResult with the audio chunk, or None if not enough samples

        Note:
            First chunk: [reflection_padding | main | lookahead]
            Subsequent: [context | main | lookahead]
        """
        is_first = self._is_first_chunk()
        chunk = self._buffer.get_chunk()

        if chunk is None:
            return None

        self._increment_chunk_count()

        expected = self.get_expected_input_samples(is_first)

        return ChunkResult(
            chunk=chunk,
            is_first_chunk=is_first,
            expected_input_samples=expected,
        )

    def clear(self) -> None:
        """Clear all internal buffers and reset state."""
        self._buffer.clear()
        self._reset_chunk_count()
        logger.debug(f"[{self.mode_name}] Cleared")

    def get_expected_input_samples(self, is_first: bool) -> int:
        """Get the expected number of input samples for a chunk.

        Args:
            is_first: Whether this is the first chunk

        Returns:
            Expected number of samples

        Note:
            First chunk: chunk + lookahead (reflection padding added internally)
            Subsequent: context + chunk + lookahead
        """
        if is_first:
            # First chunk has reflection padding added, so total includes context
            return self._chunk_samples + self._context_samples + self._lookahead_samples
        else:
            return self._context_samples + self._chunk_samples + self._lookahead_samples
