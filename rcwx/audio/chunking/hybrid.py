"""Hybrid chunking strategy (RVC hop + w-okada context).

Hybrid style processing:
- RVC-style hop-based progression (no overlap between main portions)
- w-okada style context structure [left_context | main | right_context]
- Context used for quality but trimmed from output
- SOLA optimized for stitching at chunk boundaries
"""

from __future__ import annotations

from .context_based import ContextBasedChunkingStrategy
from .protocol import ChunkConfig


class HybridChunkingStrategy(ContextBasedChunkingStrategy):
    """Hybrid chunking combining RVC hop with w-okada context structure.

    Key characteristics:
    - Uses ChunkBuffer internally (same as wokada)
    - Expected input: context + main + lookahead
    - hop = chunk_samples (no overlap, like RVC)
    - Context provides quality but is trimmed from output
    - SOLA applied at chunk boundaries for smooth stitching

    Difference from wokada:
    - Primarily semantic: hybrid emphasizes RVC-style processing
    - Same underlying buffer structure
    - Different SOLA behavior (stitching mode vs context-based)

    See ContextBasedChunkingStrategy for implementation details.
    """

    def __init__(self, config: ChunkConfig, align_to_hubert: bool = True):
        """Initialize hybrid chunking strategy.

        Args:
            config: Chunking configuration
            align_to_hubert: Whether to round chunk size to HuBERT hop alignment
        """
        super().__init__(config, align_to_hubert)

    @property
    def mode_name(self) -> str:
        """Return the name of this chunking mode."""
        return "hybrid"
