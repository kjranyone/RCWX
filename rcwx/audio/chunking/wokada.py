"""w-okada style chunking strategy.

w-okada style processing:
- Each chunk includes left_context + main + right_context (lookahead)
- Structure: [left_context | main | right_context]
- Model processes with both contexts for higher quality output
- Output is trimmed to keep only the "main" portion
- First chunk uses reflection padding for left context
"""

from __future__ import annotations

from .context_based import ContextBasedChunkingStrategy
from .protocol import ChunkConfig


class WokadaChunkingStrategy(ContextBasedChunkingStrategy):
    """w-okada style chunking with context-based processing.

    Key characteristics:
    - Uses ChunkBuffer internally
    - Expected input: context + main + lookahead (first chunk: main + lookahead with reflection)
    - hop = chunk_samples (no overlap between main portions)
    - Context provides overlap for quality, but is trimmed from output

    See ContextBasedChunkingStrategy for implementation details.
    """

    def __init__(self, config: ChunkConfig, align_to_hubert: bool = True):
        """Initialize w-okada chunking strategy.

        Args:
            config: Chunking configuration
            align_to_hubert: Whether to round chunk size to HuBERT hop alignment
        """
        super().__init__(config, align_to_hubert)

    @property
    def mode_name(self) -> str:
        """Return the name of this chunking mode."""
        return "wokada"
