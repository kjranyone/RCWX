"""Chunking strategies for real-time audio processing.

This module provides Strategy pattern implementations for different
chunking approaches:

- WokadaChunkingStrategy: w-okada style context-based chunking
- RVCWebUIChunkingStrategy: RVC WebUI overlap-based chunking
- HybridChunkingStrategy: Hybrid combining RVC hop with w-okada context

Usage:
    from rcwx.audio.chunking import create_chunking_strategy, ChunkConfig

    config = ChunkConfig(mic_sample_rate=48000, chunk_sec=0.1)
    strategy = create_chunking_strategy("wokada", config)

    strategy.add_input(audio)
    while strategy.has_chunk():
        result = strategy.get_chunk()
        # process result.chunk
"""

from __future__ import annotations

import logging
from typing import Union

from .base import BaseChunkingStrategy, round_to_hubert_hop
from .context_based import ContextBasedChunkingStrategy
from .hybrid import HybridChunkingStrategy
from .protocol import ChunkConfig, ChunkingStrategy, ChunkResult
from .rvc_webui import RVCWebUIChunkingStrategy
from .wokada import WokadaChunkingStrategy

logger = logging.getLogger(__name__)

__all__ = [
    # Protocol and config
    "ChunkingStrategy",
    "ChunkConfig",
    "ChunkResult",
    # Base classes
    "BaseChunkingStrategy",
    "ContextBasedChunkingStrategy",
    # Strategy implementations
    "WokadaChunkingStrategy",
    "RVCWebUIChunkingStrategy",
    "HybridChunkingStrategy",
    # Factory
    "create_chunking_strategy",
    # Utilities
    "round_to_hubert_hop",
]

# Type alias for all concrete strategy types
AnyChunkingStrategy = Union[
    WokadaChunkingStrategy,
    RVCWebUIChunkingStrategy,
    HybridChunkingStrategy,
]


def create_chunking_strategy(
    mode: str,
    config: ChunkConfig,
    align_to_hubert: bool = True,
) -> AnyChunkingStrategy:
    """Factory function to create a chunking strategy.

    Args:
        mode: Chunking mode - "wokada", "rvc_webui", or "hybrid"
        config: Chunking configuration
        align_to_hubert: Whether to align chunk size to HuBERT hop
                        (recommended for wokada/hybrid, optional for rvc_webui)

    Returns:
        Appropriate chunking strategy instance

    Raises:
        ValueError: If mode is not recognized

    Example:
        >>> config = ChunkConfig(mic_sample_rate=48000, chunk_sec=0.1)
        >>> strategy = create_chunking_strategy("wokada", config)
        >>> strategy.mode_name
        'wokada'
    """
    mode = mode.lower()

    if mode == "wokada":
        logger.debug("[ChunkingStrategy] Creating WokadaChunkingStrategy")
        return WokadaChunkingStrategy(config, align_to_hubert=align_to_hubert)

    elif mode == "rvc_webui":
        # RVC WebUI typically doesn't need HuBERT alignment
        if align_to_hubert:
            logger.warning(
                "[ChunkingStrategy] align_to_hubert=True is ignored for rvc_webui mode "
                "(RVC WebUI uses overlap-based chunking, not HuBERT-aligned)"
            )
        logger.debug("[ChunkingStrategy] Creating RVCWebUIChunkingStrategy")
        return RVCWebUIChunkingStrategy(config, align_to_hubert=False)

    elif mode == "hybrid":
        logger.debug("[ChunkingStrategy] Creating HybridChunkingStrategy")
        return HybridChunkingStrategy(config, align_to_hubert=align_to_hubert)

    else:
        raise ValueError(
            f"Unknown chunking mode: '{mode}'. "
            f"Valid modes: 'wokada', 'rvc_webui', 'hybrid'"
        )
