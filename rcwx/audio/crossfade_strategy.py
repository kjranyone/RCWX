"""Crossfade strategies for chunk boundary processing."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .crossfade import CrossfadeResult, SOLAState, apply_sola_crossfade, flush_sola_buffer

logger = logging.getLogger(__name__)


@dataclass
class CrossfadeConfig:
    """Configuration for crossfade strategies."""

    # Output sample rate
    output_sample_rate: int = 48000

    # Crossfade duration in seconds
    crossfade_sec: float = 0.05

    # Context duration in seconds (for wokada/hybrid modes)
    context_sec: float = 0.10

    # RVC WebUI overlap duration in seconds
    rvc_overlap_sec: float = 0.22

    # Chunk duration in seconds (for output length calculation)
    chunk_sec: float = 0.10

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.output_sample_rate <= 0:
            raise ValueError(
                f"output_sample_rate must be positive, got {self.output_sample_rate}"
            )
        if self.crossfade_sec < 0:
            raise ValueError(
                f"crossfade_sec must be non-negative, got {self.crossfade_sec}"
            )
        if self.context_sec < 0:
            raise ValueError(
                f"context_sec must be non-negative, got {self.context_sec}"
            )
        if self.rvc_overlap_sec < 0:
            raise ValueError(
                f"rvc_overlap_sec must be non-negative, got {self.rvc_overlap_sec}"
            )
        if self.chunk_sec <= 0:
            raise ValueError(
                f"chunk_sec must be positive, got {self.chunk_sec}"
            )
        # crossfade should not exceed chunk duration
        if self.crossfade_sec > self.chunk_sec:
            raise ValueError(
                f"crossfade_sec ({self.crossfade_sec}) cannot exceed "
                f"chunk_sec ({self.chunk_sec})"
            )

    @property
    def crossfade_samples(self) -> int:
        """Crossfade size in samples."""
        return int(self.output_sample_rate * self.crossfade_sec)

    @property
    def context_samples(self) -> int:
        """Context size in samples at output rate."""
        return int(self.output_sample_rate * self.context_sec)

    @property
    def overlap_samples(self) -> int:
        """RVC overlap size in samples at output rate."""
        return int(self.output_sample_rate * self.rvc_overlap_sec)


@runtime_checkable
class CrossfadeStrategy(Protocol):
    """Protocol for crossfade strategies.

    A crossfade strategy manages:
    1. Smooth transitions between chunks
    2. Context trimming (for wokada/hybrid modes)
    3. Output length normalization
    """

    @property
    def mode_name(self) -> str:
        """Return the name of this crossfade mode."""
        ...

    def process(
        self,
        output: NDArray[np.float32],
        chunk_index: int,
    ) -> CrossfadeResult:
        """Process output chunk with crossfade.

        Args:
            output: Inference output audio
            chunk_index: Current chunk index (0 = first)

        Returns:
            CrossfadeResult with processed audio
        """
        ...

    def flush(self) -> NDArray[np.float32]:
        """Flush any remaining buffered audio.

        Should be called after all chunks are processed.

        Returns:
            Remaining audio, or empty array
        """
        ...

    def reset(self) -> None:
        """Reset internal state for a new stream."""
        ...


class BaseCrossfadeStrategy(ABC):
    """Base class for crossfade strategies."""

    def __init__(self, config: CrossfadeConfig):
        """Initialize crossfade strategy.

        Args:
            config: Crossfade configuration
        """
        self._config = config
        self._chunks_processed = 0

    @property
    @abstractmethod
    def mode_name(self) -> str:
        """Return the name of this crossfade mode."""
        ...

    @property
    def config(self) -> CrossfadeConfig:
        """Return the crossfade configuration."""
        return self._config

    @abstractmethod
    def process(
        self,
        output: NDArray[np.float32],
        chunk_index: int,
    ) -> CrossfadeResult:
        """Process output chunk with crossfade."""
        ...

    @abstractmethod
    def flush(self) -> NDArray[np.float32]:
        """Flush any remaining buffered audio."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state for a new stream."""
        ...


class SOLAWokadaCrossfade(BaseCrossfadeStrategy):
    """SOLA crossfade for w-okada/hybrid modes.

    Uses left context for phase-aligned crossfading.
    Trims context from output.
    """

    def __init__(self, config: CrossfadeConfig):
        """Initialize SOLA w-okada crossfade.

        Args:
            config: Crossfade configuration
        """
        super().__init__(config)
        # w-okada mode: use crossfade_samples for SOLA buffer/search
        self._sola_state = SOLAState.create(
            config.crossfade_samples,
            config.output_sample_rate,
        )

    @property
    def mode_name(self) -> str:
        """Return the name of this crossfade mode."""
        return "sola_wokada"

    def process(
        self,
        output: NDArray[np.float32],
        chunk_index: int,
    ) -> CrossfadeResult:
        """Process output with SOLA crossfade (w-okada mode).

        Args:
            output: Inference output (includes context)
            chunk_index: Current chunk index (0 = first)

        Returns:
            CrossfadeResult with context trimmed and SOLA applied
        """
        # Calculate context to trim (skip for first chunk)
        context_samples = 0
        if chunk_index > 0 and self._config.context_sec > 0:
            context_samples = self._config.context_samples

        result = apply_sola_crossfade(
            output,
            self._sola_state,
            wokada_mode=True,
            context_samples=context_samples,
        )

        self._chunks_processed += 1

        if chunk_index < 5 or chunk_index % 50 == 0:
            logger.debug(
                f"[SOLA-wokada] chunk={chunk_index}, "
                f"offset={result.sola_offset}, corr={result.correlation:.4f}, "
                f"out_len={len(result.audio)}"
            )

        return result

    def flush(self) -> NDArray[np.float32]:
        """Flush remaining SOLA buffer."""
        return flush_sola_buffer(self._sola_state)

    def reset(self) -> None:
        """Reset SOLA state for a new stream."""
        self._sola_state = SOLAState.create(
            self._config.crossfade_samples,
            self._config.output_sample_rate,
        )
        self._sola_state.fade_out_window = 1.0 - self._sola_state.fade_in_window
        self._chunks_processed = 0


class SOLARVCWebUICrossfade(BaseCrossfadeStrategy):
    """SOLA crossfade for RVC WebUI mode.

    Uses overlap-based crossfading.
    Trims output to hop length.
    """

    def __init__(self, config: CrossfadeConfig):
        """Initialize SOLA RVC WebUI crossfade.

        Args:
            config: Crossfade configuration
        """
        super().__init__(config)
        self._sola_state = SOLAState.create(
            config.crossfade_samples,
            config.output_sample_rate,
        )

        # Calculate hop for output trimming
        chunk_samples = int(config.output_sample_rate * config.chunk_sec)
        self._hop_samples = chunk_samples - config.overlap_samples
        self._first_chunk_samples = chunk_samples - config.overlap_samples

    @property
    def mode_name(self) -> str:
        """Return the name of this crossfade mode."""
        return "sola_rvc_webui"

    def process(
        self,
        output: NDArray[np.float32],
        chunk_index: int,
    ) -> CrossfadeResult:
        """Process output with SOLA crossfade (RVC WebUI mode).

        Args:
            output: Inference output
            chunk_index: Current chunk index (0 = first)

        Returns:
            CrossfadeResult with SOLA applied and trimmed to hop length
        """
        # Ensure fixed-length chunk before SOLA (RVC WebUI expects constant chunk size)
        chunk_samples = int(self._config.output_sample_rate * self._config.chunk_sec)
        if len(output) < chunk_samples:
            pad_len = chunk_samples - len(output)
            output = np.pad(output, (0, pad_len), mode="edge")

        # Apply SOLA crossfade
        result = apply_sola_crossfade(
            output,
            self._sola_state,
            wokada_mode=False,
            context_samples=0,
        )

        audio = result.audio

        # Trim to correct length
        if chunk_index == 0:
            # First chunk: output (chunk - overlap) worth
            if len(audio) > self._first_chunk_samples:
                audio = audio[: self._first_chunk_samples]
        else:
            # Subsequent chunks: output hop worth
            if len(audio) > self._hop_samples:
                audio = audio[: self._hop_samples]

        self._chunks_processed += 1

        if chunk_index < 5 or chunk_index % 50 == 0:
            logger.debug(
                f"[SOLA-RVC] chunk={chunk_index}, "
                f"offset={result.sola_offset}, corr={result.correlation:.4f}, "
                f"out_len={len(audio)}, hop={self._hop_samples}"
            )

        return CrossfadeResult(
            audio=audio,
            sola_offset=result.sola_offset,
            correlation=result.correlation,
        )

    def flush(self) -> NDArray[np.float32]:
        """Flush remaining SOLA buffer."""
        return flush_sola_buffer(self._sola_state)

    def reset(self) -> None:
        """Reset SOLA state for a new stream."""
        self._sola_state = SOLAState.create(
            self._config.crossfade_samples,
            self._config.output_sample_rate,
        )
        self._chunks_processed = 0


class ManualTrimCrossfade(BaseCrossfadeStrategy):
    """Manual context trimming without SOLA.

    Simply trims context from output.
    No phase alignment or crossfading.
    """

    def __init__(self, config: CrossfadeConfig):
        """Initialize manual trim crossfade.

        Args:
            config: Crossfade configuration
        """
        super().__init__(config)

    @property
    def mode_name(self) -> str:
        """Return the name of this crossfade mode."""
        return "manual_trim"

    def process(
        self,
        output: NDArray[np.float32],
        chunk_index: int,
    ) -> CrossfadeResult:
        """Process output with manual context trimming.

        Args:
            output: Inference output (may include context)
            chunk_index: Current chunk index (0 = first)

        Returns:
            CrossfadeResult with context trimmed (no SOLA)
        """
        audio = output

        # Trim context (skip for first chunk)
        if chunk_index > 0 and self._config.context_sec > 0:
            context_samples = self._config.context_samples
            if len(audio) > context_samples:
                audio = audio[context_samples:]

                if chunk_index < 5:
                    logger.debug(
                        f"[TRIM] chunk={chunk_index}: trimmed {context_samples} samples"
                    )

        self._chunks_processed += 1

        return CrossfadeResult(
            audio=audio,
            sola_offset=0,
            correlation=0.0,
        )

    def flush(self) -> NDArray[np.float32]:
        """No buffer to flush."""
        return np.array([], dtype=np.float32)

    def reset(self) -> None:
        """Reset state for a new stream."""
        self._chunks_processed = 0


def create_crossfade_strategy(
    chunking_mode: str,
    use_sola: bool,
    config: CrossfadeConfig,
) -> BaseCrossfadeStrategy:
    """Factory function to create a crossfade strategy.

    Args:
        chunking_mode: Chunking mode ("wokada", "hybrid", "rvc_webui")
        use_sola: Whether to use SOLA for crossfading
        config: Crossfade configuration

    Returns:
        Appropriate crossfade strategy instance
    """
    if not use_sola:
        logger.debug("[CrossfadeStrategy] Using ManualTrimCrossfade (SOLA disabled)")
        return ManualTrimCrossfade(config)

    if chunking_mode == "rvc_webui":
        logger.debug("[CrossfadeStrategy] Using SOLARVCWebUICrossfade")
        return SOLARVCWebUICrossfade(config)
    else:
        # wokada and hybrid use the same SOLA strategy
        logger.debug(f"[CrossfadeStrategy] Using SOLAWokadaCrossfade (mode={chunking_mode})")
        return SOLAWokadaCrossfade(config)
