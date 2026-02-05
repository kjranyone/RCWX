"""Crossfade strategies for chunk boundary processing."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .crossfade import (
    CrossfadeResult,
    SOLAState,
    _apply_boundary_rms_smoothing,
    _blend_head_with_prev_tail,
    _declick_head,
    apply_sola_crossfade,
    flush_sola_buffer,
)

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
    Key features:
    - Trims context from output (except first chunk's reflection padding)
    - Uses correlation-based offset search for optimal phase alignment
    - Applies boundary smoothing for low-correlation cases
    """

    def __init__(self, config: CrossfadeConfig):
        """Initialize SOLA w-okada crossfade.

        Args:
            config: Crossfade configuration
        """
        super().__init__(config)
        # Phase 8: Use larger buffer for strong crossfade (80ms minimum)
        min_phase8_buffer = int(config.output_sample_rate * 0.08)  # 80ms
        self._effective_crossfade = max(config.crossfade_samples, min_phase8_buffer)

        # Phase 8: Enable advanced SOLA with higher fallback threshold
        self._sola_state = SOLAState.create(
            self._effective_crossfade,
            config.output_sample_rate,
            use_advanced_sola=True,
            fallback_threshold=0.8,
        )
        # Track previous output tail for boundary checks
        self._prev_tail: Optional[NDArray[np.float32]] = None

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
            output: Inference output (includes context for chunk_index > 0)
            chunk_index: Current chunk index (0 = first)

        Returns:
            CrossfadeResult with context trimmed and SOLA applied
        """
        # Calculate context to trim
        # First chunk: has reflection padding but we trim it like context
        # Subsequent chunks: trim real context from previous chunk
        if self._config.context_sec > 0:
            context_samples = self._config.context_samples
        else:
            context_samples = 0

        result = apply_sola_crossfade(
            output,
            self._sola_state,
            wokada_mode=True,
            context_samples=context_samples,
        )

        audio = result.audio

        # Apply additional boundary smoothing between chunks when SOLA correlation is low
        if self._prev_tail is not None and len(self._prev_tail) > 0 and len(audio) > 0:
            # Check boundary discontinuity
            jump = abs(float(self._prev_tail[-1]) - float(audio[0]))
            if jump > 0.1 and result.correlation < 0.5:
                # Low correlation + large jump: apply smoothing
                smooth_len = min(48, len(audio))
                t = np.linspace(0.0, 1.0, smooth_len, dtype=np.float32)
                fade = 0.5 * (1.0 - np.cos(np.pi * t))
                prev_val = float(self._prev_tail[-1])
                audio = audio.copy()
                audio[:smooth_len] = prev_val * (1.0 - fade) + audio[:smooth_len] * fade

        # Save tail for next boundary check
        if len(audio) >= 64:
            self._prev_tail = audio[-64:].copy()
        else:
            self._prev_tail = audio.copy() if len(audio) > 0 else None

        self._chunks_processed += 1

        if chunk_index < 5 or chunk_index % 50 == 0:
            logger.debug(
                f"[SOLA-wokada] chunk={chunk_index}, "
                f"offset={result.sola_offset}, corr={result.correlation:.4f}, "
                f"out_len={len(audio)}"
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
        # Phase 8: Preserve advanced SOLA settings on reset
        self._sola_state = SOLAState.create(
            self._effective_crossfade,
            self._config.output_sample_rate,
            use_advanced_sola=True,
            fallback_threshold=0.8,
        )
        self._prev_tail = None
        self._chunks_processed = 0


class SOLARVCWebUICrossfade(BaseCrossfadeStrategy):
    """SOLA crossfade for RVC WebUI mode.

    Uses overlap-based crossfading with improved phase alignment.
    Key improvements:
    - Uses larger crossfade buffer based on overlap for better phase search
    - Applies RMS matching with controlled gain limits
    - Trims output to hop length after SOLA processing
    """

    def __init__(self, config: CrossfadeConfig):
        """Initialize SOLA RVC WebUI crossfade.

        Args:
            config: Crossfade configuration
        """
        super().__init__(config)

        # Phase 8: Use larger buffer for strong crossfade (80ms minimum)
        # This ensures enough samples for smooth transitions when correlation is low
        min_phase8_buffer = int(config.output_sample_rate * 0.08)  # 80ms = 9600 samples at 48kHz

        # Prefer using the overlap size as crossfade buffer when reasonable.
        # This aligns SOLA output length with hop size and reduces trimming artifacts.
        effective_crossfade = max(
            min(config.overlap_samples, config.crossfade_samples * 6),
            min_phase8_buffer,  # Phase 8: ensure 80ms minimum
        )

        # Phase 8: Enable advanced SOLA with higher fallback threshold
        # for better handling of RVC's chunk-to-chunk discontinuities
        self._sola_state = SOLAState.create(
            effective_crossfade,
            config.output_sample_rate,
            use_advanced_sola=True,
            fallback_threshold=0.8,  # High threshold to trigger strong crossfade
        )
        # Limit SOLA search to overlap region for RVC WebUI mode
        self._sola_state.sola_search_frame = min(
            self._sola_state.sola_search_frame,
            config.overlap_samples,
        )
        self._effective_crossfade = effective_crossfade

        # Calculate hop for output trimming
        chunk_samples = int(config.output_sample_rate * config.chunk_sec)
        self._chunk_samples = chunk_samples
        self._hop_samples = chunk_samples - config.overlap_samples
        self._first_chunk_samples = chunk_samples - config.overlap_samples

        # Previous output tail for boundary smoothing
        self._prev_tail: Optional[NDArray[np.float32]] = None

        logger.debug(
            f"[SOLARVCWebUICrossfade] Created: "
            f"crossfade={effective_crossfade}, hop={self._hop_samples}, "
            f"overlap={config.overlap_samples}"
        )

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
        # Ensure sufficient length before SOLA
        if len(output) < self._chunk_samples:
            pad_len = self._chunk_samples - len(output)
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
        target_len = self._first_chunk_samples if chunk_index == 0 else self._hop_samples

        if len(audio) > target_len:
            audio = audio[:target_len]
        elif len(audio) < target_len:
            # Pad if too short (shouldn't happen normally)
            audio = np.pad(audio, (0, target_len - len(audio)), mode="edge")

        # Apply boundary smoothing between chunks
        if self._prev_tail is not None and len(self._prev_tail) > 0 and len(audio) > 0:
            # Always apply a short overlap blend at the head
            blend_len = min(64, len(audio), len(self._prev_tail))
            audio = _blend_head_with_prev_tail(audio, self._prev_tail, blend_len)

            jump = abs(float(self._prev_tail[-1]) - float(audio[0]))
            if jump > 0.12 or result.correlation < 0.5:
                # Stronger overlap blend and RMS smoothing for low correlation or large jumps
                strong_len = min(192, len(audio), len(self._prev_tail))
                audio = _blend_head_with_prev_tail(audio, self._prev_tail, strong_len)
                audio = _apply_boundary_rms_smoothing(
                    audio,
                    self._prev_tail,
                    window=min(128, len(audio), len(self._prev_tail)),
                    min_gain=0.8,
                    max_gain=1.2,
                )

        # Save tail for next boundary check
        if len(audio) >= 64:
            self._prev_tail = audio[-64:].copy()
        else:
            self._prev_tail = audio.copy() if len(audio) > 0 else None

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
        # Phase 8: Preserve advanced SOLA settings on reset
        self._sola_state = SOLAState.create(
            self._effective_crossfade,
            self._config.output_sample_rate,
            use_advanced_sola=True,
            fallback_threshold=0.8,
        )
        self._sola_state.sola_search_frame = min(
            self._sola_state.sola_search_frame,
            self._config.overlap_samples,
        )
        self._prev_tail = None
        self._chunks_processed = 0


class SOLAHybridCrossfade(BaseCrossfadeStrategy):
    """SOLA crossfade for hybrid mode.

    Combines the best of wokada and RVC WebUI modes:
    - Uses context-based chunking like wokada (for low latency)
    - Uses larger crossfade buffer like RVC WebUI (for better phase alignment)
    - Applies adaptive boundary smoothing based on audio characteristics
    """

    def __init__(self, config: CrossfadeConfig):
        """Initialize SOLA hybrid crossfade.

        Args:
            config: Crossfade configuration
        """
        super().__init__(config)

        # Phase 8: Use larger buffer for strong crossfade (80ms minimum)
        min_phase8_buffer = int(config.output_sample_rate * 0.08)  # 80ms
        effective_crossfade = max(
            int(config.crossfade_samples * 1.5),
            min_phase8_buffer,
        )

        # Phase 8: Enable advanced SOLA with higher fallback threshold
        self._sola_state = SOLAState.create(
            effective_crossfade,
            config.output_sample_rate,
            use_advanced_sola=True,
            fallback_threshold=0.8,
        )
        self._effective_crossfade = effective_crossfade

        # Track previous output for adaptive smoothing
        self._prev_tail: Optional[NDArray[np.float32]] = None
        self._prev_rms: float = 0.0

        logger.debug(
            f"[SOLAHybridCrossfade] Created: "
            f"effective_crossfade={effective_crossfade}, "
            f"context={config.context_samples}"
        )

    @property
    def mode_name(self) -> str:
        """Return the name of this crossfade mode."""
        return "sola_hybrid"

    def process(
        self,
        output: NDArray[np.float32],
        chunk_index: int,
    ) -> CrossfadeResult:
        """Process output with hybrid SOLA crossfade.

        Args:
            output: Inference output (includes context)
            chunk_index: Current chunk index (0 = first)

        Returns:
            CrossfadeResult with context trimmed and SOLA applied
        """
        # Context handling: same as wokada
        if self._config.context_sec > 0:
            context_samples = self._config.context_samples
        else:
            context_samples = 0

        result = apply_sola_crossfade(
            output,
            self._sola_state,
            wokada_mode=True,
            context_samples=context_samples,
        )

        audio = result.audio

        # Calculate current RMS for adaptive processing
        curr_rms = float(np.sqrt(np.mean(audio**2))) if len(audio) > 0 else 0.0

        # Adaptive boundary smoothing
        if self._prev_tail is not None and len(self._prev_tail) > 0 and len(audio) > 0:
            jump = abs(float(self._prev_tail[-1]) - float(audio[0]))

            # Adaptive threshold based on signal level
            threshold = max(0.05, min(0.15, (curr_rms + self._prev_rms) * 0.5))

            if jump > threshold:
                # Calculate smoothing length based on correlation and signal
                if result.correlation > 0.7:
                    smooth_len = 16  # Good correlation: minimal smoothing
                elif result.correlation > 0.4:
                    smooth_len = 32  # Medium correlation
                else:
                    smooth_len = 64  # Poor correlation: more smoothing

                smooth_len = min(smooth_len, len(audio))
                if smooth_len > 1:
                    t = np.linspace(0.0, 1.0, smooth_len, dtype=np.float32)
                    fade = 0.5 * (1.0 - np.cos(np.pi * t))
                    prev_val = float(self._prev_tail[-1])
                    audio = audio.copy()
                    audio[:smooth_len] = prev_val * (1.0 - fade) + audio[:smooth_len] * fade

        # Save state for next chunk
        if len(audio) >= 64:
            self._prev_tail = audio[-64:].copy()
        else:
            self._prev_tail = audio.copy() if len(audio) > 0 else None
        self._prev_rms = curr_rms

        self._chunks_processed += 1

        if chunk_index < 5 or chunk_index % 50 == 0:
            logger.debug(
                f"[SOLA-hybrid] chunk={chunk_index}, "
                f"offset={result.sola_offset}, corr={result.correlation:.4f}, "
                f"out_len={len(audio)}, rms={curr_rms:.4f}"
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
        # Phase 8: Preserve advanced SOLA settings on reset
        self._sola_state = SOLAState.create(
            self._effective_crossfade,
            self._config.output_sample_rate,
            use_advanced_sola=True,
            fallback_threshold=0.8,
        )
        self._prev_tail = None
        self._prev_rms = 0.0
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

    Strategy selection:
    - wokada: Context-based with standard SOLA (lowest latency)
    - hybrid: Context-based with enhanced SOLA (better quality)
    - rvc_webui: Overlap-based with RVC-style SOLA (highest quality)
    """
    if not use_sola:
        logger.debug("[CrossfadeStrategy] Using ManualTrimCrossfade (SOLA disabled)")
        return ManualTrimCrossfade(config)

    if chunking_mode == "rvc_webui":
        logger.debug("[CrossfadeStrategy] Using SOLARVCWebUICrossfade")
        return SOLARVCWebUICrossfade(config)
    elif chunking_mode == "hybrid":
        logger.debug("[CrossfadeStrategy] Using SOLAHybridCrossfade")
        return SOLAHybridCrossfade(config)
    else:
        # wokada mode: standard SOLA with context
        logger.debug(f"[CrossfadeStrategy] Using SOLAWokadaCrossfade (mode={chunking_mode})")
        return SOLAWokadaCrossfade(config)
