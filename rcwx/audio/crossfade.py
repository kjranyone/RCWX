"""Crossfade utilities for chunk-based audio processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class SOLAState:
    """State for RVC-style SOLA crossfade processing."""

    sola_buffer: Optional[NDArray[np.float32]] = None
    sola_buffer_frame: int = 0
    sola_search_frame: int = 0
    fade_in_window: Optional[NDArray[np.float32]] = None
    fade_out_window: Optional[NDArray[np.float32]] = None
    frames_processed: int = 0

    @staticmethod
    def create(crossfade_samples: int, sample_rate: int = 48000) -> "SOLAState":
        """
        Create SOLA state with proper window initialization.

        Args:
            crossfade_samples: Crossfade length in samples
            sample_rate: Audio sample rate

        Returns:
            Initialized SOLAState
        """
        # Estimate zero-crossing interval (assume ~100Hz fundamental for voice)
        zc = sample_rate // 100  # ~480 samples at 48kHz

        sola_buffer_frame = min(crossfade_samples, 4 * zc)
        sola_search_frame = zc

        # Sin^2 fade windows (as used in RVC)
        t = np.linspace(0.0, 1.0, sola_buffer_frame, dtype=np.float32)
        fade_in = np.sin(0.5 * np.pi * t) ** 2
        fade_out = 1 - fade_in

        return SOLAState(
            sola_buffer=None,
            sola_buffer_frame=sola_buffer_frame,
            sola_search_frame=sola_search_frame,
            fade_in_window=fade_in,
            fade_out_window=fade_out,
        )


@dataclass
class CrossfadeResult:
    """Result of crossfade operation."""

    audio: NDArray[np.float32]
    sola_offset: int = 0
    sola_correlation: float = 0.0


def flush_sola_buffer(state: SOLAState) -> NDArray[np.float32]:
    """
    Flush remaining SOLA buffer at the end of processing.

    This outputs the final buffer that was saved from the last chunk.
    Should be called after all chunks have been processed.

    Args:
        state: SOLA state

    Returns:
        Remaining buffer audio, or empty array if no buffer
    """
    if state.sola_buffer is not None:
        buffer = state.sola_buffer.copy()
        state.sola_buffer = None
        return buffer
    return np.array([], dtype=np.float32)


def apply_sola_crossfade(
    infer_wav: NDArray[np.float32],
    state: SOLAState,
) -> CrossfadeResult:
    """
    Apply RVC-style SOLA crossfade.

    Based on RVC WebUI implementation. Each chunk outputs its main portion
    WITHOUT the tail (sola_buffer region). The tail is saved and crossfaded
    with the beginning of the next chunk, ensuring smooth boundaries.

    Flow:
    - Chunk 0: output[:-buffer], save buffer
    - Chunk N: crossfade(buffer, chunk), output[:-buffer], save new buffer
    - Final: call flush_sola_buffer() to output remaining buffer
    - Boundary: chunk[N-1] ends at result[-buffer-1], chunk[N] starts with
                crossfade of result[-buffer], which are adjacent samples.

    Args:
        infer_wav: Current inference output
        state: SOLA state (modified in place)

    Returns:
        CrossfadeResult with processed audio
    """
    state.frames_processed += 1

    sola_buffer_frame = state.sola_buffer_frame
    sola_search_frame = state.sola_search_frame

    # First chunk: save buffer, output WITHOUT buffer
    if state.sola_buffer is None:
        if len(infer_wav) > sola_buffer_frame:
            state.sola_buffer = infer_wav[-sola_buffer_frame:].copy()
            return CrossfadeResult(audio=infer_wav[:-sola_buffer_frame].copy())
        return CrossfadeResult(audio=infer_wav.copy())

    # Not enough samples for SOLA search
    if len(infer_wav) < sola_buffer_frame + sola_search_frame:
        # Fallback: simple crossfade without offset search
        if len(infer_wav) >= sola_buffer_frame:
            result = infer_wav.copy()
            result[:sola_buffer_frame] = (
                state.sola_buffer * state.fade_out_window +
                result[:sola_buffer_frame] * state.fade_in_window
            )
            state.sola_buffer = result[-sola_buffer_frame:].copy()
        else:
            result = infer_wav.copy()
            state.sola_buffer = None
        return CrossfadeResult(audio=result)

    # Find optimal offset using correlation
    sola_offset = _find_sola_offset(
        state.sola_buffer,
        infer_wav,
        sola_search_frame,
    )

    # Shift infer_wav by offset (RVC style)
    result = infer_wav[sola_offset:].copy()

    # Apply crossfade in-place
    if len(result) >= 2 * sola_buffer_frame:
        # Crossfade the beginning with previous buffer
        result[:sola_buffer_frame] = (
            state.sola_buffer * state.fade_out_window +
            result[:sola_buffer_frame] * state.fade_in_window
        )
        # Save new buffer (tail of current result)
        state.sola_buffer = result[-sola_buffer_frame:].copy()
        # Output WITHOUT tail (tail will appear via crossfade in next chunk)
        return CrossfadeResult(audio=result[:-sola_buffer_frame], sola_offset=sola_offset)
    elif len(result) >= sola_buffer_frame:
        # Short result: crossfade but can't save buffer
        result[:sola_buffer_frame] = (
            state.sola_buffer * state.fade_out_window +
            result[:sola_buffer_frame] * state.fade_in_window
        )
        state.sola_buffer = None
        return CrossfadeResult(audio=result, sola_offset=sola_offset)
    else:
        state.sola_buffer = None
        return CrossfadeResult(audio=result, sola_offset=sola_offset)


def _find_sola_offset(
    sola_buffer: NDArray[np.float32],
    infer_wav: NDArray[np.float32],
    sola_search_frame: int,
) -> int:
    """
    Find optimal SOLA offset using normalized cross-correlation.

    Args:
        sola_buffer: Previous chunk's tail
        infer_wav: Current inference output
        sola_search_frame: Search range in samples

    Returns:
        Optimal offset for alignment
    """
    sola_buffer_frame = len(sola_buffer)

    if len(infer_wav) < sola_buffer_frame + sola_search_frame:
        return 0

    # Search region of current output
    search_region = infer_wav[:sola_buffer_frame + sola_search_frame]

    # Compute correlation using sliding window
    best_offset = 0
    best_corr = -np.inf

    for offset in range(sola_search_frame + 1):
        window = search_region[offset:offset + sola_buffer_frame]

        if len(window) != sola_buffer_frame:
            continue

        # Normalized correlation
        nom = np.sum(sola_buffer * window)
        den = np.sqrt(np.sum(window ** 2) + 1e-8)

        corr = nom / den
        if corr > best_corr:
            best_corr = corr
            best_offset = offset

    return best_offset
