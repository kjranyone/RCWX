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
    wokada_mode: bool = True,
    context_samples: int = 0,
) -> CrossfadeResult:
    """
    Apply SOLA crossfade for smooth chunk boundaries.

    Two modes supported:
    1. RVC WebUI mode (wokada_mode=False): Original implementation
       - Removes buffer from end of each chunk
       - Requires overlapping input chunks
       - Can cause audio loss with non-overlapping chunks

    2. w-okada mode (wokada_mode=True): Adapted for context-based chunking
       - Uses LEFT context for crossfading with previous buffer
       - Keeps full output (no tail removal)
       - Compatible with w-okada style processing
       - **Default and recommended**

    Flow (w-okada mode):
    - Chunk 0: [main0] -> output [main0], save buffer from end
    - Chunk 1: [context1 | main1] -> crossfade buffer with context1,
               trim context1, output [main1], save new buffer from end

    Flow (RVC WebUI mode):
    - Chunk 0: output[:-buffer], save buffer
    - Chunk N: crossfade(buffer, chunk), output[:-buffer], save new buffer
    - Final: call flush_sola_buffer() to output remaining buffer

    Args:
        infer_wav: Current inference output (includes context in w-okada mode)
        state: SOLA state (modified in place)
        wokada_mode: Use w-okada compatible mode (default: True)
        context_samples: Context size in samples (required for w-okada mode)

    Returns:
        CrossfadeResult with processed audio
    """
    state.frames_processed += 1

    sola_buffer_frame = state.sola_buffer_frame
    sola_search_frame = state.sola_search_frame

    # First chunk: save buffer from end
    if state.sola_buffer is None:
        if len(infer_wav) > sola_buffer_frame:
            state.sola_buffer = infer_wav[-sola_buffer_frame:].copy()
            if wokada_mode:
                # w-okada mode: return full output (keep the tail)
                return CrossfadeResult(audio=infer_wav.copy())
            else:
                # RVC WebUI mode: remove tail
                return CrossfadeResult(audio=infer_wav[:-sola_buffer_frame].copy())
        return CrossfadeResult(audio=infer_wav.copy())

    # w-okada mode: different processing strategy
    if wokada_mode:
        # In w-okada mode, infer_wav = [context | main]
        # We crossfade the saved buffer with the context region,
        # then trim the context and return [main]

        # Use context_samples if provided, otherwise fall back to sola_buffer_frame
        trim_samples = context_samples if context_samples > 0 else sola_buffer_frame

        # Not enough samples for SOLA search
        if len(infer_wav) < sola_buffer_frame + sola_search_frame:
            # Fallback: simple crossfade at the beginning
            result = infer_wav.copy()
            if len(result) >= sola_buffer_frame and len(result) > trim_samples:
                result[:sola_buffer_frame] = (
                    state.sola_buffer * state.fade_out_window +
                    result[:sola_buffer_frame] * state.fade_in_window
                )
                # Save new buffer from end (but keep it in output)
                state.sola_buffer = result[-sola_buffer_frame:].copy()
                # Trim fixed context from beginning
                return CrossfadeResult(audio=result[trim_samples:])
            else:
                state.sola_buffer = None
                return CrossfadeResult(audio=result)

        # Find optimal offset in the LEFT part (context) for crossfading
        sola_offset = _find_sola_offset(
            state.sola_buffer,
            infer_wav,
            sola_search_frame,
        )

        # Crossfade at the found offset
        result = infer_wav.copy()
        crossfade_start = sola_offset
        crossfade_end = crossfade_start + sola_buffer_frame

        if crossfade_end <= len(result) and len(result) > trim_samples:
            # Apply crossfade
            result[crossfade_start:crossfade_end] = (
                state.sola_buffer * state.fade_out_window +
                result[crossfade_start:crossfade_end] * state.fade_in_window
            )

            # Save new buffer from end (but keep it in output)
            state.sola_buffer = result[-sola_buffer_frame:].copy()

            # Trim FIXED context size from the beginning (not variable based on offset)
            # This ensures consistent output length regardless of offset
            return CrossfadeResult(audio=result[trim_samples:], sola_offset=sola_offset)
        else:
            # Not enough room for crossfade, fallback
            state.sola_buffer = None
            return CrossfadeResult(audio=result, sola_offset=sola_offset)

    # RVC WebUI mode (original implementation)
    else:
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
