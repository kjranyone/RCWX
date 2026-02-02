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

        # Use the requested crossfade_samples directly (no 4*zc limitation)
        # For w-okada mode, we need full context length for crossfading
        sola_buffer_frame = crossfade_samples
        # Use 3x zero-crossing interval (30ms @ 48kHz)
        # Optimal search range (40ms+ increases latency without quality gain)
        sola_search_frame = zc * 3

        # Hann (raised cosine) fade windows - optimal for smooth audio transitions
        t = np.linspace(0.0, 1.0, sola_buffer_frame, dtype=np.float32)
        fade_in = 0.5 * (1.0 - np.cos(np.pi * t))
        fade_out = 1.0 - fade_in

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
    correlation: float = 0.0  # Renamed from sola_correlation for consistency


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

    # w-okada mode: Enhanced with SOLA crossfading for smooth chunk boundaries
    # Uses left context for phase-aligned crossfading with previous buffer
    if wokada_mode:
        # In w-okada mode, infer_wav = [context | main]
        # Use context for crossfading with previous buffer, then trim it

        trim_samples = context_samples if context_samples > 0 else 0

        # Not enough samples for crossfading
        if len(infer_wav) < trim_samples + sola_buffer_frame:
            # Fallback: simple trim without crossfade
            if trim_samples > 0 and len(infer_wav) > trim_samples:
                result = infer_wav[trim_samples:].copy()
                # Save buffer from end for next chunk
                if len(result) > sola_buffer_frame:
                    state.sola_buffer = result[-sola_buffer_frame:].copy()
                return CrossfadeResult(audio=result, sola_offset=0, correlation=0.0)
            else:
                return CrossfadeResult(audio=infer_wav.copy(), sola_offset=0, correlation=0.0)

        # SOLA crossfade: use left context for offset search
        # Context region: first (trim_samples + sola_search_frame) samples
        context_region = infer_wav[: trim_samples + sola_search_frame]

        # Find optimal offset within context region
        sola_offset, correlation = _find_sola_offset(
            state.sola_buffer,
            context_region,
            min(sola_search_frame, trim_samples),  # Search within context
            state.fade_in_window,
            state.fade_out_window,
        )

        # Apply crossfade at the found offset
        result = infer_wav.copy()
        crossfade_start = sola_offset
        crossfade_end = crossfade_start + sola_buffer_frame

        if crossfade_end <= len(result):
            # Blend previous buffer with current audio at optimal position
            result[crossfade_start:crossfade_end] = (
                state.sola_buffer * state.fade_out_window
                + result[crossfade_start:crossfade_end] * state.fade_in_window
            )

        # Trim context from the beginning
        if trim_samples > 0 and len(result) > trim_samples:
            result = result[trim_samples:]

        # Save buffer from end for next chunk
        if len(result) > sola_buffer_frame:
            state.sola_buffer = result[-sola_buffer_frame:].copy()

        return CrossfadeResult(audio=result, sola_offset=sola_offset, correlation=correlation)

    # RVC WebUI mode (original implementation)
    else:
        # Not enough samples for SOLA search
        if len(infer_wav) < sola_buffer_frame + sola_search_frame:
            # Fallback: simple crossfade without offset search
            if len(infer_wav) >= sola_buffer_frame:
                result = infer_wav.copy()
                result[:sola_buffer_frame] = (
                    state.sola_buffer * state.fade_out_window
                    + result[:sola_buffer_frame] * state.fade_in_window
                )
                state.sola_buffer = result[-sola_buffer_frame:].copy()
            else:
                result = infer_wav.copy()
                state.sola_buffer = None
            return CrossfadeResult(audio=result, sola_offset=0, correlation=0.0)

        # Find optimal offset using multi-candidate evaluation
        sola_offset, correlation = _find_sola_offset(
            state.sola_buffer,
            infer_wav,
            sola_search_frame,
            state.fade_in_window,
            state.fade_out_window,
        )

        # FIXED: Apply offset ONLY to crossfade region, not entire chunk
        # This maintains fixed output length and matches batch processing behavior
        result = infer_wav.copy()

        # Apply crossfade in-place
        if len(result) >= 2 * sola_buffer_frame and len(result) >= sola_offset + sola_buffer_frame:
            # Extract crossfade region at optimal offset position
            crossfade_region = result[sola_offset : sola_offset + sola_buffer_frame].copy()

            # Crossfade previous buffer with current chunk at offset position
            blended = (
                state.sola_buffer * state.fade_out_window
                + crossfade_region * state.fade_in_window
            )

            # Place blended region at the beginning of output
            result[:sola_buffer_frame] = blended

            # Save new buffer (tail of current result)
            state.sola_buffer = result[-sola_buffer_frame:].copy()

            # Output WITHOUT tail (fixed length, matches batch processing)
            return CrossfadeResult(audio=result[:-sola_buffer_frame], sola_offset=sola_offset, correlation=correlation)
        elif len(result) >= sola_buffer_frame:
            # Short result: crossfade but can't save buffer
            result[:sola_buffer_frame] = (
                state.sola_buffer * state.fade_out_window
                + result[:sola_buffer_frame] * state.fade_in_window
            )
            state.sola_buffer = None
            return CrossfadeResult(audio=result, sola_offset=sola_offset, correlation=correlation)
        else:
            state.sola_buffer = None
            return CrossfadeResult(audio=result, sola_offset=sola_offset, correlation=correlation)


def _find_sola_offset(
    sola_buffer: NDArray[np.float32],
    infer_wav: NDArray[np.float32],
    sola_search_frame: int,
    fade_in_window: NDArray[np.float32] | None = None,
    fade_out_window: NDArray[np.float32] | None = None,
) -> tuple[int, float]:
    """
    Find optimal SOLA offset using multi-candidate evaluation.

    Evaluates multiple high-scoring candidates by actually applying crossfade
    and measuring smoothness (gradient continuity) at boundaries.

    Args:
        sola_buffer: Previous chunk's tail
        infer_wav: Current inference output
        sola_search_frame: Search range in samples
        fade_in_window: Fade-in window for crossfade evaluation
        fade_out_window: Fade-out window for crossfade evaluation

    Returns:
        Tuple of (optimal offset, best correlation coefficient)
    """
    sola_buffer_frame = len(sola_buffer)

    if len(infer_wav) < sola_buffer_frame + sola_search_frame:
        return 0, 0.0

    # Search region of current output
    search_region = infer_wav[: sola_buffer_frame + sola_search_frame]

    # Pre-compute buffer statistics
    sola_centered = sola_buffer - np.mean(sola_buffer)
    sola_rms = np.sqrt(np.mean(sola_buffer**2)) + 1e-8

    # First pass: compute scores for all candidates
    candidates = []

    for offset in range(sola_search_frame + 1):
        window = search_region[offset : offset + sola_buffer_frame]

        if len(window) != sola_buffer_frame:
            continue

        # Proper normalized cross-correlation
        window_centered = window - np.mean(window)

        nom = np.sum(sola_centered * window_centered)
        den = np.sqrt(np.sum(sola_centered**2) * np.sum(window_centered**2)) + 1e-8

        corr = nom / den

        # Energy (RMS) matching score
        window_rms = np.sqrt(np.mean(window**2)) + 1e-8
        energy_ratio = min(sola_rms, window_rms) / max(sola_rms, window_rms)

        # Combined score (increased energy ratio weight for better pattern matching)
        score = corr * (0.6 + 0.4 * energy_ratio)

        candidates.append((offset, corr, score))

    if not candidates:
        return 0, 0.0

    # Sort by score and take top 10 candidates for thorough evaluation
    # More candidates increases chance of finding optimal phase alignment
    candidates.sort(key=lambda x: x[2], reverse=True)
    top_candidates = candidates[:10]

    # If windows not provided, return best by score
    if fade_in_window is None or fade_out_window is None:
        best_offset, best_corr, _ = top_candidates[0]
        return best_offset, best_corr

    # Second pass: evaluate phase continuity for top candidates
    best_offset = top_candidates[0][0]
    best_corr = top_candidates[0][1]
    best_smoothness = np.inf

    for offset, corr, _ in top_candidates:
        # Apply crossfade
        window = search_region[offset : offset + sola_buffer_frame]
        blended = sola_buffer * fade_out_window + window * fade_in_window

        # Measure phase continuity at crossfade boundaries
        # 1. Gradient continuity at crossfade start
        if len(sola_buffer) > 1:
            # Gradient before crossfade (last few samples of previous buffer)
            pre_grad = sola_buffer[-1] - sola_buffer[-2]
            # Gradient at start of crossfade
            start_grad = blended[0] - sola_buffer[-1]
            # Gradient change (phase discontinuity measure)
            start_disc = abs(start_grad - pre_grad)
        else:
            start_disc = 0.0

        # 2. Gradient continuity at crossfade end
        remaining = search_region[offset + sola_buffer_frame : offset + sola_buffer_frame + 2]
        if len(remaining) >= 2:
            # Gradient at end of crossfade
            end_grad = blended[-1] - blended[-2]
            # Gradient after crossfade
            post_grad = remaining[0] - blended[-1]
            # Gradient change
            end_disc = abs(post_grad - end_grad)
        else:
            end_disc = 0.0

        # 3. Maximum absolute gradient within crossfade (avoid sharp jumps)
        internal_grad = np.max(np.abs(np.diff(blended)))

        # 4. Standard deviation of gradient (prefer consistent slope)
        grad_std = np.std(np.diff(blended))

        # 5. Curvature (2nd derivative) - prefer smooth curves over sharp bends
        # Second derivative measures rate of change of slope
        if len(blended) > 2:
            curvature = np.max(np.abs(np.diff(np.diff(blended))))
        else:
            curvature = 0.0

        # Combined smoothness score (lower is better)
        # Prioritize boundary continuity, then internal smoothness and curvature
        smoothness = 8.0 * start_disc + 8.0 * end_disc + 2.0 * internal_grad + 1.0 * grad_std + 0.5 * curvature

        if smoothness < best_smoothness:
            best_smoothness = smoothness
            best_offset = offset
            best_corr = corr

    return best_offset, best_corr
