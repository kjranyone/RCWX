"""Crossfade utilities for chunk-based audio processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray


def _rms(x: NDArray[np.float32]) -> float:
    return float(np.sqrt(np.mean(x**2))) + 1e-8


def _match_rms(
    target: NDArray[np.float32],
    reference: NDArray[np.float32],
    min_gain: float = 0.3,
    max_gain: float = 3.0,
) -> NDArray[np.float32]:
    """Scale target to match RMS of reference (clamped)."""
    ref_rms = _rms(reference)
    tgt_rms = _rms(target)
    gain = ref_rms / tgt_rms
    gain = float(np.clip(gain, min_gain, max_gain))
    return target * gain


def _apply_boundary_rms_smoothing(
    audio: NDArray[np.float32],
    reference_tail: NDArray[np.float32],
    window: int,
    min_gain: float = 0.7,
    max_gain: float = 1.3,
) -> NDArray[np.float32]:
    """Gently match boundary RMS to reduce dips using a gain ramp."""
    if window <= 0 or len(audio) == 0:
        return audio
    win = min(window, len(audio), len(reference_tail))
    if win <= 0:
        return audio
    ref_rms = _rms(reference_tail[-win:])
    head_rms = _rms(audio[:win])
    gain = ref_rms / head_rms
    gain = float(np.clip(gain, min_gain, max_gain))
    if abs(gain - 1.0) < 1e-3:
        return audio
    ramp = np.linspace(gain, 1.0, win, dtype=np.float32)
    out = audio.copy()
    out[:win] = out[:win] * ramp
    return out


def _declick_head(
    audio: NDArray[np.float32],
    prev_tail: NDArray[np.float32],
    samples: int,
) -> NDArray[np.float32]:
    """Apply a short fade-in from the previous tail to avoid clicks.

    Uses a Hann window to smoothly blend from the previous chunk's last sample
    into the current audio, preventing abrupt discontinuities.
    """
    if samples <= 1 or len(audio) < samples or len(prev_tail) == 0:
        return audio

    # Create a Hann fade-in window (0 to 1)
    t = np.linspace(0.0, 1.0, samples, dtype=np.float32)
    fade = 0.5 * (1.0 - np.cos(np.pi * t))  # Hann window: smooth S-curve

    # Blend: start from previous tail's last value, fade into current audio
    prev_last = float(prev_tail[-1])
    out = audio.copy()
    out[:samples] = prev_last * (1.0 - fade) + audio[:samples] * fade
    return out


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
        # Use up to 12x zero-crossing interval, capped by crossfade length
        # This improves alignment on sustained vowels without increasing latency
        sola_search_frame = min(sola_buffer_frame, zc * 12)

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
        if wokada_mode:
            # w-okada mode: trim context even for first chunk
            trim_samples = context_samples if context_samples > 0 else 0
            if len(infer_wav) > trim_samples:
                result = infer_wav[trim_samples:].copy()
                # Save buffer from end
                if len(result) > sola_buffer_frame:
                    state.sola_buffer = result[-sola_buffer_frame:].copy()
                return CrossfadeResult(audio=result)
            return CrossfadeResult(audio=infer_wav.copy())
        else:
            # RVC WebUI mode
            if len(infer_wav) > sola_buffer_frame:
                state.sola_buffer = infer_wav[-sola_buffer_frame:].copy()
                return CrossfadeResult(audio=infer_wav[:-sola_buffer_frame].copy())
            return CrossfadeResult(audio=infer_wav.copy())

    # w-okada mode: Enhanced with SOLA crossfading for smooth chunk boundaries
    # Uses left context for phase-aligned crossfading with previous buffer
    if wokada_mode:
        # In w-okada mode, infer_wav = [context | main]
        # Trim context first, then crossfade at the boundary

        prev_tail = state.sola_buffer.copy() if state.sola_buffer is not None else None
        trim_samples = context_samples if context_samples > 0 else 0

        # Not enough samples for processing
        if len(infer_wav) <= trim_samples:
            return CrossfadeResult(audio=infer_wav.copy(), sola_offset=0, correlation=0.0)

        # Trim context from the beginning to get the main output
        result = infer_wav[trim_samples:].copy()

        # Crossfade with previous buffer at the boundary (where context was trimmed)
        if prev_tail is not None and len(result) >= sola_buffer_frame:
            # Extract the head of current output for crossfading
            head = result[:sola_buffer_frame].copy()

            # Match RMS to avoid energy dips
            head = _match_rms(head, prev_tail)

            # Apply Hann window crossfade
            result[:sola_buffer_frame] = (
                prev_tail * state.fade_out_window + head * state.fade_in_window
            )

            # Apply short declick processing at the very beginning (10ms @ 48kHz)
            # This smooths any remaining discontinuities from RVC inference
            declick_n = max(1, min(480, sola_buffer_frame // 5))
            result = _declick_head(result, prev_tail, declick_n)

        # Save buffer from end for next chunk
        if len(result) > sola_buffer_frame:
            state.sola_buffer = result[-sola_buffer_frame:].copy()

        return CrossfadeResult(audio=result, sola_offset=0, correlation=0.0)

    # RVC WebUI mode (original implementation)
    else:
        prev_tail = state.sola_buffer.copy() if state.sola_buffer is not None else None
        # Not enough samples for SOLA search
        if len(infer_wav) < sola_buffer_frame + sola_search_frame:
            # Pad to required length to keep overlap-add stable
            if len(infer_wav) > 0:
                pad_len = sola_buffer_frame + sola_search_frame - len(infer_wav)
                infer_wav = np.pad(infer_wav, (0, pad_len), mode="edge")
            else:
                return CrossfadeResult(audio=infer_wav.copy(), sola_offset=0, correlation=0.0)

        # Find optimal offset using RVC-style correlation-only search
        sola_offset, correlation = _find_sola_offset_rvc(
            state.sola_buffer,
            infer_wav,
            sola_search_frame,
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

            # RVC WebUI mode: minimal de-click at boundary if needed
            if prev_tail is not None and len(result) > 0:
                delta = abs(float(prev_tail[-1]) - float(result[0]))
                if delta > 0.05:
                    declick_n = max(1, min(64, sola_buffer_frame // 4))
                    head_end = float(result[declick_n - 1]) if len(result) >= declick_n else float(result[-1])
                    ramp = np.linspace(float(prev_tail[-1]), head_end, declick_n, dtype=np.float32)
                    result[:declick_n] = 0.7 * ramp + 0.3 * result[:declick_n]

            # Save new buffer (tail of current result)
            state.sola_buffer = result[-sola_buffer_frame:].copy()

            # Output WITHOUT tail (fixed length, matches batch processing)
            return CrossfadeResult(audio=result[:-sola_buffer_frame], sola_offset=sola_offset, correlation=correlation)
        elif len(result) >= sola_buffer_frame:
            # Short result: crossfade but can't save buffer
            region = result[:sola_buffer_frame]
            region = _match_rms(region, state.sola_buffer)
            result[:sola_buffer_frame] = (
                state.sola_buffer * state.fade_out_window
                + region * state.fade_in_window
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

    # Sort by score and take top 30 candidates for thorough evaluation
    # More candidates increases chance of finding optimal phase alignment
    candidates.sort(key=lambda x: x[2], reverse=True)
    top_candidates = candidates[:30]

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

        # Energy constraint: avoid dips at boundary
        buffer_rms = np.sqrt(np.mean(sola_buffer**2)) + 1e-8
        window_rms = np.sqrt(np.mean(window**2)) + 1e-8
        blended_rms = np.sqrt(np.mean(blended**2)) + 1e-8
        target_rms = min(buffer_rms, window_rms)
        energy_penalty = max(0.0, (target_rms * 0.95 - blended_rms) / target_rms)

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
        smoothness = (
            8.0 * start_disc
            + 8.0 * end_disc
            + 2.0 * internal_grad
            + 1.0 * grad_std
            + 0.5 * curvature
            + 4.0 * energy_penalty
        )

        if smoothness < best_smoothness:
            best_smoothness = smoothness
            best_offset = offset
            best_corr = corr

    return best_offset, best_corr


def _find_sola_offset_rvc(
    sola_buffer: NDArray[np.float32],
    infer_wav: NDArray[np.float32],
    sola_search_frame: int,
) -> tuple[int, float]:
    """RVC-style SOLA: pick max normalized correlation only."""
    sola_buffer_frame = len(sola_buffer)
    if len(infer_wav) < sola_buffer_frame + sola_search_frame:
        return 0, 0.0

    search_region = infer_wav[: sola_buffer_frame + sola_search_frame]
    sola_centered = sola_buffer - np.mean(sola_buffer)
    sola_energy = np.sum(sola_centered**2) + 1e-8

    best_offset = 0
    best_corr = -1.0
    for offset in range(sola_search_frame + 1):
        window = search_region[offset : offset + sola_buffer_frame]
        if len(window) != sola_buffer_frame:
            continue
        window_centered = window - np.mean(window)
        corr = float(np.sum(sola_centered * window_centered) / (np.sqrt(sola_energy * (np.sum(window_centered**2) + 1e-8))))
        if corr > best_corr:
            best_corr = corr
            best_offset = offset

    return best_offset, best_corr
