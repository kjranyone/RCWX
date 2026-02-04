"""Crossfade utilities for chunk-based audio processing."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


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


def calculate_dynamic_search_range(f0_hz: float, sample_rate: int) -> int:
    """Calculate dynamic SOLA search range based on F0.

    Uses F0 period to determine optimal search range for phase alignment.

    Args:
        f0_hz: Fundamental frequency in Hz (0 or negative = use default)
        sample_rate: Audio sample rate

    Returns:
        Search range in samples (2-4 pitch periods, clamped to 10-50ms)
    """
    if f0_hz <= 0:
        f0_hz = 150  # Default for female voice range

    period_samples = sample_rate / f0_hz
    # Search 3 pitch periods for good alignment
    search_range = int(period_samples * 3)

    # Clamp to 10-50ms range
    min_search = sample_rate // 100  # 10ms
    max_search = sample_rate // 20   # 50ms

    return max(min_search, min(search_range, max_search))


def _find_zero_crossing(
    audio: NDArray[np.float32],
    start: int,
    search_range: int,
    direction: str = "forward"
) -> int:
    """Find nearest zero crossing point in audio.

    Args:
        audio: Audio signal
        start: Starting position to search from
        search_range: Maximum samples to search
        direction: "forward" or "backward"

    Returns:
        Index of zero crossing, or start if none found
    """
    if direction == "forward":
        end = min(start + search_range, len(audio) - 1)
        for i in range(start, end):
            if audio[i] * audio[i + 1] <= 0:  # Sign change
                # Return the sample closer to zero
                if abs(audio[i]) < abs(audio[i + 1]):
                    return i
                return i + 1
    else:  # backward
        end = max(start - search_range, 0)
        for i in range(start, end, -1):
            if audio[i] * audio[i - 1] <= 0:  # Sign change
                if abs(audio[i]) < abs(audio[i - 1]):
                    return i
                return i - 1
    return start


def _apply_strong_crossfade(
    prev_tail: NDArray[np.float32],
    main_output: NDArray[np.float32],
    crossfade_len: int,
    sample_rate: int = 48000,
) -> tuple[NDArray[np.float32], int]:
    """Apply strong crossfade without relying on correlation.

    Phase 8: Improved crossfade for RVC output where correlation-based
    alignment fails due to independent chunk processing.

    Uses:
    1. Zero-crossing alignment for smoother transitions
    2. cos^6 fade curve for very gradual blending (slower than cos^4)
    3. DC offset removal before blending
    4. Longer crossfade region (150ms+)
    5. Dual-stage blending: fast at edges, slow in middle

    Args:
        prev_tail: Previous chunk's tail
        main_output: Current chunk's output
        crossfade_len: Desired crossfade length in samples
        sample_rate: Audio sample rate

    Returns:
        Tuple of (blended audio, actual crossfade length used)
    """
    # Minimum crossfade: 150ms for very smooth RVC transitions
    min_crossfade = int(sample_rate * 0.15)  # 150ms
    crossfade_len = max(crossfade_len, min_crossfade)

    # Ensure we have enough samples
    max_available = min(len(prev_tail), len(main_output) // 2)
    actual_len = min(crossfade_len, max_available)

    if actual_len < sample_rate // 100:  # Less than 10ms
        # Too short, just concatenate
        return main_output.copy(), 0

    # Remove DC offset from both signals
    prev_dc = np.mean(prev_tail[-actual_len:])
    main_dc = np.mean(main_output[:actual_len])
    prev_region = prev_tail[-actual_len:] - prev_dc
    main_region = main_output[:actual_len].copy() - main_dc

    # Find zero crossing at the end of prev_region for smoother start
    zc_search = min(actual_len // 10, sample_rate // 200)  # 5ms search range
    if zc_search > 5:
        zc_offset = _find_zero_crossing(prev_region, actual_len - 1, zc_search, "backward")
        trim_amount = (actual_len - 1) - zc_offset
        if trim_amount > 0 and trim_amount < actual_len // 4:
            # Adjust crossfade to start at zero crossing
            actual_len -= trim_amount
            prev_region = prev_region[-actual_len:]
            main_region = main_output[:actual_len].copy() - main_dc

    # Create smooth fade curves that sum to 1.0 (energy-preserving)
    t = np.linspace(0.0, 1.0, actual_len, dtype=np.float32)
    # Use smoothstep-like curve: 3t² - 2t³ for smooth S-curve
    # This ensures fade_in + fade_out = 1.0 at all points
    fade_in = 3 * t**2 - 2 * t**3  # Smoothstep: 0→1
    fade_out = 1.0 - fade_in        # Complementary: 1→0

    # RMS matching with wider tolerance - preserve dynamics
    prev_rms = np.sqrt(np.mean(prev_region**2)) + 1e-8
    main_rms = np.sqrt(np.mean(main_region**2)) + 1e-8
    rms_ratio = prev_rms / main_rms

    # Very gentle RMS matching to preserve natural dynamics
    rms_ratio = np.clip(rms_ratio, 0.8, 1.25)
    main_region = main_region * rms_ratio

    # Apply crossfade
    blended = prev_region * fade_out + main_region * fade_in

    # Add DC back (average of both)
    avg_dc = (prev_dc + main_dc) / 2
    blended = blended + avg_dc

    # Build full output
    result = np.concatenate([blended, main_output[actual_len:]])

    return result, actual_len


@dataclass
class SOLAState:
    """State for RVC-style SOLA crossfade processing."""

    sola_buffer: Optional[NDArray[np.float32]] = None
    sola_buffer_frame: int = 0
    sola_search_frame: int = 0
    fade_in_window: Optional[NDArray[np.float32]] = None
    fade_out_window: Optional[NDArray[np.float32]] = None
    frames_processed: int = 0
    # Phase 3: Extended state for advanced SOLA
    use_advanced_sola: bool = False
    fallback_threshold: float = 0.3
    last_f0_hz: float = 0.0
    # Phase 8: Track first chunk for special handling
    # First chunk boundary often has discontinuity due to reflection padding
    is_first_chunk_boundary: bool = True

    @staticmethod
    def create(
        crossfade_samples: int,
        sample_rate: int = 48000,
        use_advanced_sola: bool = False,
        fallback_threshold: float = 0.3,
    ) -> "SOLAState":
        """
        Create SOLA state with proper window initialization.

        Args:
            crossfade_samples: Crossfade length in samples
            sample_rate: Audio sample rate
            use_advanced_sola: Use advanced multi-candidate SOLA evaluation
            fallback_threshold: Correlation threshold for fallback to simple crossfade

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
            use_advanced_sola=use_advanced_sola,
            fallback_threshold=fallback_threshold,
            is_first_chunk_boundary=True,
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

    # First chunk: handled within each mode's logic below

    # w-okada mode: SOLA crossfading for smooth chunk boundaries
    # In w-okada mode, infer_wav = [context | main]
    #
    # Overlap-Add principle:
    # - Each chunk outputs WITHOUT its tail (which will be crossfaded with next chunk)
    # - The tail is saved in sola_buffer
    # - Next chunk crossfades saved tail with its head, then outputs
    # - This ensures seamless concatenation of chunk outputs
    if wokada_mode:
        prev_tail = state.sola_buffer.copy() if state.sola_buffer is not None else None
        trim_samples = context_samples if context_samples > 0 else 0

        # Not enough samples for processing
        if len(infer_wav) <= trim_samples:
            return CrossfadeResult(audio=infer_wav.copy(), sola_offset=0, correlation=0.0)

        # Trim context to get main output
        main_output = infer_wav[trim_samples:].copy()

        # First chunk: output WITHOUT tail, save tail for crossfade with next chunk
        if prev_tail is None:
            if len(main_output) > sola_buffer_frame:
                # Save tail for crossfade
                saved_tail = main_output[-sola_buffer_frame:].copy()

                # Phase 8: Apply strong fade-out to saved tail for first chunk
                # The first chunk uses reflection padding which creates a large
                # discontinuity with the next chunk. Strong fade-out (to ~0) is
                # required to eliminate clicks at this boundary.
                t = np.linspace(0.0, 1.0, len(saved_tail), dtype=np.float32)
                fade_out = np.power(np.cos(np.pi * t / 2), 4)  # cos^4 for smooth fade to 0
                saved_tail = saved_tail * fade_out

                state.sola_buffer = saved_tail
                result = main_output[:-sola_buffer_frame].copy()
            else:
                result = main_output.copy()
            return CrossfadeResult(audio=result, sola_offset=0, correlation=0.0)

        # Subsequent chunks: crossfade prev_tail with head of main_output
        sola_offset = 0
        correlation = 0.0

        if len(main_output) >= sola_buffer_frame + sola_search_frame:
            # Choose SOLA algorithm based on state settings
            if state.use_advanced_sola:
                # Advanced multi-candidate SOLA with phase continuity evaluation
                sola_offset, correlation = _find_sola_offset(
                    prev_tail,
                    main_output,
                    sola_search_frame,
                    fade_in_window=state.fade_in_window,
                    fade_out_window=state.fade_out_window,
                )
            else:
                # Standard RVC-style correlation-only SOLA
                sola_offset, correlation = _find_sola_offset_rvc(
                    prev_tail,
                    main_output,
                    sola_search_frame,
                )

            # Phase 8: Low correlation fallback with strong crossfade
            # If correlation is too low, SOLA alignment won't work well
            # Use extended crossfade with zero-crossing alignment instead
            # Also always use fallback for first chunk boundary (reflection padding issue)
            use_fallback = correlation < state.fallback_threshold or state.is_first_chunk_boundary

            if use_fallback:
                # Phase 8: Use strong crossfade optimized for RVC output
                sample_rate = 48000  # Assumed output rate

                # First chunk boundary needs extra-long crossfade (200ms)
                # due to reflection padding vs real audio mismatch
                if state.is_first_chunk_boundary:
                    min_crossfade_len = int(sample_rate * 0.20)  # 200ms for first boundary
                    state.is_first_chunk_boundary = False  # Mark as processed
                else:
                    min_crossfade_len = int(sample_rate * 0.15)  # 150ms for others

                target_crossfade_len = max(sola_buffer_frame * 3, min_crossfade_len)

                # Apply strong crossfade with zero-crossing detection
                blended_output, actual_crossfade = _apply_strong_crossfade(
                    prev_tail,
                    main_output,
                    target_crossfade_len,
                    sample_rate,
                )

                # Update SOLA buffer with tail of result
                if len(blended_output) > sola_buffer_frame:
                    state.sola_buffer = blended_output[-sola_buffer_frame:].copy()
                    result = blended_output[:-sola_buffer_frame].copy()
                else:
                    state.sola_buffer = None
                    result = blended_output.copy()

                return CrossfadeResult(
                    audio=result,
                    sola_offset=0,
                    correlation=correlation
                )

            # Extract crossfade region at optimal offset
            crossfade_region = main_output[sola_offset : sola_offset + sola_buffer_frame].copy()

            # Match RMS with tighter limits
            crossfade_region = _match_rms(crossfade_region, prev_tail, min_gain=0.5, max_gain=2.0)

            # Apply Hann window crossfade
            blended = (
                prev_tail * state.fade_out_window + crossfade_region * state.fade_in_window
            )

            # Build result: [blended | rest]
            # Note: samples before sola_offset are skipped (phase alignment)
            rest_start = sola_offset + sola_buffer_frame
            if rest_start < len(main_output) - sola_buffer_frame:
                # Have room for full result minus new tail
                full_result = np.concatenate([blended, main_output[rest_start:]])
                state.sola_buffer = full_result[-sola_buffer_frame:].copy()
                result = full_result[:-sola_buffer_frame].copy()
            elif rest_start < len(main_output):
                # Short result, save tail if possible
                full_result = np.concatenate([blended, main_output[rest_start:]])
                if len(full_result) > sola_buffer_frame:
                    state.sola_buffer = full_result[-sola_buffer_frame:].copy()
                    result = full_result[:-sola_buffer_frame].copy()
                else:
                    state.sola_buffer = None
                    result = full_result
            else:
                # Only blended region
                state.sola_buffer = None
                result = blended.copy()

        elif len(main_output) >= sola_buffer_frame:
            # Fallback: simple crossfade at position 0 without offset search
            head = main_output[:sola_buffer_frame].copy()
            head = _match_rms(head, prev_tail, min_gain=0.5, max_gain=2.0)
            blended = (
                prev_tail * state.fade_out_window + head * state.fade_in_window
            )

            # Build result
            if len(main_output) > 2 * sola_buffer_frame:
                full_result = np.concatenate([blended, main_output[sola_buffer_frame:]])
                state.sola_buffer = full_result[-sola_buffer_frame:].copy()
                result = full_result[:-sola_buffer_frame].copy()
            elif len(main_output) > sola_buffer_frame:
                full_result = np.concatenate([blended, main_output[sola_buffer_frame:]])
                state.sola_buffer = full_result[-sola_buffer_frame:].copy() if len(full_result) > sola_buffer_frame else None
                result = full_result[:-sola_buffer_frame] if len(full_result) > sola_buffer_frame else full_result
            else:
                state.sola_buffer = None
                result = blended.copy()

        else:
            # Very short output, can't crossfade properly
            state.sola_buffer = None
            result = main_output.copy()

        return CrossfadeResult(audio=result, sola_offset=sola_offset, correlation=correlation)

    # RVC WebUI mode (original implementation)
    else:
        # First chunk in RVC WebUI mode: just save buffer and return
        if state.sola_buffer is None:
            if len(infer_wav) > sola_buffer_frame:
                saved_tail = infer_wav[-sola_buffer_frame:].copy()

                # Phase 8: Apply strong fade-out to saved tail for first chunk
                # Same as wokada mode - first chunk uses reflection padding
                t = np.linspace(0.0, 1.0, len(saved_tail), dtype=np.float32)
                fade_out = np.power(np.cos(np.pi * t / 2), 4)  # cos^4 fade to 0
                saved_tail = saved_tail * fade_out

                state.sola_buffer = saved_tail
                return CrossfadeResult(audio=infer_wav[:-sola_buffer_frame].copy())
            return CrossfadeResult(audio=infer_wav.copy())

        prev_tail = state.sola_buffer.copy()
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

        # Phase 8: Low correlation or first chunk boundary fallback
        use_fallback = correlation < state.fallback_threshold or state.is_first_chunk_boundary

        logger.debug(
            f"[SOLA-Phase8] corr={correlation:.4f}, threshold={state.fallback_threshold}, "
            f"is_first={state.is_first_chunk_boundary}, use_fallback={use_fallback}"
        )

        # Phase 8: Relax the length check - need at least buffer + some headroom
        if use_fallback and len(infer_wav) >= sola_buffer_frame + sola_search_frame:
            sample_rate = 48000  # Assumed output rate

            # First chunk boundary needs extra-long crossfade (200ms)
            if state.is_first_chunk_boundary:
                min_crossfade_len = int(sample_rate * 0.20)  # 200ms
                state.is_first_chunk_boundary = False  # Mark as processed
            else:
                min_crossfade_len = int(sample_rate * 0.15)  # 150ms

            target_crossfade_len = max(sola_buffer_frame * 3, min_crossfade_len)

            # Apply strong crossfade with zero-crossing detection
            blended_output, actual_crossfade = _apply_strong_crossfade(
                prev_tail,
                infer_wav,
                target_crossfade_len,
                sample_rate,
            )

            logger.info(
                f"[SOLA-Phase8-FALLBACK] Applied strong crossfade: "
                f"target={target_crossfade_len}, actual={actual_crossfade}, "
                f"min={min_crossfade_len} samples"
            )

            # Save new buffer from tail
            if len(blended_output) > sola_buffer_frame:
                state.sola_buffer = blended_output[-sola_buffer_frame:].copy()
                return CrossfadeResult(
                    audio=blended_output[:-sola_buffer_frame],
                    sola_offset=0,
                    correlation=correlation
                )
            else:
                state.sola_buffer = None
                return CrossfadeResult(
                    audio=blended_output.copy(),
                    sola_offset=0,
                    correlation=correlation
                )

        # FIXED: Apply offset ONLY to crossfade region, not entire chunk
        # This maintains fixed output length and matches batch processing behavior
        result = infer_wav.copy()

        # Apply crossfade in-place
        if len(result) >= 2 * sola_buffer_frame and len(result) >= sola_offset + sola_buffer_frame:
            # Extract crossfade region at optimal offset position
            crossfade_region = result[sola_offset : sola_offset + sola_buffer_frame].copy()

            # Match RMS for energy consistency (with controlled limits)
            crossfade_region = _match_rms(crossfade_region, state.sola_buffer, min_gain=0.5, max_gain=2.0)

            # Crossfade previous buffer with current chunk at offset position
            blended = (
                state.sola_buffer * state.fade_out_window
                + crossfade_region * state.fade_in_window
            )

            # Place blended region at the beginning of output
            result[:sola_buffer_frame] = blended

            # RVC WebUI mode: boundary smoothing if needed
            if prev_tail is not None and len(result) > sola_buffer_frame:
                # Check discontinuity at end of crossfade region
                delta = abs(float(result[sola_buffer_frame - 1]) - float(result[sola_buffer_frame]))
                if delta > 0.1:
                    # Smooth transition from crossfade to rest of audio
                    smooth_len = min(64, len(result) - sola_buffer_frame)
                    if smooth_len > 1:
                        t = np.linspace(0.0, 1.0, smooth_len, dtype=np.float32)
                        fade = 0.5 * (1.0 - np.cos(np.pi * t))
                        end_val = float(result[sola_buffer_frame - 1])
                        result[sola_buffer_frame : sola_buffer_frame + smooth_len] = (
                            end_val * (1.0 - fade) + result[sola_buffer_frame : sola_buffer_frame + smooth_len] * fade
                        )

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
