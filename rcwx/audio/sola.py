"""Simple SOLA (Synchronized Overlap-Add) crossfade implementation.

Single code path, no fallback. Uses cross-correlation to find optimal
splice position, then applies Hann window crossfade.

Hold-back design: the buffer stores the last `crossfade_samples` of each
chunk's output, withholding them from playback.  The next chunk crossfades
its start with the held-back tail.  When paired with `sola_extra_samples`
in `infer_streaming()` (= crossfade + search), the net output per chunk
is approximately `hop_samples` (± search offset), keeping the ring buffer
balanced.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class SolaState:
    """State for SOLA crossfade between consecutive chunks."""

    buffer: Optional[NDArray[np.float32]] = None
    crossfade_samples: int = 0
    search_samples: int = 0  # ~10ms search window

    # Pre-computed Hann window (created on first use)
    _hann_fade_in: Optional[NDArray[np.float32]] = field(
        default=None, repr=False
    )
    _hann_fade_out: Optional[NDArray[np.float32]] = field(
        default=None, repr=False
    )

    def _ensure_window(self) -> None:
        """Create Hann crossfade windows if not yet created."""
        if self._hann_fade_in is None and self.crossfade_samples > 0:
            hann = np.hanning(2 * self.crossfade_samples).astype(np.float32)
            # First half rises 0→1, second half falls 1→0
            self._hann_fade_in = hann[:self.crossfade_samples]   # rising: 0→1
            self._hann_fade_out = hann[self.crossfade_samples:]  # falling: 1→0


def sola_crossfade(
    audio: NDArray[np.float32],
    state: SolaState,
    target_len: int = 0,
) -> NDArray[np.float32]:
    """Apply SOLA crossfade to consecutive audio chunks.

    Hold-back design: the buffer stores `cf` samples immediately after
    the output region, withholding them from playback.  The next chunk
    crossfades its start with that held-back tail.

    When ``target_len > 0`` the output is forced to exactly that many
    samples, and the hold-back is taken from immediately after the
    output boundary.  This prevents cumulative latency drift while
    keeping the hold-back contiguous with the output (no content gap).

    Args:
        audio: Current chunk audio (at output sample rate).
        state: SOLA state (modified in-place).
        target_len: If >0, force output to this exact length.

    Returns:
        Crossfaded audio chunk.
    """
    cf = state.crossfade_samples
    search = state.search_samples

    # No crossfade configured — passthrough
    if cf <= 0:
        return audio

    state._ensure_window()

    # First chunk: output + hold-back, no crossfade
    if state.buffer is None:
        if target_len > 0 and len(audio) >= target_len + cf:
            state.buffer = audio[target_len:target_len + cf].copy()
            return audio[:target_len]
        if len(audio) > cf:
            state.buffer = audio[-cf:].copy()
            return audio[:-cf]
        # Chunk too short to hold back — output nothing, hold everything
        state.buffer = audio.copy()
        return np.array([], dtype=np.float32)

    # Subsequent chunks — need at least cf samples in audio
    if len(audio) < cf:
        # Chunk shorter than crossfade — prepend buffer, no hold-back
        result = np.concatenate([state.buffer, audio])
        state.buffer = None
        return result

    prev_tail = state.buffer  # [cf] samples held from previous chunk

    # Determine search range
    search_end = min(cf + search, len(audio))
    search_region = audio[:search_end]

    # Find optimal splice offset using cross-correlation
    offset = _find_best_offset(prev_tail, search_region, cf, search)

    # Extract the crossfade region from current chunk at found offset
    curr_cf = audio[offset:offset + cf]

    # Apply Hann window crossfade
    crossfaded = prev_tail * state._hann_fade_out + curr_cf * state._hann_fade_in

    # Remaining audio after crossfade region
    remaining = audio[offset + cf:]

    # Output length: cf (crossfaded) + take (from remaining)
    if target_len > 0:
        take = target_len - cf
        if take >= 0 and take + cf <= len(remaining):
            # Hold-back contiguous with output — no content gap
            state.buffer = remaining[take:take + cf].copy()
            result = np.concatenate([crossfaded, remaining[:take]])
            return result
        # Fall through to default if audio too short for target

    if len(remaining) >= cf:
        state.buffer = remaining[-cf:].copy()
        result = np.concatenate([crossfaded, remaining[:-cf]])
    else:
        state.buffer = audio[-cf:].copy()
        result = np.concatenate([crossfaded, remaining])

    return result


def sola_flush(state: SolaState) -> NDArray[np.float32]:
    """Flush SOLA state at end of stream.

    The buffer contains held-back samples that have not been played yet.
    Return them to complete the stream.

    Returns:
        Held-back samples (or empty if no state).
    """
    if state.buffer is not None:
        out = state.buffer
        state.buffer = None
        return out
    return np.array([], dtype=np.float32)


def _find_best_offset(
    prev_tail: NDArray[np.float32],
    search_region: NDArray[np.float32],
    crossfade_samples: int,
    search_samples: int,
) -> int:
    """Find the best splice offset using normalized cross-correlation.

    Searches within [0, search_samples] for the position in search_region
    where a crossfade_samples-long window best correlates with prev_tail.

    Args:
        prev_tail: Previous chunk's tail [crossfade_samples].
        search_region: Start of current chunk to search [crossfade_samples + search_samples].
        crossfade_samples: Length of crossfade window.
        search_samples: Search range beyond crossfade start.

    Returns:
        Optimal offset (0 to search_samples).
    """
    if search_samples <= 0 or len(search_region) <= crossfade_samples:
        return 0

    max_offset = min(search_samples, len(search_region) - crossfade_samples)
    if max_offset <= 0:
        return 0

    # Normalize prev_tail (zero-mean by construction)
    pt_mean = np.mean(prev_tail)
    pt_centered = prev_tail - pt_mean
    pt_norm = np.sqrt(np.sum(pt_centered ** 2))
    if pt_norm < 1e-8:
        return 0

    cf = crossfade_samples
    region = search_region[:cf + max_offset]

    # Dot products via np.correlate (C-implemented):
    # Since sum(pt_centered)==0, centering the candidate windows doesn't
    # change the dot product: dot(pt_centered, w - mean(w)) = dot(pt_centered, w).
    dots = np.correlate(region, pt_centered, mode="valid")  # length: max_offset+1

    # Per-window norms via cumulative sums (O(N), no large intermediates):
    # norm^2 = sum((w - mean)^2) = sum(w^2) - cf * mean^2
    x = region.astype(np.float64)
    cumsum = np.empty(len(x) + 1, dtype=np.float64)
    cumsum[0] = 0.0
    np.cumsum(x, out=cumsum[1:])
    cumsum_sq = np.empty(len(x) + 1, dtype=np.float64)
    cumsum_sq[0] = 0.0
    np.cumsum(x * x, out=cumsum_sq[1:])

    window_sums = cumsum[cf:cf + max_offset + 1] - cumsum[:max_offset + 1]
    window_sq_sums = cumsum_sq[cf:cf + max_offset + 1] - cumsum_sq[:max_offset + 1]
    norms_sq = np.maximum(window_sq_sums - window_sums * window_sums / cf, 0.0)
    norms = np.sqrt(norms_sq)

    # Normalized cross-correlation (invalid windows → -inf)
    corrs = np.full(max_offset + 1, -np.inf)
    valid = norms > 1e-8
    corrs[valid] = dots[valid] / (pt_norm * norms[valid])

    # argmax returns first index on tie — same as the sequential scan
    return int(np.argmax(corrs))
