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
) -> NDArray[np.float32]:
    """Apply SOLA crossfade to consecutive audio chunks.

    Hold-back design: the buffer stores the last `cf` samples of each
    chunk, withholding them from output.  The next chunk's start is
    crossfaded with that held-back tail.

    When the upstream produces `hop + cf + search` samples (via
    sola_extra_samples), the net output is approximately `hop + search
    - offset`, which is close to `hop` samples per chunk.

    Args:
        audio: Current chunk audio (at output sample rate).
        state: SOLA state (modified in-place).

    Returns:
        Crossfaded audio chunk.
    """
    cf = state.crossfade_samples
    search = state.search_samples

    # No crossfade configured — passthrough
    if cf <= 0:
        return audio

    state._ensure_window()

    # First chunk: hold back last cf samples, output the rest
    if state.buffer is None:
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

    if len(remaining) >= cf:
        # Hold back last cf samples for next crossfade
        state.buffer = remaining[-cf:].copy()
        result = np.concatenate([crossfaded, remaining[:-cf]])
    else:
        # Not enough remaining to hold back — use crossfaded + remaining
        # and hold back from the full audio tail
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

    # Slide prev_tail across search_region to find best match
    best_offset = 0
    best_corr = -np.inf

    # Normalize prev_tail
    pt_mean = np.mean(prev_tail)
    pt_centered = prev_tail - pt_mean
    pt_norm = np.sqrt(np.sum(pt_centered ** 2))
    if pt_norm < 1e-8:
        return 0

    for off in range(max_offset + 1):
        candidate = search_region[off:off + crossfade_samples]
        if len(candidate) < crossfade_samples:
            break
        c_mean = np.mean(candidate)
        c_centered = candidate - c_mean
        c_norm = np.sqrt(np.sum(c_centered ** 2))
        if c_norm < 1e-8:
            continue
        corr = np.dot(pt_centered, c_centered) / (pt_norm * c_norm)
        if corr > best_corr:
            best_corr = corr
            best_offset = off

    return best_offset
