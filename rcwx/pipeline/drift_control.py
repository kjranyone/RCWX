"""Floor-based, skip-only drift control for the realtime output ring.

Latency state is judged on the MINIMUM post-read ring level over the last
~2 chunk periods (the floor), never on the instantaneous level — the
burst/drain oscillation and device blocksize phase beats don't lift the
floor, so they can't trigger false skips.

Shedding is done exclusively via hard-skip (one splice + fade-in), NEVER
per-callback time-compression: with small device blocks (e.g. 64-frame
ASIO) an np.interp every callback restarts the resampling phase each block,
producing an audible tone at sample_rate/blocksize.  A periodic clean skip
is inaudible by comparison.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ShedPolicy:
    """Floor limit, post-trim target, and the guard the target encodes."""

    threshold: int
    target: int
    guard_ms: float


def compute_shed_policy(
    *,
    deadline_mode: bool,
    hop_samples_out: int,
    output_sample_rate: int,
    chunk_sec: float,
    buffer_margin: float,
    inference_sample_count: int,
    inference_p50_ms: float,
    inference_p99_ms: float,
    requested_callback_sec: float,
    observed_callback_sec: float,
) -> ShedPolicy:
    """Return the shed policy for the post-read floor.

    Normal bounds persistent lag to a fraction of one hop.  Deadline
    (aggressive) derives an adaptive guard from observed inference jitter
    plus the real callback duration.

    Args:
        deadline_mode: True for the aggressive/deadline latency policy.
        hop_samples_out: Output hop length in samples.
        output_sample_rate: Runtime output sample rate in Hz.
        chunk_sec: Hop duration in seconds.
        buffer_margin: Normal-mode persistent-lag margin (fraction of hop).
        inference_sample_count: Number of live inference timings collected.
        inference_p50_ms / inference_p99_ms: Rolling latency percentiles.
        requested_callback_sec: Callback duration requested for non-ASIO.
        observed_callback_sec: Callback duration actually observed; ASIO
            follows the driver panel, not the request.
    """
    if deadline_mode:
        if inference_sample_count < 20:
            # Preserve the initial one-hop standing floor until enough live
            # samples exist to estimate scheduler/GPU jitter.
            return ShedPolicy(
                threshold=hop_samples_out * 5 // 4,
                target=hop_samples_out,
                guard_ms=chunk_sec * 1000.0,
            )

        hop_ms = chunk_sec * 1000.0
        jitter_ms = max(0.0, inference_p99_ms - inference_p50_ms)
        callback_ms = max(requested_callback_sec, observed_callback_sec) * 1000.0
        guard_ms = max(hop_ms * 0.5, jitter_ms + callback_ms)
        guard_ms = min(hop_ms * 0.875, guard_ms)
        target = max(1, int(round(guard_ms * output_sample_rate / 1000.0)))
        # Hysteresis band between target and threshold.  It must exceed one
        # callback of wobble, or the post-trim level sits right at the
        # threshold and re-triggers on the next phase beat.
        band = max(
            hop_samples_out // 4,
            int(round(callback_ms * output_sample_rate / 1000.0)),
        )
        threshold = min(hop_samples_out * 5 // 4, target + band)
        return ShedPolicy(
            threshold=max(1, threshold),
            target=max(0, target),
            guard_ms=target * 1000.0 / output_sample_rate,
        )

    # The two-hop floor window filters the normal burst/drain sawtooth.
    # Any floor above this bound is persistent queued audio, so Normal sheds
    # it to a small cushion instead of preserving one full hop.
    margin = max(0.1, min(2.0, float(buffer_margin)))
    threshold = int((0.5 + margin) * hop_samples_out)
    target = hop_samples_out // 4
    return ShedPolicy(
        threshold=max(1, threshold),
        target=max(0, target),
        guard_ms=target * 1000.0 / output_sample_rate,
    )


class FloorTracker:
    """Sliding-window MINIMUM of post-read ring levels (~2 chunk periods).

    The true sawtooth trough occurs right before a burst lands, and only
    the post-read sample at the preceding callback captures it (pre-read
    sampling sits one callback higher and hides the real floor).
    """

    def __init__(self) -> None:
        self._window: Optional[deque] = None
        self._window_frames = 0
        self.floor_samples = 0

    def ensure_window(self, frames: int, hop_samples_out: int) -> None:
        """(Re)create the window when the device blocksize changes."""
        if self._window is None or self._window_frames != frames:
            self._window_frames = frames
            window_len = max(4, -(-2 * hop_samples_out // max(1, frames)))
            self._window = deque(maxlen=window_len)
            self.floor_samples = 0

    @property
    def full(self) -> bool:
        return self._window is not None and len(self._window) == self._window.maxlen

    @property
    def floor(self) -> int:
        """Windowed minimum; only meaningful when ``full``."""
        return min(self._window)

    def record_post_read(self, level: int) -> None:
        self._window.append(level)
        if len(self._window) == self._window.maxlen:
            self.floor_samples = min(self._window)

    def after_skip(self, target: int) -> None:
        """Re-measure from the new level after a hard-skip."""
        self._window.clear()
        self.floor_samples = target

    def clear(self) -> None:
        """Drop measurements but keep the window size (underrun re-arm)."""
        if self._window is not None:
            self._window.clear()
        self.floor_samples = 0

    def reset(self) -> None:
        """Full reset including window sizing (session start)."""
        self._window = None
        self._window_frames = 0
        self.floor_samples = 0

    def prime(self, frames: int, hop_samples_out: int, floor: int) -> None:
        """Pre-fill the window so the next callback sees ``floor`` (tests)."""
        self.ensure_window(frames, hop_samples_out)
        assert self._window is not None
        self._window.extend([floor] * self._window.maxlen)
        self.floor_samples = floor
