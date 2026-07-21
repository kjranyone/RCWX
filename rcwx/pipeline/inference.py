"""RVC inference pipeline."""

from __future__ import annotations

import logging
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.signal import butter, filtfilt, medfilt

from rcwx.audio.denoise import denoise as denoise_audio
from rcwx.audio.resample import resample
from rcwx.device import get_device, get_dtype
from rcwx.downloader import get_hubert_path, get_rmvpe_path
from rcwx.models.fcpe import FCPE, is_fcpe_available
from rcwx.models.hubert_loader import HuBERTLoader
from rcwx.models.rmvpe import RMVPE
from rcwx.models.swiftf0 import SwiftF0 as SwiftF0Model
from rcwx.models.swiftf0 import is_swiftf0_available
from rcwx.models.synthesizer import SynthesizerLoader

logger = logging.getLogger(__name__)

# Minimum feature frames required by the synthesizer decoder
# The decoder uses upsampling convolutions that require sufficient input length
# 64 frames @ 100fps = 640ms worth of features
MIN_SYNTH_FEATURE_FRAMES = 64

# Upper bound for moe_boost strength (F0-only stylization intensity).
MAX_MOE_BOOST = 1.0
F0_HISTORY_FRAMES = 20  # 200ms @ 100fps, Butterworth ord-2 warmup に十分
# Windowed-F0 extraction floor (16kHz samples, 0.56s).  RMVPE's U-Net needs a
# minimum number of mel frames (short inputs crash in avg_pool), and a fixed
# slice size avoids per-chunk kernel recompilation on Intel XPU.
F0_MIN_WINDOW_16K = 8960
# Fallback stage-profiling cadence when device timing events are unavailable:
# wall-clock + device-wide sync is only paid on every Nth infer_streaming call
# (plus the first 5).  With event support, profiling is sync-free and runs on
# every chunk.
STAGE_PROFILE_INTERVAL = 10
FCPE_VOICING_THRESHOLD = 0.006
RMVPE_VOICING_THRESHOLD = 0.015
SWIFTF0_VOICING_THRESHOLD = 0.35
# F0 correction hysteresis: corrections sustained this many consecutive
# frames are treated as genuine pitch transitions and accepted (the raw
# values are restored), so single-frame glitch suppression never latches
# onto real octave jumps or fast slides.
F0_CORRECTION_SUSTAIN_FRAMES = 3

class _StageProfiler:
    """Per-stage timing without hot-path device syncs.

    GPU stages are bracketed with device timing events and resolved in
    :meth:`finalize` after the chunk's natural end-of-pipeline sync
    (``output.cpu()``), so no ``synchronize()`` lands on the hot path.
    If timing events are unavailable, :meth:`stop` falls back to wall-clock
    plus an explicit device sync — callers then only profile sampled chunks
    (see ``STAGE_PROFILE_INTERVAL``).  CPU-synchronous stages use
    :meth:`stop_wall` (exact without any sync).  ``list.append`` from the
    parallel HuBERT/F0 threads is GIL-safe.
    """

    def __init__(self, pipeline: "RVCPipeline", use_events: bool):
        self._pipeline = pipeline
        self._use_events = use_events
        self._pending: list = []  # (key, start_event, end_event)
        self.times: dict = {}

    def _new_event(self):
        try:
            dev = str(self._pipeline.device)
            if "xpu" in dev:
                return torch.xpu.Event(enable_timing=True)
            if "cuda" in dev:
                return torch.cuda.Event(enable_timing=True)
        except Exception:
            pass
        return None

    def start(self):
        """Begin a GPU stage: a timing event if available, else wall clock."""
        if self._use_events:
            ev = self._new_event()
            if ev is not None:
                ev.record()
                return ev
            self._use_events = False
            self._pipeline._profile_events_ok = False
        return time.perf_counter()

    def stop(self, key: str, token) -> None:
        """End a GPU stage started with :meth:`start`."""
        if isinstance(token, float):
            # Wall-clock fallback needs a device sync for real numbers.
            self._pipeline._sync_device_for_profile()
            self.times[key] = (
                self.times.get(key, 0.0) + (time.perf_counter() - token) * 1000.0
            )
            return
        end = self._new_event()
        if end is None:
            return
        end.record()
        self._pending.append((key, token, end))

    def stop_wall(self, key: str, t0: float) -> None:
        """End a CPU-synchronous stage (no device sync needed)."""
        self.times[key] = self.times.get(key, 0.0) + (time.perf_counter() - t0) * 1000.0

    def finalize(self) -> None:
        """Resolve event-based stages; call after the end-of-chunk sync."""
        for key, start, end in self._pending:
            try:
                self.times[key] = self.times.get(key, 0.0) + float(
                    start.elapsed_time(end)
                )
                self._pipeline._profile_events_ok = True
            except Exception:
                self._pipeline._profile_events_ok = False
        self._pending.clear()


@dataclass
class StreamingParams:
    """Tunable parameters for :meth:`RVCPipeline.infer_streaming`.

    Folds the (previously 17) per-call keyword arguments into a single object
    so callers can build one config-derived params bundle instead of
    transcribing every argument at each call site. Defaults mirror the
    historical ``infer_streaming`` keyword defaults, so existing callers that
    pass individual keywords keep working unchanged (they are collected into
    ``**overrides`` and used to build a ``StreamingParams``).
    """

    pitch_shift: int = 0
    f0_method: str = "fcpe"
    index_rate: float = 0.0
    index_k: int = 4
    voice_gate_mode: str = "off"
    energy_threshold: float = 0.05
    use_parallel_extraction: bool = True
    noise_scale: float = 0.66666
    sola_extra_samples: int = 0
    moe_boost: float = 0.0
    f0_lowpass_cutoff_hz: float = 16.0
    enable_octave_flip_suppress: bool = True
    enable_f0_slew_limit: bool = True
    f0_slew_max_step_st: float = 2.8
    hubert_context_sec: float = 1.0
    fixed_harmonics: bool = False
    # F0 extraction window: leading context (sec) before the new hop.
    # Pitch is local, so the F0 model only needs [context | new_hop] instead
    # of the full HuBERT history; older frames are served from the streaming
    # F0 cache.  <= 0 disables windowing (extract on the full context).
    f0_context_sec: float = 0.32
    # Longest unvoiced hole (ms) inside a voiced run to fill by interpolation
    # before synthesis (prevents noise-excitation bursts / raspiness).
    # <= 0 disables hole filling.
    f0_hole_fill_ms: float = 30.0
    # Voiced/unvoiced crossfade ramp (ms) for the NSF sine/noise excitation
    # switch.  0 keeps the original RVC hard switch.
    uv_ramp_ms: float = 5.0


def highpass_filter(audio: np.ndarray, sr: int = 16000, cutoff: int = 48) -> np.ndarray:
    """Apply high-pass filter to remove DC offset and low-frequency noise.

    Original RVC uses 5th order Butterworth filter with 48Hz cutoff at 16kHz.
    """
    if len(audio) < 100:  # Too short to filter
        return audio
    nyquist = sr / 2
    normalized_cutoff = cutoff / nyquist
    b, a = butter(5, normalized_cutoff, btype="high")
    return filtfilt(b, a, audio).astype(np.float32)


def sigmoid_blend_weights(steps: int, steepness: float = 4.0) -> np.ndarray:
    """Generate sigmoid-shaped blend weights for smoother transitions.

    Args:
        steps: Number of blend steps
        steepness: Controls the sharpness of the transition (4.0 = smooth S-curve)

    Returns:
        Array of weights from ~1.0 to ~0.0 (sigmoid curve)
    """
    x = np.linspace(-steepness, steepness, steps)
    return 1.0 / (1.0 + np.exp(x))


def smooth_f0_spikes(f0: torch.Tensor, window: int = 3) -> torch.Tensor:
    """Remove F0 spikes using median filter on voiced regions.

    Args:
        f0: F0 tensor [B, T]
        window: Median filter window size (odd number, default 3)

    Returns:
        Smoothed F0 tensor with spikes removed
    """
    if f0.shape[1] < window:
        return f0

    # Convert to numpy for scipy median filter (must be float32 for medfilt)
    f0_np = f0.cpu().to(torch.float32).numpy()
    result = np.zeros_like(f0_np)

    for b in range(f0_np.shape[0]):
        # Get voiced mask (f0 > 0)
        voiced = f0_np[b] > 0

        # Apply median filter only to voiced regions
        if np.any(voiced):
            # medfilt preserves array length
            filtered = medfilt(f0_np[b].astype(np.float64), window).astype(np.float32)

            # Only apply to voiced regions (keep unvoiced as 0)
            result[b] = np.where(voiced, filtered, f0_np[b])
        else:
            result[b] = f0_np[b]

    return torch.from_numpy(result).to(f0.device, dtype=f0.dtype)


def lowpass_f0(f0: torch.Tensor, cutoff_hz: float = 16.0, sample_rate: float = 100.0) -> torch.Tensor:
    """Apply lowpass filter to F0 to remove high-frequency jitter.

    Phase 6: Butterworth lowpass filter for smoother F0 contour.
    Only applies to voiced regions to preserve unvoiced detection.

    Args:
        f0: F0 tensor [B, T] in Hz
        cutoff_hz: Cutoff frequency in Hz (default 8Hz - removes jitter above ~8Hz)
        sample_rate: F0 sample rate in Hz (default 100fps)

    Returns:
        Lowpass filtered F0 tensor
    """
    if f0.shape[1] < 10:  # Too short to filter
        return f0

    # Convert to numpy
    f0_np = f0.cpu().to(torch.float32).numpy()
    result = np.zeros_like(f0_np)

    # Design lowpass filter
    nyquist = sample_rate / 2
    normalized_cutoff = cutoff_hz / nyquist
    # Use order 2 for gentler filtering
    b, a = butter(2, normalized_cutoff, btype="low")

    for batch in range(f0_np.shape[0]):
        voiced = f0_np[batch] > 0

        if np.sum(voiced) > 10:  # Need enough voiced samples
            # Build a gap-free contour in log2 domain (pitch is log-scale):
            # np.interp fills interior unvoiced gaps AND extends the leading/
            # trailing unvoiced regions with the first/last voiced value.
            # Leaving those edges at 0 Hz would feed a 0->F0 step into the
            # zero-phase filter, smearing an onset scoop / offset droop tens
            # of ms into the voiced region.
            idx = np.arange(f0_np.shape[1])
            voiced_indices = idx[voiced]
            log_contour = np.interp(
                idx, voiced_indices, np.log2(f0_np[batch, voiced_indices])
            )

            # Apply lowpass filter on the log contour
            try:
                filtered = filtfilt(b, a, log_contour)
                # Only keep filtered values in voiced regions
                result[batch] = np.where(
                    voiced, np.exp2(filtered).astype(np.float32), 0.0
                )
            except ValueError:
                # Filter failed, return original
                result[batch] = f0_np[batch]
        else:
            result[batch] = f0_np[batch]

    return torch.from_numpy(result).to(f0.device, dtype=f0.dtype)


def _smooth_fcpe_f0(f0: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Smooth FCPE F0 with a voiced-mask-aware log-domain moving average.

    Unvoiced frames (f0=0) mean "no pitch", not "0 Hz": including them in a
    plain average drags boundary frames down by several semitones (and FCPE
    NaN->0 holes make this constant).  Averaging only voiced neighbors, in
    log2 domain (pitch is log-scale), fixes both.
    """
    voiced = (f0 > 0).to(f0.dtype)
    log_f0 = torch.where(
        f0 > 0, torch.log2(torch.clamp(f0, min=1e-3)), torch.zeros_like(f0)
    )
    num = torch.nn.functional.avg_pool1d(
        (log_f0 * voiced).unsqueeze(1),
        kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
    ).squeeze(1)
    den = torch.nn.functional.avg_pool1d(
        voiced.unsqueeze(1),
        kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
    ).squeeze(1)
    f0_smooth = torch.exp2(num / torch.clamp(den, min=1e-6))
    return torch.where(f0 > 0, f0_smooth, f0)


def _resize_f0_track(f0: torch.Tensor, target_frames: int) -> torch.Tensor:
    """Resize an F0 track to a new frame count, voiced-mask aware.

    Values are interpolated in log2 domain over a gap-free contour (interior
    gaps and edges filled from the nearest voiced values), so no frame is
    ever averaged with an unvoiced 0.  The voiced mask is resized separately
    and re-applied, so boundary frames never become spurious low pitches.
    """
    if f0.shape[1] == target_frames:
        return f0

    f0_np = f0.detach().cpu().to(torch.float32).numpy()
    n = f0_np.shape[1]
    out = np.zeros((f0_np.shape[0], target_frames), dtype=np.float32)
    src_pos = np.arange(n, dtype=np.float64)
    dst_pos = np.linspace(0.0, n - 1, target_frames)

    for b in range(f0_np.shape[0]):
        voiced = f0_np[b] > 0
        if not voiced.any():
            continue
        voiced_idx = src_pos[voiced]
        log_contour = np.interp(src_pos, voiced_idx, np.log2(f0_np[b][voiced]))
        resized = np.exp2(np.interp(dst_pos, src_pos, log_contour))
        mask = np.interp(dst_pos, src_pos, voiced.astype(np.float64)) >= 0.5
        out[b] = np.where(mask, resized, 0.0).astype(np.float32)

    return torch.from_numpy(out).to(f0.device, dtype=f0.dtype)


def _align_f0_frames(f0: torch.Tensor, target_frames: int) -> torch.Tensor:
    """Align an F0 track to a target frame count without a timeline squeeze.

    When the counts differ by only a few frames the track has the SAME frame
    rate as the target grid (the difference is edge frames), so a uniform
    resize would compress the whole timeline and misplace pitch near the
    tail; instead crop / edge-pad on the right so frame k stays at time k.
    A genuine rate mismatch (>5%) is resized mask-aware in log domain.
    """
    cur = f0.shape[1]
    if cur == target_frames:
        return f0
    if abs(cur - target_frames) > max(2, target_frames // 20):
        return _resize_f0_track(f0, target_frames)
    if cur > target_frames:
        return f0[:, :target_frames]
    return torch.cat([f0, f0[:, -1:].expand(-1, target_frames - cur)], dim=1)


def apply_moe_f0_style(f0: torch.Tensor, strength: float) -> torch.Tensor:
    """Apply F0-only moe stylization for brighter, feminine-leaning prosody.

    Design goals:
    - lift only low-register contours toward a safer target median F0,
    - preserve and enhance upward accents while softening downward dips,
    - fill short unvoiced gaps to reduce raspy flicker,
    - avoid touching HuBERT input audio, which is fragile under pitch warps.

    ``strength`` sets the coefficient magnitudes (how aggressive the target is);
    a separate per-window voiced-ratio ``confidence`` blends the styled contour
    toward the raw one.  Keeping these two axes distinct (instead of folding
    ``strength * confidence`` into every coefficient) makes the mapping
    continuous as confidence -> 0, so a marginally-voiced streaming chunk cannot
    snap its frames to the register floor and produce an audible pitch step at
    chunk boundaries.
    """
    strength = max(0.0, min(MAX_MOE_BOOST, float(strength)))
    if strength <= 0.0 or f0.numel() == 0:
        return f0

    def _fill_short_unvoiced_gaps(row: torch.Tensor, max_gap_frames: int) -> torch.Tensor:
        """Interpolate short unvoiced gaps to reduce frame-level F0 dropouts."""
        if max_gap_frames <= 0 or row.numel() == 0:
            return row
        filled = row.clone()
        voiced = filled > 0
        n = int(filled.shape[0])
        i = 0
        while i < n:
            if voiced[i]:
                i += 1
                continue
            start = i
            while i < n and not voiced[i]:
                i += 1
            end = i
            gap = end - start
            if (
                gap > 0
                and gap <= max_gap_frames
                and start > 0
                and end < n
                and voiced[start - 1]
                and voiced[end]
            ):
                left = filled[start - 1]
                right = filled[end]
                w = torch.linspace(
                    0.0, 1.0, gap + 2, device=filled.device, dtype=filled.dtype
                )[1:-1]
                filled[start:end] = left * (1.0 - w) + right * w
        return filled

    # Conservative F0-only mapping: keep register lift bounded and make
    # expressiveness mostly contour shaping rather than a large constant
    # pitch jump.  All coefficients depend only on ``strength`` (constant over
    # the batch), so they are computed once; ``confidence`` enters later purely
    # as a blend weight.
    max_gap_frames = int(round(2 + 4 * strength))  # 20-60ms @ 100fps
    target_median_hz = 165.0 + 55.0 * strength  # 165-220Hz
    max_up_shift_st = 1.5 + 4.5 * strength      # +1.5..+6 semitones
    up_gain = 1.0 + 0.45 * strength
    down_gain = 1.0 - 0.25 * strength
    phrase_bias_st = 0.10 + 0.45 * strength
    sat_coef = 0.08 + 0.16 * strength
    window = max(7, int(7 + 14 * strength))
    if window % 2 == 0:
        window += 1
    floor_rel = target_median_hz * (0.55 + 0.08 * strength)
    floor_abs = 85.0 + 45.0 * strength
    floor_hz = float(np.clip(max(floor_rel, floor_abs), 85.0, 220.0))
    max_dev_st = 22.0

    styled = f0.clone()
    for b in range(styled.shape[0]):
        row = _fill_short_unvoiced_gaps(styled[b], max_gap_frames=max_gap_frames)
        voiced = row > 0
        n_voiced = int(voiced.sum().item())
        if n_voiced < 6:
            styled[b] = row
            continue

        # Confidence is a blend weight only: styling fades smoothly to identity
        # on low-voiced windows instead of switching off abruptly, so adjacent
        # streaming chunks with slightly different voiced ratios stay continuous.
        voiced_ratio = n_voiced / float(max(1, row.numel()))
        confidence = max(0.0, min(1.0, (voiced_ratio - 0.12) / 0.50))
        if confidence <= 1e-4:
            styled[b] = row
            continue

        raw_voiced = row[voiced]
        voiced_f0 = torch.clamp(raw_voiced, min=1e-5, max=1400.0)
        log2_f0 = torch.log2(voiced_f0)

        # Local trend (phrase-level baseline at F0 frame rate ~100Hz).
        trend_input = torch.nn.functional.pad(
            log2_f0.view(1, 1, -1), (window // 2, window // 2), mode="replicate"
        )
        trend = torch.nn.functional.avg_pool1d(
            trend_input, kernel_size=window, stride=1
        ).view(-1)

        median_hz = torch.median(voiced_f0)
        raw_shift_st = 12.0 * torch.log2(
            torch.tensor(target_median_hz, device=row.device, dtype=row.dtype)
            / torch.clamp(median_hz, min=1e-5)
        )
        reg_shift_st = torch.clamp(raw_shift_st, min=0.0, max=max_up_shift_st)

        dev_st = (log2_f0 - trend) * 12.0
        dev_st = torch.where(dev_st >= 0, dev_st * up_gain, dev_st * down_gain)

        # Soft saturation prevents extreme overshoot at high boost.
        sat = 1.0 + sat_coef * torch.abs(dev_st)
        dev_st = torch.clamp(dev_st / sat, -max_dev_st, max_dev_st)

        shaped_log2 = trend + (dev_st + reg_shift_st + phrase_bias_st) / 12.0
        shaped = torch.exp2(shaped_log2)

        # Keep low floor above chesty dips; this is critical for male->female tilt.
        shaped = torch.maximum(
            shaped, torch.tensor(floor_hz, device=row.device, dtype=row.dtype)
        )
        shaped = torch.clamp(shaped, min=0.0, max=940.0)

        # Blend styled -> raw by confidence (continuous as confidence -> 0).
        row[voiced] = raw_voiced * (1.0 - confidence) + shaped * confidence
        styled[b] = row

    return styled


def suppress_octave_flips(
    f0: torch.Tensor,
    octave_ratio_center: float = 2.0,
    octave_ratio_tolerance: float = 0.16,
    sustain_frames: int = F0_CORRECTION_SUSTAIN_FRAMES,
) -> torch.Tensor:
    """Suppress transient +-1 octave frame-to-frame F0 flips.

    Corrections are made relative to the previous *corrected* frame, so a
    genuine octave transition would keep matching the flip band forever and
    get halved/doubled indefinitely.  The sustain counter breaks that latch:
    once the correction persists ``sustain_frames`` consecutive frames in
    the same direction, the raw values are restored and tracking resumes at
    the new octave.
    """
    if f0.numel() == 0 or f0.shape[1] < 2:
        return f0

    f0_np = f0.detach().cpu().to(torch.float32).numpy()
    corrected = f0_np.copy()
    low = octave_ratio_center - octave_ratio_tolerance
    high = octave_ratio_center + octave_ratio_tolerance
    inv_low = 1.0 / high
    inv_high = 1.0 / low

    for b in range(corrected.shape[0]):
        streak_dir = 0
        streak_start = 0
        streak_len = 0
        for i in range(1, corrected.shape[1]):
            prev = float(corrected[b, i - 1])
            cur = float(f0_np[b, i])
            # Only compare adjacent voiced frames.
            # Never carry references across unvoiced gaps.
            if prev <= 0.0 or cur <= 0.0:
                streak_dir = 0
                streak_len = 0
                continue

            ratio = cur / max(prev, 1e-6)
            if low <= ratio <= high:
                direction = 1
                corrected[b, i] = cur * 0.5
            elif inv_low <= ratio <= inv_high:
                direction = -1
                corrected[b, i] = cur * 2.0
            else:
                streak_dir = 0
                streak_len = 0
                continue

            if direction == streak_dir:
                streak_len += 1
            else:
                streak_dir = direction
                streak_start = i
                streak_len = 1

            if streak_len >= sustain_frames:
                # Sustained -> genuine transition: accept the raw octave.
                corrected[b, streak_start : i + 1] = f0_np[b, streak_start : i + 1]
                streak_dir = 0
                streak_len = 0

    return torch.from_numpy(corrected).to(f0.device, dtype=f0.dtype)


def limit_f0_slew(
    f0: torch.Tensor,
    max_step_st: float = 2.8,
    sustain_frames: int = F0_CORRECTION_SUSTAIN_FRAMES,
) -> torch.Tensor:
    """Limit frame-to-frame F0 step to suppress short flutter artifacts.

    A genuine pitch jump would otherwise be turned into a multi-frame
    portamento (clamped max_step_st per frame until it catches up).  When
    the clamp persists ``sustain_frames`` consecutive frames in the same
    direction, the raw values are restored — flutter is 1-2 frames; anything
    longer is the singer actually moving.
    """
    if f0.numel() == 0 or f0.shape[1] < 2:
        return f0

    f0_np = f0.detach().cpu().to(torch.float32).numpy()
    corrected = f0_np.copy()
    max_ratio = float(2 ** (max_step_st / 12.0))

    for b in range(corrected.shape[0]):
        streak_dir = 0
        streak_start = 0
        streak_len = 0
        for i in range(1, corrected.shape[1]):
            prev = float(corrected[b, i - 1])
            cur = float(f0_np[b, i])
            if prev <= 0.0 or cur <= 0.0:
                streak_dir = 0
                streak_len = 0
                continue

            upper = prev * max_ratio
            lower = prev / max_ratio
            if cur > upper:
                direction = 1
                corrected[b, i] = upper
            elif cur < lower:
                direction = -1
                corrected[b, i] = lower
            else:
                corrected[b, i] = cur
                streak_dir = 0
                streak_len = 0
                continue

            if direction == streak_dir:
                streak_len += 1
            else:
                streak_dir = direction
                streak_start = i
                streak_len = 1

            if streak_len >= sustain_frames:
                # Sustained -> genuine jump: accept the raw values.
                corrected[b, streak_start : i + 1] = f0_np[b, streak_start : i + 1]
                streak_dir = 0
                streak_len = 0

    return torch.from_numpy(corrected).to(f0.device, dtype=f0.dtype)


def stabilize_f0_boundaries(
    f0: torch.Tensor,
    edge_frames: int = 10,
) -> torch.Tensor:
    """Stabilize boundary F0 frames in batch mode to reduce edge flutter.

    Only voiced frames near the start/end are smoothed. Unvoiced regions remain 0.
    """
    if f0.numel() == 0 or f0.shape[1] < 4 or edge_frames <= 0:
        return f0

    corrected = f0.clone()
    n = int(corrected.shape[1])

    for b in range(corrected.shape[0]):
        row = corrected[b]
        voiced = row > 0
        if voiced.sum().item() < 4:
            continue

        nz = torch.where(voiced)[0]
        first = int(nz[0].item())
        last = int(nz[-1].item())

        if first <= edge_frames:
            search_end = min(n, first + edge_frames * 3)
            stable_vals = row[first:search_end]
            stable_voiced = stable_vals[stable_vals > 0]
            if stable_voiced.numel() > 0:
                target = torch.median(stable_voiced)
                blend_end = min(n, first + edge_frames)
                if blend_end > first:
                    m = blend_end - first
                    w = torch.linspace(0.0, 1.0, m, device=row.device, dtype=row.dtype)
                    seg = row[first:blend_end]
                    row[first:blend_end] = target * (1.0 - w) + seg * w

        if last >= n - 1 - edge_frames:
            search_start = max(0, last - edge_frames * 3 + 1)
            stable_vals = row[search_start:last + 1]
            stable_voiced = stable_vals[stable_vals > 0]
            if stable_voiced.numel() > 0:
                target = torch.median(stable_voiced)
                blend_start = max(0, last - edge_frames + 1)
                if last + 1 > blend_start:
                    m = last + 1 - blend_start
                    w = torch.linspace(0.0, 1.0, m, device=row.device, dtype=row.dtype)
                    seg = row[blend_start:last + 1]
                    row[blend_start:last + 1] = seg * (1.0 - w) + target * w

        corrected[b] = row

    return corrected


def fill_short_unvoiced_gaps(f0: torch.Tensor, max_gap_frames: int = 3) -> torch.Tensor:
    """Fill short unvoiced holes inside voiced runs by log2 interpolation.

    F0 extractors occasionally drop single frames to 0 mid-vowel (FCPE NaN
    frames, RMVPE threshold flicker).  The NSF decoder switches those frames
    to pure noise excitation (sine_amp/3 vs noise_std when voiced), which is
    audible as raspiness.  Only holes bounded by voiced frames on BOTH sides
    and no longer than ``max_gap_frames`` are filled — longer gaps are true
    unvoiced consonants / silence and must stay unvoiced.

    Args:
        f0: F0 tensor [B, T] in Hz, 0 = unvoiced
        max_gap_frames: Longest hole to fill (frames @100fps; 3 = 30ms)

    Returns:
        F0 tensor with short holes filled
    """
    if max_gap_frames <= 0 or f0.shape[1] < 3:
        return f0

    # Cheap on-device gate: a fillable hole needs >= 2 voiced frames with at
    # least one unvoiced frame between them.  The common steady-voiced and
    # silent cases (all-voiced / <2 voiced) short-circuit here and skip the
    # host transfer + Python loop below — the only forced sync in the F0
    # chain.  One scalar reduction is far cheaper than moving the full tensor.
    voiced = f0 > 0
    n_voiced = int(voiced.sum())
    if n_voiced < 2 or n_voiced == f0.numel():
        return f0

    f0_np = f0.detach().cpu().to(torch.float32).numpy().copy()
    changed = False
    for b in range(f0_np.shape[0]):
        row = f0_np[b]
        vidx = np.flatnonzero(row > 0)
        if vidx.size < 2:
            continue
        # Unvoiced frame count between consecutive voiced frames
        gaps = np.diff(vidx) - 1
        for gi in np.flatnonzero((gaps > 0) & (gaps <= max_gap_frames)):
            left = int(vidx[gi])
            right = int(vidx[gi + 1])
            n = right - left
            t = np.arange(1, n, dtype=np.float32) / n
            lv = np.log2(row[left])
            rv = np.log2(row[right])
            row[left + 1:right] = np.exp2(lv * (1.0 - t) + rv * t)
            changed = True

    if not changed:
        return f0
    return torch.from_numpy(f0_np).to(device=f0.device, dtype=f0.dtype)


def apply_f0_filter_chain(
    f0: torch.Tensor,
    *,
    f0_lowpass_cutoff_hz: float,
    enable_octave_flip_suppress: bool,
    enable_f0_slew_limit: bool,
    f0_slew_max_step_st: float,
    f0_hole_fill_ms: float = 30.0,
) -> torch.Tensor:
    """Apply the shared F0 post-processing chain.

    short-hole fill -> median spike removal -> lowpass -> (optional)
    octave-flip suppression -> (optional) slew limiting. Used identically by
    ``infer`` (on the final F0) and ``infer_streaming`` (on the
    history-extended F0).
    """
    f0 = fill_short_unvoiced_gaps(f0, max_gap_frames=int(round(f0_hole_fill_ms / 10.0)))
    f0 = smooth_f0_spikes(f0, window=3)
    f0 = lowpass_f0(f0, cutoff_hz=f0_lowpass_cutoff_hz, sample_rate=100.0)
    if enable_octave_flip_suppress:
        f0 = suppress_octave_flips(f0)
    if enable_f0_slew_limit:
        f0 = limit_f0_slew(f0, max_step_st=f0_slew_max_step_st)
    return f0


def quantize_f0_to_pitch(f0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize continuous F0 (Hz) to RVC's 256-bin mel pitch index.

    Mirrors the original RVC WebUI quantization exactly: F0 -> mel scale ->
    normalize voiced frames (f0_mel > 0) to 1..255; unvoiced stays 0 pre-clamp
    and becomes 1 after the final clamp (RVC's unvoiced marker).

    Returns:
        (pitch, voiced_mask) where ``pitch`` is int64 [B, T] in 1..255 and
        ``voiced_mask`` is float [B, T] (1.0 where voiced, else 0.0).
    """
    f0_mel_min = 1127 * math.log(1 + 50 / 700)  # ~69.07 (50Hz)
    f0_mel_max = 1127 * math.log(1 + 1100 / 700)  # ~942.46 (1100Hz)
    f0_mel = 1127 * torch.log(1 + f0 / 700)
    voiced_mask = f0_mel > 0
    f0_mel_normalized = torch.where(
        voiced_mask,
        (f0_mel - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1,
        f0_mel,  # Keep 0 for unvoiced
    )
    pitch = torch.clamp(f0_mel_normalized, 1, 255).round().long()
    return pitch, voiced_mask.float()


def apply_voice_gate(
    output: torch.Tensor,
    gate_mask_src: torch.Tensor,
    *,
    voice_gate_mode: str,
    energy_threshold: float,
    sample_rate: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the shared voice gate to synthesized output.

    ``gate_mask_src`` is the per-feature-frame voiced mask, already sliced by
    the caller to the region that ``output`` covers (batch ``infer`` passes the
    full mask; ``infer_streaming`` passes the output-region slice).

    Modes:
        - "expand": dilate voiced regions (~30ms) to include adjacent plosives.
        - "energy": OR the voiced mask with an energy-threshold mask.
        - anything else (e.g. "strict"): use the F0 voiced mask as-is.

    The mask is upsampled to ``output`` length and given a 5ms attack/release
    smooth to avoid clicks.

    Returns:
        (gated_output, gate_mask) — ``gate_mask`` is returned so the caller can
        compute the passed ratio for logging without this helper forcing a
        GPU->CPU sync on the hot streaming path.
    """
    output_len = output.shape[-1]
    gate_mask = gate_mask_src.clone()

    if voice_gate_mode == "expand":
        # Expand by ~30ms on each side (covers most plosives); at feature rate
        # (~50fps) that's ~1-2 frames. Max pooling dilates the mask.
        expand_frames = 2
        gate_mask = torch.nn.functional.max_pool1d(
            gate_mask.unsqueeze(1),
            kernel_size=expand_frames * 2 + 1,
            stride=1,
            padding=expand_frames,
        ).squeeze(1)
    elif voice_gate_mode == "energy":
        # Combine F0 voicing with short-time energy at the feature frame rate.
        frame_size = output_len // gate_mask.shape[-1]
        if frame_size > 0:
            output_frames = output.unfold(-1, frame_size, frame_size)
            frame_energy = (output_frames ** 2).mean(dim=-1)
            energy_max = frame_energy.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
            energy_mask = (frame_energy / energy_max > energy_threshold).float()
            if energy_mask.shape[-1] == gate_mask.shape[-1]:
                gate_mask = torch.maximum(gate_mask, energy_mask)

    # Upsample mask to match output length
    gate_mask = torch.nn.functional.interpolate(
        gate_mask.unsqueeze(1),
        size=output_len,
        mode="linear",
        align_corners=False,
    ).squeeze(1)

    # Smooth attack/release (5ms) to avoid clicks
    smooth_samples = int(sample_rate * 0.005)
    if smooth_samples > 1:
        kernel = torch.ones(1, 1, smooth_samples, device=gate_mask.device) / smooth_samples
        gate_mask = torch.nn.functional.conv1d(
            gate_mask.unsqueeze(1),
            kernel,
            padding=smooth_samples // 2,
        ).squeeze(1)
        if gate_mask.shape[-1] != output_len:
            gate_mask = gate_mask[..., :output_len]
        gate_mask = torch.clamp(gate_mask, 0, 1)

    return output * gate_mask, gate_mask


def apply_output_edge_fade(
    audio: np.ndarray,
    sample_rate: int,
    fade_ms: float = 8.0,
) -> np.ndarray:
    """Apply a short fade-in/out to suppress batch boundary clicks."""
    if audio.ndim != 1:
        return audio
    fade = int(sample_rate * max(0.0, float(fade_ms)) / 1000.0)
    if fade <= 1 or len(audio) <= 2 * fade:
        return audio

    out = audio.copy()
    ramp = np.linspace(0.0, 1.0, fade, dtype=out.dtype)
    out[:fade] *= ramp
    out[-fade:] *= ramp[::-1]
    return out


class RVCPipeline:
    """
    Complete RVC voice conversion pipeline.

    Integrates HuBERT feature extraction, optional RMVPE F0 extraction,
    FAISS index retrieval, and synthesizer inference.
    """

    def __init__(
        self,
        model_path: str,
        index_path: Optional[str] = None,
        device: str = "auto",
        dtype: str = "float16",
        use_f0: bool = True,
        use_compile: bool = True,
        models_dir: Optional[str] = None,
    ):
        """
        Initialize the RVC pipeline.

        Args:
            model_path: Path to the RVC .pth model
            index_path: Path to FAISS .index file (optional, auto-detected if None)
            device: Device preference (auto, xpu, cuda, cpu)
            dtype: Data type (float16, float32, bfloat16)
            use_f0: Whether to use F0 extraction (if model supports it)
            use_compile: Whether to use torch.compile optimization
            models_dir: Directory containing HuBERT and RMVPE models
        """
        self.model_path = Path(model_path)

        # Auto-detect index file if not provided
        if index_path:
            self.index_path = Path(index_path)
        else:
            # Look for .index file in same directory as model
            index_candidates = list(self.model_path.parent.glob("*.index"))
            self.index_path = index_candidates[0] if index_candidates else None

        # Resolve device and dtype
        self.device = get_device(device)
        self.dtype = get_dtype(self.device, dtype)

        # torch.compile: XPU uses inductor backend (no Triton needed)
        # CUDA on Windows needs Triton, which is not supported
        if use_compile and sys.platform == "win32" and self.device == "cuda":
            logger.info("torch.compile disabled for CUDA on Windows (Triton not supported)")
            use_compile = False
        self.use_compile = use_compile

        # Model directory for HuBERT/RMVPE
        if models_dir:
            self.models_dir = Path(models_dir)
        else:
            self.models_dir = Path.home() / ".cache" / "rcwx" / "models"

        # Components (initialized lazily)
        self.hubert: Optional[HuBERTLoader] = None
        self.rmvpe: Optional[RMVPE] = None
        self.fcpe: Optional[FCPE] = None
        self.swiftf0: Optional[SwiftF0Model] = None
        self.synthesizer: Optional[SynthesizerLoader] = None

        # FAISS index components
        self.faiss_index = None
        self.index_features: Optional[np.ndarray] = None  # big_npy

        # Model properties
        self.has_f0: bool = use_f0
        self.sample_rate: int = 40000
        self._loaded: bool = False

        # Feature cache for chunk continuity (HuBERT/F0 boundary blending)
        self._feature_cache: Optional[torch.Tensor] = None  # [1, T_cache, C]
        self._f0_cache: Optional[torch.Tensor] = None  # [1, T_cache]
        self._f0_voiced_cache: Optional[torch.Tensor] = None  # [1, T_cache] bool
        # Cache lengths (frames). Can be tuned by realtime controller.
        self._feature_cache_frames: int = 20  # HuBERT frames @ 50fps
        self._f0_cache_frames: int = 40  # F0 frames @ 100fps

        # Phase 5: Audio-level overlap cache for F0/HuBERT extraction
        # Store the tail of each audio chunk to prepend to the next chunk
        # This allows F0/HuBERT to see continuous audio across boundaries
        self._audio_cache: Optional[np.ndarray] = None  # [T_audio] at 16kHz
        self._audio_cache_len: int = 3200  # 200ms at 16kHz (covers F0/HuBERT receptive fields)
        # Streaming history buffer for HuBERT context
        self._stream_history: Optional[np.ndarray] = None  # [T_audio] at 16kHz

        # Phase 8: Output overlap cache for smooth chunk boundaries
        # Store the tail of synthesizer output for crossfade with next chunk
        # This enables overlap-add blending at the audio output level
        self._output_cache: Optional[np.ndarray] = None  # [T_output] at model sample rate
        self._output_overlap_len: int = 0  # Set dynamically based on crossfade_sec

        # Streaming caches used exclusively by infer_streaming().  These are
        # (re)set by clear_cache(); initialize them here too so infer_streaming
        # works even if called before the first clear_cache()/warmup.
        self._streaming_feat_cache = None
        self._streaming_audio_history: Optional[np.ndarray] = None
        self._streaming_f0_pre_filter_tail: Optional[torch.Tensor] = None
        # Windowed-F0 history: post-shift/moe, pre-filter F0 (Hz) @100fps,
        # kept in lockstep with the tail of _streaming_audio_history.
        self._streaming_f0_hz_history: Optional[torch.Tensor] = None

        # Per-stage times (ms) of the last profiled infer_streaming() call.
        self.stage_times: dict = {}
        self._stage_profile_counter: int = 0
        # Device timing-event support: None=unknown, resolved on first use.
        self._profile_events_ok: Optional[bool] = None

    def load(self) -> None:
        """Load all models."""
        if self._loaded:
            return

        # Set deterministic behavior for reproducible inference
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.manual_seed_all(0)

        logger.info(f"Loading RVC pipeline on {self.device} with {self.dtype}")

        # Load synthesizer first to detect model type
        self.synthesizer = SynthesizerLoader(
            str(self.model_path),
            device=self.device,
            dtype=self.dtype,
            use_compile=self.use_compile,
        )
        self.synthesizer.load()

        # Update properties based on loaded model
        self.sample_rate = self.synthesizer.sample_rate
        self.has_f0 = self.synthesizer.has_f0 and self.has_f0

        # Load HuBERT
        hubert_path = get_hubert_path(self.models_dir)
        logger.info(f"Loading HuBERT from: {hubert_path}")

        # Use the new loader that handles both RVC and transformers formats
        self.hubert = HuBERTLoader(
            str(hubert_path) if hubert_path.exists() else None,
            device=self.device,
            dtype=self.dtype,
        )

        if self.use_compile and hasattr(self.hubert, "model"):
            logger.info("Compiling HuBERT model...")
            self.hubert.model = torch.compile(self.hubert.model, mode="reduce-overhead")

        # Load F0 models if F0 is used
        if self.has_f0:
            # Load RMVPE
            rmvpe_path = get_rmvpe_path(self.models_dir)
            if rmvpe_path.exists():
                self.rmvpe = RMVPE(
                    str(rmvpe_path),
                    device=self.device,
                    dtype=self.dtype,
                )
                if self.use_compile:
                    logger.info("Compiling RMVPE model...")
                    self.rmvpe.model = torch.compile(self.rmvpe.model, mode="reduce-overhead")
            else:
                logger.warning("RMVPE model not found")

            # Load FCPE if available (lightweight alternative)
            if is_fcpe_available():
                try:
                    self.fcpe = FCPE(
                        device=self.device,
                        dtype=self.dtype,
                    )
                    logger.info("FCPE model loaded (low-latency F0 available)")
                except Exception as e:
                    logger.warning(f"Failed to load FCPE: {e}")
                    self.fcpe = None
            else:
                logger.info("FCPE not available (install with: pip install torchfcpe)")

            # Load SwiftF0 if available (lightweight ONNX/CPU alternative)
            if is_swiftf0_available():
                try:
                    self.swiftf0 = SwiftF0Model(
                        confidence_threshold=SWIFTF0_VOICING_THRESHOLD,
                    )
                    logger.info("SwiftF0 model loaded (ultra-fast ONNX/CPU F0 available)")
                except Exception as e:
                    logger.warning(f"Failed to load SwiftF0: {e}")
                    self.swiftf0 = None
            else:
                logger.info("SwiftF0 not available (install with: pip install swift-f0)")

            # Disable F0 if no model is available
            if self.rmvpe is None and self.fcpe is None and self.swiftf0 is None:
                logger.warning("No F0 model available, F0 extraction disabled")
                self.has_f0 = False

        # Load FAISS index if available
        if self.index_path and self.index_path.exists():
            self._load_faiss_index()

        self._loaded = True
        logger.info("RVC pipeline loaded successfully")

    def _load_faiss_index(self) -> None:
        """Load FAISS index for feature retrieval."""
        try:
            import faiss

            logger.info(f"Loading FAISS index from: {self.index_path}")
            self.faiss_index = faiss.read_index(str(self.index_path))

            # Reconstruct all feature vectors from the index
            # These are used for weighted averaging during retrieval
            self.index_features = self.faiss_index.reconstruct_n(0, self.faiss_index.ntotal)
            logger.info(
                f"FAISS index loaded: {self.faiss_index.ntotal} vectors, "
                f"dim={self.index_features.shape[1]}"
            )
        except ImportError:
            logger.warning("faiss-cpu not installed, index retrieval disabled")
            self.faiss_index = None
            self.index_features = None
        except Exception as e:
            logger.warning(f"Failed to load FAISS index: {e}")
            self.faiss_index = None
            self.index_features = None

    def _search_index(
        self,
        features: torch.Tensor,
        index_rate: float = 0.5,
        k: int = 4,
    ) -> torch.Tensor:
        """
        Search FAISS index and blend retrieved features with original.

        Args:
            features: HuBERT features [B, T, C]
            index_rate: Blending ratio (0=original only, 1=index only)
            k: Number of nearest neighbors to retrieve (4=fast, 8=quality)

        Returns:
            Blended features [B, T, C]
        """
        if self.faiss_index is None or index_rate <= 0:
            return features

        # Convert to numpy for FAISS search
        npy = features[0].cpu().numpy()
        if self.dtype == torch.float16:
            npy = npy.astype(np.float32)

        # Search for k nearest neighbors (k=4 for speed, k=8 for quality)
        logger.debug(
            f"FAISS search: input shape={npy.shape}, k={k}, index_vectors={self.faiss_index.ntotal}"
        )
        score, ix = self.faiss_index.search(npy, k=k)
        logger.debug(
            f"FAISS search results: scores shape={score.shape}, min={score.min():.4f}, max={score.max():.4f}"
        )

        # Compute inverse squared distance weights
        # Add small epsilon to avoid division by zero
        weight = np.square(1 / (score + 1e-6))
        weight /= weight.sum(axis=1, keepdims=True)

        # Weighted average of retrieved features
        # index_features[ix] has shape [T, k, C]
        # weight has shape [T, k]
        retrieved = np.sum(
            self.index_features[ix] * np.expand_dims(weight, axis=2),
            axis=1,
        )  # [T, C]
        logger.debug(f"Retrieved features: mean={retrieved.mean():.4f}, std={retrieved.std():.4f}")

        if self.dtype == torch.float16:
            retrieved = retrieved.astype(np.float16)

        # Blend with original features
        retrieved_tensor = torch.from_numpy(retrieved).unsqueeze(0).to(features.device)
        blended = index_rate * retrieved_tensor + (1 - index_rate) * features
        logger.debug(
            f"Blended features: mean={blended.mean():.4f}, std={blended.std():.4f}, index_rate={index_rate}"
        )

        return blended

    def unload(self) -> None:
        """Unload all models to free memory."""
        if self.hubert is not None:
            self.hubert.clear_graph_cache()
        if self.synthesizer is not None:
            self.synthesizer.clear_graph_cache()
        self.hubert = None
        self.rmvpe = None
        self.fcpe = None
        self.swiftf0 = None
        self.synthesizer = None
        self.faiss_index = None
        self.index_features = None
        self._loaded = False
        self.clear_cache()

        # Clear CUDA/XPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()

    def clear_cache(self) -> None:
        """Clear feature cache for chunk continuity.

        Call this when starting a new audio stream or after a long pause.
        """
        self._feature_cache = None
        self._f0_cache = None
        self._f0_voiced_cache = None
        self._audio_cache = None
        self._output_cache = None
        self._stream_history = None
        # Streaming feature cache (used by infer_streaming to avoid synth padding)
        self._streaming_feat_cache = None
        # HuBERT audio context buffer (accumulates 16kHz audio for richer context)
        self._streaming_audio_history = None
        # F0 pre-filter tail for cross-chunk filter continuity
        self._streaming_f0_pre_filter_tail: Optional[torch.Tensor] = None
        # Windowed-F0 history (Hz @100fps, aligned with _streaming_audio_history)
        self._streaming_f0_hz_history = None

    def _sync_device_for_profile(self) -> None:
        """Synchronize the device so stage wall-times reflect real GPU work.

        Device-wide sync: with parallel HuBERT+F0 extraction the two stage
        times overlap and each may include the other's queued kernels; set
        use_parallel_extraction=False for exact per-stage attribution.
        """
        try:
            if "xpu" in str(self.device):
                torch.xpu.synchronize()
            elif "cuda" in str(self.device):
                torch.cuda.synchronize()
        except Exception:
            pass

    def _assemble_windowed_f0(
        self,
        f0_win: torch.Tensor,
        target_frames: int,
        window_samples: int,
        slice_samples: int,
        new_samples: int,
        input_length: int,
        t_pad: int,
    ) -> torch.Tensor:
        """Build a full-context F0 track from a windowed extraction.

        The F0 model only saw [t_pad ctx | window | t_pad ctx]; frames older
        than the window are served from ``_streaming_f0_hz_history`` (values
        cached post-shift/moe, pre-filter, so re-filtering the assembled
        track each chunk matches the legacy full-recompute path).  The
        returned track matches the feature timeline layout
        [left pad | real audio | right pad]; pad regions are edge-replicated
        (context-only, trimmed from the synthesized output).
        """
        samples_per_frame = 160  # 100fps @ 16kHz
        pad_frames = t_pad // samples_per_frame

        # Align model output onto the padded-slice frame grid (crop same-rate
        # edge-frame differences, mask-aware log resize for genuine rate
        # mismatches), then keep only the frames that map 1:1 onto the
        # window's real audio (the slice may extend past it into reflect
        # padding to keep its size fixed).
        win_grid = (slice_samples + 2 * t_pad) // samples_per_frame
        f0_win = _align_f0_frames(f0_win, win_grid)
        f0_win = f0_win[
            :, pad_frames : pad_frames + window_samples // samples_per_frame
        ]

        # Append the new-hop frames to the history.  The context portion of
        # the window is re-extracted only as model warmup; cached values win
        # so a frame's F0 never changes after it is first published.
        new_frames = min(new_samples // samples_per_frame, f0_win.shape[1])
        hist_cap = input_length // samples_per_frame
        hist = self._streaming_f0_hz_history
        if hist is not None and hist.shape[0] == f0_win.shape[0]:
            new_tail = f0_win[:, f0_win.shape[1] - new_frames :]
            hist = torch.cat([hist.to(f0_win.dtype), new_tail], dim=1)
        else:
            hist = f0_win
        hist = hist[:, -hist_cap:]
        self._streaming_f0_hz_history = hist

        # Assemble [left pad | (front fill +) history | right fill].
        h = hist.shape[1]
        front_fill = pad_frames + max(0, hist_cap - h)
        right_fill = max(0, target_frames - front_fill - h)
        parts = [hist[:, :1].expand(-1, front_fill), hist]
        if right_fill > 0:
            parts.append(hist[:, -1:].expand(-1, right_fill))
        return torch.cat(parts, dim=1)[:, :target_frames]

    def _set_fixed_harmonics(self, enabled: bool) -> None:
        """Set SineGen harmonic initial phase mode (fixed=zero or random)."""
        model = getattr(self.synthesizer, "model", None)
        if model is None:
            return
        dec = getattr(model, "dec", None)
        if dec is None:
            return
        m_source = getattr(dec, "m_source", None)
        if m_source is None:
            return  # non-F0 model — no SineGen
        sin_gen = getattr(m_source, "l_sin_gen", None)
        if sin_gen is not None:
            sin_gen.fixed_harmonics = enabled

    def _set_uv_ramp(self, ramp_ms: float) -> None:
        """Set SineGen voiced/unvoiced excitation crossfade length (ms)."""
        model = getattr(self.synthesizer, "model", None)
        if model is None:
            return
        dec = getattr(model, "dec", None)
        if dec is None:
            return
        m_source = getattr(dec, "m_source", None)
        if m_source is None:
            return  # non-F0 model — no SineGen
        sin_gen = getattr(m_source, "l_sin_gen", None)
        if sin_gen is not None:
            sin_gen.uv_ramp_ms = float(ramp_ms)

    @torch.no_grad()
    def infer(
        self,
        audio: np.ndarray | torch.Tensor,
        input_sr: int = 16000,
        pitch_shift: int = 0,
        f0_method: str = "rmvpe",
        index_rate: float = 0.0,
        index_k: int = 4,
        voice_gate_mode: str = "off",
        energy_threshold: float = 0.05,
        denoise: bool = False,
        noise_reference: Optional[np.ndarray] = None,
        use_feature_cache: bool = True,
        use_parallel_extraction: bool = True,
        allow_short_input: bool = False,
        pad_mode: str = "chunk",
        synth_min_frames: int | None = MIN_SYNTH_FEATURE_FRAMES,
        history_sec: float = 0.0,
        noise_scale: float = 0.66666,
        moe_boost: float = 0.0,
        f0_lowpass_cutoff_hz: float = 16.0,
        enable_octave_flip_suppress: bool = True,
        enable_f0_slew_limit: bool = True,
        f0_slew_max_step_st: float = 2.8,
        f0_hole_fill_ms: float = 30.0,
    ) -> np.ndarray:
        """
        Convert voice using the RVC pipeline.

        Args:
            audio: Input audio (1D numpy array or tensor)
            input_sr: Input sample rate (default 16kHz)
            pitch_shift: Pitch shift in semitones
            f0_method: F0 extraction method ("rmvpe" or "none")
            index_rate: FAISS index blending ratio (0=off, 0.5=balanced, 1=index only)
            index_k: Number of FAISS neighbors to search (4=fast, 8=quality, default 4)
            voice_gate_mode: Voice gate mode for unvoiced segments:
                - "off": no gating, all audio passes through
                - "strict": F0-based only (may cut plosives like p/t/k)
                - "expand": expand voiced regions to include adjacent plosives
                - "energy": use energy + F0 (plosives with energy pass through)
            energy_threshold: Energy threshold for "energy" mode (0.01-0.2, default 0.05)
            denoise: If True, apply spectral gate noise reduction before processing
            noise_reference: Optional noise sample for denoiser (auto-learns if None)
            use_feature_cache: Enable feature caching for chunk continuity (default True)
            use_parallel_extraction: Enable parallel HuBERT+F0 extraction (~10-15% speedup)
            pad_mode: Audio padding mode:
                - "chunk": per-chunk padding (default, legacy behavior)
                - "batch": batch-style padding (reflection at stream boundaries)
                - "none": no audio-level padding (use for chunked streaming)
            synth_min_frames: Minimum feature frames for synthesizer decoder.
                Set to 0 or None to disable short-input padding (test-only).
            history_sec: Prepend this many seconds of past audio for HuBERT context.
            moe_boost: Moe voice style strength for F0 contour shaping
                (0.0=off, 1.0=strong).

        Returns:
            Converted audio at model sample rate (usually 40kHz)
        """
        if not self._loaded:
            self.load()

        # Convert to tensor if needed
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()

        # ALWAYS resample to 16kHz regardless of input_sr parameter
        # This prevents double-resampling bug when input_sr != 16000
        if len(audio.shape) == 1:
            audio_np = audio.numpy()  # [T]
            if input_sr != 16000:
                audio_np = resample(audio_np, input_sr, 16000)
            audio = torch.from_numpy(audio_np).float()
        else:  # Multi-channel
            raise ValueError(f"Expected 1D audio, got {len(audio.shape)}D")

        # Apply noise reduction if enabled
        if denoise:
            audio_np = audio.numpy()
            # Resample noise reference if provided (ONLY resample ONCE)
            if noise_reference is not None and len(noise_reference) > 0:
                # Resample noise reference if sample rate differs
                if input_sr != 16000:
                    noise_reference = resample(noise_reference, input_sr, 16000)
                # Use DeepFilterNet if available, otherwise spectral gate
            audio_np = denoise_audio(
                audio_np,
                sample_rate=16000,
                method="auto",  # DeepFilterNet if available, else spectral gate
                noise_reference=noise_reference,
                device=self.device,
            )
            audio = torch.from_numpy(audio_np).float()
            logger.debug("Applied noise reduction")

        # Keep a copy of current chunk for history/output length tracking
        chunk_audio_np = audio_np
        chunk_length = len(chunk_audio_np)

        # Optional: prepend past audio for HuBERT context
        history_samples = int(max(0.0, history_sec) * 16000)
        history_len_used = 0
        if history_samples > 0:
            if self._stream_history is None:
                self._stream_history = np.zeros(0, dtype=chunk_audio_np.dtype)
            if len(self._stream_history) > 0:
                history_len_used = min(len(self._stream_history), history_samples)
                history_slice = self._stream_history[-history_len_used:]
                audio_np = np.concatenate([history_slice, chunk_audio_np])
            # Update history buffer with current chunk
            self._stream_history = np.concatenate([self._stream_history, chunk_audio_np])[-history_samples:]

        # Apply high-pass filter to remove DC offset and low-frequency noise
        # Original RVC uses 48Hz cutoff Butterworth filter
        audio_np = highpass_filter(audio_np, sr=16000, cutoff=48)

        # Add reflection padding for edge handling
        # Base padding: 50ms (800 samples @ 16kHz)
        # For short inputs, increase padding to ensure we get MIN_SYNTH_FEATURE_FRAMES
        # without needing feature-level padding (which causes length mismatch issues)
        input_length = len(audio_np)
        hubert_hop = 320
        f0_hop = 160  # F0 models use 160 sample hop (100fps at 16kHz)

        # Phase 5: Audio-level overlap for F0/HuBERT extraction continuity
        # Re-enabled: prepend cached audio and discard overlap frames from features/F0.
        audio_overlap_samples = 0
        hubert_overlap_frames = 0
        f0_overlap_frames = 0
        raw_audio_np = chunk_audio_np
        if history_samples <= 0 and use_feature_cache and self._audio_cache is not None and len(self._audio_cache) > 0:
            audio_overlap_samples = len(self._audio_cache)
            audio_np = np.concatenate([self._audio_cache, audio_np])
            hubert_overlap_frames = audio_overlap_samples // hubert_hop
            f0_overlap_frames = audio_overlap_samples // f0_hop
            logger.debug(
                f"Audio overlap: prepended {audio_overlap_samples} samples "
                f"({audio_overlap_samples/16:.1f}ms), "
                f"hubert_overlap_frames={hubert_overlap_frames}, "
                f"f0_overlap_frames={f0_overlap_frames}"
            )

        # Update audio cache with tail of current chunk (for future use)
        # Currently disabled but keeping cache updated for potential re-enabling
        if use_feature_cache:
            cache_len = min(self._audio_cache_len, chunk_length)
            self._audio_cache = raw_audio_np[-cache_len:].copy()

        pad_mode = pad_mode.lower()
        valid_pad_modes = {"chunk", "batch", "none"}
        if pad_mode not in valid_pad_modes:
            raise ValueError(f"Invalid pad_mode: {pad_mode} (valid: {sorted(valid_pad_modes)})")

        # Base padding: 50ms (800 samples) for batch processing
        # Chunk processing uses reduced padding (1 HuBERT hop) below
        base_pad = int(16000 * 0.05)  # 800 samples (50ms @ 16kHz)

        # For chunk processing, use minimal padding to avoid excessive padding artifacts
        # Optimal: 1 HuBERT hop (320 samples = 20ms) for batch/chunk consistency
        # This reduces padding accumulation at chunk boundaries while maintaining edge quality
        if pad_mode == "none":
            t_pad = 0
        elif pad_mode == "chunk" and allow_short_input:
            t_pad = hubert_hop  # 320 samples (20ms @ 16kHz, 1 HuBERT hop) - optimal
        else:
            # Calculate minimum input samples needed for MIN_SYNTH_FEATURE_FRAMES features
            # MIN_SYNTH_FEATURE_FRAMES is at 100fps, HuBERT produces 50fps, so /2
            # HuBERT produces approximately (samples / hop) - 1 frames due to its internal handling,
            # so we add 2 extra hops to ensure we get enough frames
            min_hubert_frames = MIN_SYNTH_FEATURE_FRAMES // 2  # 32 frames
            min_input_samples = (min_hubert_frames + 2) * hubert_hop  # 10880 samples (with buffer)

            # Check if we need extra padding to meet minimum
            total_with_base = base_pad + input_length + base_pad
            if total_with_base < min_input_samples:
                # Need more padding - distribute evenly
                extra_needed = min_input_samples - total_with_base
                extra_per_side = (extra_needed + 1) // 2
                t_pad = base_pad + extra_per_side
                logger.info(
                    f"Short input: increased padding from {base_pad} to {t_pad} samples per side"
                )
            else:
                t_pad = base_pad

        t_pad_tgt = int(t_pad * self.sample_rate / 16000)  # Output padding samples (proportional)

        if pad_mode == "none":
            extra_pad = 0
        else:
            # Pad to multiple of HuBERT hop size (320) to avoid frame truncation
            padded_for_hubert = t_pad + input_length + t_pad
            remainder = padded_for_hubert % hubert_hop
            if remainder != 0:
                extra_pad = hubert_hop - remainder
            else:
                extra_pad = 0

        if t_pad > 0 or extra_pad > 0:
            audio_np = np.pad(audio_np, (t_pad, t_pad + extra_pad), mode="reflect")
        logger.info(
            f"Padding: chunk={chunk_length}, input={input_length}, "
            f"t_pad={t_pad}, extra_pad={extra_pad}, "
            f"final={len(audio_np)} (mode={pad_mode})"
        )

        audio_cpu = torch.from_numpy(audio_np).float()

        # Ensure 2D for batch processing
        if audio_cpu.dim() == 1:
            audio_cpu = audio_cpu.unsqueeze(0)

        audio = audio_cpu.to(self.device)
        f0_audio = (
            audio_cpu
            if f0_method == "swiftf0" and self.swiftf0 is not None
            else audio
        )
        residual_f0_shift = float(pitch_shift)
        moe_strength = max(0.0, min(MAX_MOE_BOOST, float(moe_boost)))

        # Debug: input audio stats
        logger.info(
            f"Input audio: shape={audio.shape} (chunk={chunk_length}, input={input_length}), min={audio.min():.4f}, max={audio.max():.4f}"
        )

        # Determine output dimension and layer based on model version
        # v1 models: layer 9, 256-dim features
        # v2 models: layer 12, 768-dim features (original RVC specification)
        if self.synthesizer.version == 1:
            output_dim = 256
            output_layer = 9
        else:
            output_dim = 768
            output_layer = 12

        # Parallel extraction: HuBERT features + F0 (if enabled)
        # Use ThreadPoolExecutor for ~10% speedup (more stable than GPU streams)
        features = None
        f0_raw = None
        use_f0 = self.has_f0 and f0_method != "none"

        graph_capture_pending = self.hubert.graph_capture_pending(
            audio, output_layer, output_dim
        )
        if graph_capture_pending:
            logger.info("HuBERT Accelerator Graph capture will run before parallel F0")

        if use_parallel_extraction and use_f0 and not graph_capture_pending:

            def extract_hubert():
                with torch.autocast(device_type=self.device, dtype=self.dtype):
                    return self.hubert.extract(
                        audio, output_layer=output_layer, output_dim=output_dim
                    )

            def extract_f0():
                if f0_method == "fcpe" and self.fcpe is not None:
                    with torch.autocast(device_type=self.device, dtype=self.dtype):
                        return self.fcpe.infer(
                            audio, threshold=FCPE_VOICING_THRESHOLD
                        )
                elif f0_method == "swiftf0" and self.swiftf0 is not None:
                    return self.swiftf0.infer(
                        f0_audio, threshold=SWIFTF0_VOICING_THRESHOLD
                    )
                elif self.rmvpe is not None:
                    with torch.autocast(device_type=self.device, dtype=self.dtype):
                        return self.rmvpe.infer(
                            audio, threshold=RMVPE_VOICING_THRESHOLD
                        )
                return None

            # Run HuBERT and F0 extraction in parallel threads
            with ThreadPoolExecutor(max_workers=2) as executor:
                hubert_future = executor.submit(extract_hubert)
                f0_future = executor.submit(extract_f0)

                features = hubert_future.result()
                f0_raw = f0_future.result()

            logger.debug("Parallel extraction complete (HuBERT + F0, ThreadPool)")

        # Fallback: sequential extraction
        if features is None:
            with torch.autocast(device_type=self.device, dtype=self.dtype):
                features = self.hubert.extract(
                    audio, output_layer=output_layer, output_dim=output_dim
                )

        logger.info(
            f"HuBERT features: shape={features.shape}, min={features.min():.4f}, max={features.max():.4f}"
        )

        # Phase 5: Trim overlap frames from HuBERT features
        # These frames came from the prepended audio cache and should be discarded
        if hubert_overlap_frames > 0 and features.shape[1] > hubert_overlap_frames:
            features = features[:, hubert_overlap_frames:, :]
            logger.debug(
                f"HuBERT overlap trim: removed {hubert_overlap_frames} frames, new shape={features.shape}"
            )

        # Apply FAISS index retrieval if enabled (before interpolation, like original RVC)
        if index_rate > 0 and self.faiss_index is not None:
            logger.info(
                f"Applying index retrieval: index_rate={index_rate}, k={index_k}, features_before={features.shape} mean={features.mean():.4f} std={features.std():.4f}"
            )
            features = self._search_index(features, index_rate, k=index_k)
            logger.info(
                f"Index retrieval applied: index_rate={index_rate}, features_after={features.shape} mean={features.mean():.4f} std={features.std():.4f}"
            )

        # Feature cache blending for chunk continuity (50fps HuBERT features)
        # Phase 2 improvement: Adaptive blending based on cosine similarity
        if use_feature_cache and self._feature_cache is not None:
            max_cache_len = max(1, int(self._feature_cache_frames))
            cache_avail = min(max_cache_len, self._feature_cache.shape[1], features.shape[1])

            if cache_avail > 0:
                prev_tail = self._feature_cache[:, -cache_avail:, :]
                curr_head = features[:, :cache_avail, :].clone()

                # Calculate cosine similarity at boundary
                prev_last = prev_tail[:, -1:, :]  # [B, 1, C]
                curr_first = curr_head[:, :1, :]  # [B, 1, C]
                cos_sim = torch.nn.functional.cosine_similarity(
                    prev_last.squeeze(1), curr_first.squeeze(1), dim=-1
                )  # [B]

                # Adaptive blend length: lower similarity = longer blend
                # similarity 0.9+ -> 5 frames, similarity 0.5 -> 15 frames
                adaptive_blend = int((1.0 - cos_sim.mean().item()) * max_cache_len) + 5
                blend_len = min(adaptive_blend, cache_avail)

                if blend_len > 0:
                    # Sigmoid blending for smoother transitions
                    alpha_np = sigmoid_blend_weights(blend_len, steepness=4.0)
                    alpha = torch.from_numpy(alpha_np).to(
                        device=features.device, dtype=features.dtype
                    ).view(1, -1, 1)

                    # Blend features
                    blended = prev_tail[:, -blend_len:, :] * alpha + curr_head[:, :blend_len, :] * (1.0 - alpha)

                    # Preserve original norm (feature magnitude)
                    orig_norms = torch.norm(curr_head[:, :blend_len, :], dim=-1, keepdim=True)
                    blended_norms = torch.norm(blended, dim=-1, keepdim=True) + 1e-8
                    blended = blended * (orig_norms / blended_norms)

                    features[:, :blend_len, :] = blended

                    logger.debug(
                        f"HuBERT adaptive blend: cos_sim={cos_sim.mean().item():.3f}, blend_len={blend_len}"
                    )

        # Update HuBERT feature cache (store tail for next chunk)
        if use_feature_cache:
            cache_len = max(1, int(self._feature_cache_frames))
            if features.shape[1] > 0:
                self._feature_cache = features[:, -cache_len:, :].detach()

        # Interpolate features to match synthesizer expectation
        # Linear interpolation is perceptually smoother and helps reduce metallic buzz.
        # HuBERT hop=320 @ 16kHz (50fps) -> Synthesizer needs 100fps
        original_frames = features.shape[1]
        features = torch.nn.functional.interpolate(
            features.permute(0, 2, 1),  # [B, T, C] -> [B, C, T]
            scale_factor=2,  # Fixed 2x upscale (matches original RVC)
            mode="linear",
            align_corners=False,
        ).permute(0, 2, 1)  # [B, C, T] -> [B, T, C]
        logger.info(
            f"Interpolated features: {original_frames} -> {features.shape[1]} frames (2x linear)"
        )

        # Feature length
        feature_lengths = torch.tensor([features.shape[1]], dtype=torch.long, device=self.device)

        # Extract F0 if using F0 model
        pitch = None
        pitchf = None
        if use_f0:
            # Use F0 from parallel extraction if available
            f0 = f0_raw if f0_raw is not None else None

            # Otherwise extract sequentially (fallback or CPU mode)
            if f0 is None:
                # Use FCPE if requested and available
                if f0_method == "fcpe" and self.fcpe is not None:
                    with torch.autocast(device_type=self.device, dtype=self.dtype):
                        f0 = self.fcpe.infer(
                            audio, threshold=FCPE_VOICING_THRESHOLD
                        )
                    logger.debug("F0 extracted with FCPE (sequential)")

                # Use SwiftF0 if requested and available
                elif f0_method == "swiftf0" and self.swiftf0 is not None:
                    f0 = self.swiftf0.infer(
                        f0_audio, threshold=SWIFTF0_VOICING_THRESHOLD
                    )
                    logger.debug("F0 extracted with SwiftF0 (sequential)")

                # Use RMVPE if requested and available (or fallback if others failed)
                elif self.rmvpe is not None and (f0_method == "rmvpe" or f0 is None):
                    with torch.autocast(device_type=self.device, dtype=self.dtype):
                        f0 = self.rmvpe.infer(
                            audio, threshold=RMVPE_VOICING_THRESHOLD
                        )
                    logger.debug("F0 extracted with RMVPE (sequential)")
            else:
                logger.debug(f"F0 from parallel extraction ({f0_method})")

            if f0 is not None and f0.device != torch.device(self.device):
                f0 = f0.to(self.device)

            # Phase 5: Trim overlap frames from F0
            # These frames came from the prepended audio cache and should be discarded
            if f0 is not None and f0_overlap_frames > 0 and f0.shape[1] > f0_overlap_frames:
                f0 = f0[:, f0_overlap_frames:]
                logger.debug(
                    f"F0 overlap trim: removed {f0_overlap_frames} frames, new shape={f0.shape}"
                )

            if f0 is not None and f0.numel() > 0:  # Check for non-empty F0
                # FCPE smoothing for stability (reduce frame-to-frame jitter)
                if f0_method == "fcpe":
                    # Light smoothing to reduce jitter without adding artifacts
                    f0 = _smooth_fcpe_f0(f0)

                # Apply pitch shift (only to voiced regions where f0 > 0)
                if abs(residual_f0_shift) > 0.01:
                    f0 = torch.where(
                        f0 > 0, f0 * (2 ** (residual_f0_shift / 12)), f0
                    )
                if moe_strength > 0.0:
                    f0 = apply_moe_f0_style(f0, moe_strength)

                # Align F0 length with features.
                # Features are ~2 frames short at the RIGHT edge (HuBERT
                # frame deficit — the same convention as the right trim), so
                # a uniform resize would squeeze the whole F0 timeline and
                # misplace pitch by up to ~20ms at the tail.  Rate-normalize
                # to the 100fps grid, then crop the deficit on the right.
                if f0.shape[1] != features.shape[1]:
                    f0 = _align_f0_frames(f0, features.shape[1] + 2)
                    f0 = f0[:, : features.shape[1]]

                # F0 cache blending for chunk continuity (100fps F0)
                # Phase 1 improvement: Extended cache, sigmoid blending, jump detection
                if use_feature_cache and self._f0_cache is not None:
                    cache_len = max(1, int(self._f0_cache_frames))
                    blend_len = min(cache_len, self._f0_cache.shape[1], f0.shape[1])
                    if blend_len > 0:
                        prev_tail = self._f0_cache[:, -blend_len:]
                        prev_voiced = (
                            self._f0_voiced_cache[:, -blend_len:]
                            if self._f0_voiced_cache is not None
                            else prev_tail > 0
                        )
                        cur_head = f0[:, :blend_len].clone()
                        cur_voiced = cur_head > 0
                        blend_mask = prev_voiced & cur_voiced

                        # Sigmoid blending for smoother transitions
                        alpha_np = sigmoid_blend_weights(blend_len, steepness=4.0)
                        alpha = torch.from_numpy(alpha_np).to(
                            device=f0.device, dtype=f0.dtype
                        ).view(1, -1)

                        # Detect large jumps and apply linear interpolation
                        # Check the boundary between cache and current
                        prev_last_f0 = prev_tail[:, -1]
                        cur_first_f0 = cur_head[:, 0]
                        both_voiced = (prev_last_f0 > 0) & (cur_first_f0 > 0)

                        if both_voiced.any():
                            jump = torch.abs(prev_last_f0 - cur_first_f0)
                            # If jump > 50Hz, apply linear interpolation at boundary
                            if jump.item() > 50:
                                # Linear interpolation for first few frames
                                interp_len = min(10, blend_len)
                                interp_weights = torch.linspace(
                                    0.0, 1.0, interp_len, device=f0.device, dtype=f0.dtype
                                ).view(1, -1)
                                # Interpolate from prev_last_f0 to values further in cur_head
                                target_f0 = cur_head[:, interp_len - 1 : interp_len]
                                if target_f0.numel() > 0:
                                    interp_f0 = prev_last_f0.view(-1, 1) * (1 - interp_weights) + \
                                                target_f0 * interp_weights
                                    # Apply interpolation only where both are voiced
                                    for i in range(interp_len):
                                        if cur_head[:, i].item() > 0:
                                            cur_head[:, i] = interp_f0[:, i]
                                logger.debug(
                                    f"F0 jump {jump.item():.1f}Hz detected, applied linear interpolation"
                                )

                        # Apply sigmoid blending
                        blended = prev_tail * alpha + cur_head * (1.0 - alpha)
                        f0[:, :blend_len] = torch.where(blend_mask, blended, f0[:, :blend_len])

                # Shared F0 post-processing chain
                # (median spike removal -> lowpass -> octave -> slew)
                f0 = apply_f0_filter_chain(
                    f0,
                    f0_lowpass_cutoff_hz=f0_lowpass_cutoff_hz,
                    enable_octave_flip_suppress=enable_octave_flip_suppress,
                    enable_f0_slew_limit=enable_f0_slew_limit,
                    f0_slew_max_step_st=f0_slew_max_step_st,
                    f0_hole_fill_ms=f0_hole_fill_ms,
                )

                # Batch boundary stabilization to suppress start/end F0 flutter.
                f0 = stabilize_f0_boundaries(f0, edge_frames=10)

                # pitchf: continuous F0 values for NSF decoder
                pitchf = f0.to(self.dtype)

                # pitch: quantized F0 for the 256-bin pitch embedding, plus the
                # voiced mask reused for post-synthesis gating.
                pitch, voiced_mask_for_gate = quantize_f0_to_pitch(f0)
                logger.info(
                    f"F0: shape={f0.shape}, min={f0.min():.1f}, max={f0.max():.1f}, "
                    f"voiced={int(voiced_mask_for_gate.sum().item())}/{f0.numel()}, "
                    f"pitch_range=[{pitch.min().item()}, {pitch.max().item()}]"
                )

                # Update F0 cache (store tail for next chunk)
                # Extended cache length for better boundary blending
                if use_feature_cache:
                    cache_len = max(1, int(self._f0_cache_frames))
                    self._f0_cache = f0[:, -cache_len:].detach()
                    self._f0_voiced_cache = (f0 > 0)[:, -cache_len:].detach()
            elif f0 is not None and f0.numel() == 0:
                # F0 extraction returned empty array (input too short)
                logger.warning(
                    f"F0 extraction returned empty array (input too short: {len(audio)/16000:.3f}s). "
                    f"Minimum recommended: FCPE=0.10s, RMVPE=0.32s. Falling back to unvoiced."
                )
                f0 = None  # Fall through to unvoiced handling

            if f0 is None:
                # No F0 available - use pitch=1 (unvoiced marker per RVC convention)
                # No F0 model available - use pitch=1 (unvoiced marker per RVC convention)
                pitch = torch.ones(
                    features.shape[0], features.shape[1], dtype=torch.long, device=self.device
                )
                pitchf = torch.zeros(
                    features.shape[0], features.shape[1], dtype=self.dtype, device=self.device
                )
                voiced_mask_for_gate = None
                logger.info("F0: using unvoiced pitch=1 (no F0 model)")
        elif self.has_f0:
            # F0-capable model with F0 explicitly disabled (f0_method="none")
            pitch = torch.ones(
                features.shape[0], features.shape[1], dtype=torch.long, device=self.device
            )
            pitchf = torch.zeros(
                features.shape[0], features.shape[1], dtype=self.dtype, device=self.device
            )
            voiced_mask_for_gate = None
            logger.info(f"F0: skipped (f0_method={f0_method}), using unvoiced pitch=1")
        else:
            voiced_mask_for_gate = None

        # Pad features if too short for synthesizer decoder
        # The decoder's upsampling convolutions require minimum input length
        synth_pad_frames = 0
        min_synth_frames = (
            MIN_SYNTH_FEATURE_FRAMES if synth_min_frames is None else synth_min_frames
        )
        if min_synth_frames > 0 and features.shape[1] < min_synth_frames:
            synth_pad_frames = min_synth_frames - features.shape[1]
            pad_left = synth_pad_frames // 2
            pad_right = synth_pad_frames - pad_left

            # Choose padding mode based on input size
            # reflect mode requires padding < input size, use replicate for very short inputs
            current_size = features.shape[1]
            if max(pad_left, pad_right) >= current_size:
                pad_mode = "replicate"  # Edge replication for very short inputs
                logger.warning(
                    f"Input too short ({current_size} frames) for reflection padding ({pad_left}+{pad_right}). "
                    f"Using replicate mode. Consider increasing chunk_sec to >= 0.15s"
                )
            else:
                pad_mode = "reflect"  # Preferred for normal inputs

            features = torch.nn.functional.pad(
                features.permute(0, 2, 1),  # [B, T, C] -> [B, C, T]
                (pad_left, pad_right),
                mode=pad_mode,
            ).permute(0, 2, 1)  # [B, C, T] -> [B, T, C]
            # Update feature lengths
            feature_lengths = torch.tensor(
                [features.shape[1]], dtype=torch.long, device=self.device
            )
            # Pad pitch/pitchf if present
            if pitch is not None:
                pitch = torch.nn.functional.pad(pitch, (pad_left, pad_right), mode=pad_mode)
            if pitchf is not None:
                pitchf = torch.nn.functional.pad(pitchf, (pad_left, pad_right), mode=pad_mode)
            logger.info(
                f"Padded short input: {features.shape[1] - synth_pad_frames} -> {features.shape[1]} frames (min={min_synth_frames})"
            )

        # Run synthesizer
        logger.info(
            f"Synthesizer input: features={features.shape}, pitch={pitch.shape if pitch is not None else None}"
        )
        with torch.autocast(device_type=self.device, dtype=self.dtype):
            output = self.synthesizer.infer(
                features,
                feature_lengths,
                pitch=pitch,
                pitchf=pitchf,
                noise_scale=noise_scale,
            )

        logger.info(
            f"Synthesizer output: shape={output.shape}, min={output.min():.4f}, max={output.max():.4f}"
        )

        # Trim synthesizer padding if we added it for short inputs
        if synth_pad_frames > 0:
            # Calculate output samples to trim based on feature-to-audio ratio
            # Each feature frame = samples_per_frame samples at synthesizer sample rate
            samples_per_frame = self.sample_rate // 100  # 100fps features -> samples/frame
            trim_left = (synth_pad_frames // 2) * samples_per_frame
            trim_right = (synth_pad_frames - synth_pad_frames // 2) * samples_per_frame

            # Debug: check synth output before trimming
            synth_total = output.shape[-1]
            synth_tail_rms = (
                float(
                    torch.sqrt(
                        torch.mean(output[..., -trim_right - samples_per_frame : -trim_right] ** 2)
                    )
                )
                if trim_right > 0
                else 0
            )

            if output.shape[-1] > trim_left + trim_right:
                output = (
                    output[..., trim_left:-trim_right]
                    if trim_right > 0
                    else output[..., trim_left:]
                )
                logger.info(
                    f"Trimmed synth padding: {trim_left} + {trim_right} samples (synth_total={synth_total}, synth_tail_rms={synth_tail_rms:.4f})"
                )

        # Apply voice gating based on mode (shared with infer_streaming)
        if voice_gate_mode != "off" and voiced_mask_for_gate is not None:
            output, gate_mask = apply_voice_gate(
                output,
                voiced_mask_for_gate,
                voice_gate_mode=voice_gate_mode,
                energy_threshold=energy_threshold,
                sample_rate=self.sample_rate,
            )
            voiced_ratio = gate_mask.mean().item()
            logger.info(f"Voice gate ({voice_gate_mode}): {voiced_ratio * 100:.1f}% passed")

        # Convert to numpy
        output = output.cpu().float().numpy()

        if output.ndim == 2:
            output = output[0]

        # Trim padding from output (match the input padding ratio)
        # t_pad_tgt was calculated earlier based on actual t_pad used
        # Also account for extra_pad added for HuBERT alignment
        extra_pad_tgt = int(extra_pad * self.sample_rate / 16000)
        # HuBERT produces (input / 320) - 1 frames, so synthesizer output
        # is 1 HuBERT frame shorter than naive expectation.  Reduce trim_end
        # to compensate (same fix as infer_streaming).
        hubert_deficit = 2 * (self.sample_rate // 100)
        trim_start = t_pad_tgt
        trim_end = max(0, t_pad_tgt + extra_pad_tgt - hubert_deficit)

        # Debug: check output before trimming
        pre_trim_tail_rms = (
            np.sqrt(np.mean(output[-trim_end - 480 : -trim_end] ** 2))
            if trim_end > 0 and len(output) > trim_end + 480
            else 0
        )
        post_trim_tail_start = -trim_end if trim_end > 0 else len(output)

        if len(output) > trim_start + trim_end:
            output = output[trim_start:-trim_end] if trim_end > 0 else output[trim_start:]
            logger.info(
                f"Trimmed {trim_start} from start, {trim_end} from end (pre-trim tail rms={pre_trim_tail_rms:.4f})"
            )

        if history_len_used > 0:
            history_output_samples = int(history_len_used * self.sample_rate / 16000)
            if len(output) > history_output_samples:
                output = output[history_output_samples:]
                logger.info(
                    f"Trimmed history from start: {history_output_samples} samples (history_sec={history_sec:.3f})"
                )

        # Note: HuBERT frame quantization causes output to be slightly shorter than ideally expected.
        # We intentionally do NOT pad/extend to match expected length, as artificial waveform
        # repetition creates artifacts at chunk boundaries. Instead:
        # - Accept the actual output length (crossfade handles variable-length chunks)
        # - Only trim if output is too long (rare edge case)
        expected_output_samples = int(chunk_length * self.sample_rate / 16000)
        length_diff = len(output) - expected_output_samples
        logger.info(
            f"Length check: got {len(output)}, expected {expected_output_samples}, diff={length_diff}"
        )

        if length_diff > 0:
            # Output too long - trim from end
            output = output[:expected_output_samples]
            logger.debug(f"Trimmed {length_diff} extra samples from end")
        elif length_diff < 0 and abs(length_diff) > 100:
            if allow_short_input and pad_mode == "chunk":
                # Output too short (chunk processing) - pad zeros to expected length
                # This prevents cumulative drift across chunks while avoiding resampling artifacts.
                output = np.pad(output, (0, -length_diff))
                logger.debug(
                    f"Padded output from {expected_output_samples + length_diff} to {expected_output_samples} samples"
                )
            else:
                # Output too short - resample to stretch to expected length
                # Used for batch, and for streaming with pad_mode="none" to avoid zero-padding artifacts.
                output = resample(output, len(output), expected_output_samples)
                logger.debug(
                    f"Resampled output from {expected_output_samples + length_diff} to {expected_output_samples} samples"
                )

        output = apply_output_edge_fade(output, sample_rate=self.sample_rate, fade_ms=8.0)

        logger.info(
            f"Final output: shape={output.shape}, min={output.min():.4f}, max={output.max():.4f}"
        )
        return output

    @torch.no_grad()
    def infer_streaming(
        self,
        audio_16k: np.ndarray,
        overlap_samples: int,
        params: Optional[StreamingParams] = None,
        **overrides,
    ) -> np.ndarray:
        """Streaming inference with audio-level overlap.

        Processes [overlap | new_hop] audio through the full pipeline
        (HuBERT -> synthesizer), then trims the synthesizer OUTPUT to
        keep only the new_hop portion (plus optional SOLA extra).

        Args:
            audio_16k: Input audio at 16kHz, shape [overlap + new_hop].
                       Length MUST be a multiple of 320 (HuBERT hop).
                       overlap_samples MUST also be a multiple of 320.
            overlap_samples: Number of overlap samples from previous chunk.
            params: Tunable parameters bundle (:class:`StreamingParams`).
                When ``None``, one is built from ``**overrides`` so legacy
                callers passing individual keyword arguments keep working.
            **overrides: Individual :class:`StreamingParams` fields. Merged
                onto ``params`` when both are given.

        StreamingParams fields of note:
            sola_extra_samples: Extra samples to keep from the overlap region
                (at model sample rate) to compensate for SOLA crossfade
                deficit — consecutive outputs overlap by this amount so SOLA
                can crossfade without losing samples.
            moe_boost: Moe voice style strength for F0 contour shaping.

        Returns:
            Converted audio at model sample rate (usually 40kHz).
            Output length = new_hop + sola_extra_samples.
        """
        if not self._loaded:
            self.load()

        # Accept either a StreamingParams bundle or legacy per-call keyword
        # arguments (collected into **overrides), then unpack into locals so
        # the body below reads exactly as it did before the refactor.
        if params is None:
            params = StreamingParams(**overrides)
        elif overrides:
            params = replace(params, **overrides)

        pitch_shift = params.pitch_shift
        f0_method = params.f0_method
        index_rate = params.index_rate
        index_k = params.index_k
        voice_gate_mode = params.voice_gate_mode
        energy_threshold = params.energy_threshold
        use_parallel_extraction = params.use_parallel_extraction
        noise_scale = params.noise_scale
        sola_extra_samples = params.sola_extra_samples
        moe_boost = params.moe_boost
        f0_lowpass_cutoff_hz = params.f0_lowpass_cutoff_hz
        enable_octave_flip_suppress = params.enable_octave_flip_suppress
        enable_f0_slew_limit = params.enable_f0_slew_limit
        f0_slew_max_step_st = params.f0_slew_max_step_st
        hubert_context_sec = params.hubert_context_sec
        fixed_harmonics = params.fixed_harmonics
        f0_context_sec = params.f0_context_sec
        f0_hole_fill_ms = params.f0_hole_fill_ms
        uv_ramp_ms = params.uv_ramp_ms

        # Per-stage times (ms); published on self for the realtime
        # controller's [PERF] logging.  With device timing events (preferred)
        # GPU stages are measured on the GPU timeline and resolved after the
        # chunk's natural end-of-pipeline sync — nothing lands on the hot
        # path.  Without event support, fall back to wall-clock + device sync
        # on sampled chunks only.
        self._stage_profile_counter += 1
        use_timing_events = self._profile_events_ok is not False
        if use_timing_events or (
            self._stage_profile_counter <= 5
            or (self._stage_profile_counter - 1) % STAGE_PROFILE_INTERVAL == 0
        ):
            prof: Optional[_StageProfiler] = _StageProfiler(self, use_timing_events)
            self.stage_times = prof.times
        else:
            prof = None

        hubert_hop = 320

        # Validate alignment
        total_samples = len(audio_16k)
        assert total_samples % hubert_hop == 0, (
            f"audio_16k length {total_samples} is not a multiple of {hubert_hop}"
        )
        assert overlap_samples % hubert_hop == 0, (
            f"overlap_samples {overlap_samples} is not a multiple of {hubert_hop}"
        )

        new_samples = total_samples - overlap_samples
        assert new_samples > 0, "No new audio samples after overlap"

        # Expected output length (at model sample rate)
        # Include sola_extra so consecutive outputs overlap by that amount,
        # allowing SOLA crossfade to consume those samples without deficit.
        expected_output = int(new_samples * self.sample_rate / 16000) + sola_extra_samples

        # --- HuBERT audio context buffer ---
        # Accumulate real 16kHz audio so HuBERT processes a larger context
        # window each chunk.  Up to MAX_HUBERT_CONTEXT_16K (2.0s) of audio
        # is kept, producing coherent features from a single forward pass.
        # F0 is extracted on the current chunk only (pitch is local).
        t_pad = 800

        new_hop_16k = audio_16k[overlap_samples:]

        # Minimum 16kHz audio (before padding) to produce MIN_SYNTH_FEATURE_FRAMES
        # from a single HuBERT forward pass.
        # HuBERT: (input + 2*t_pad) / 320 - 1 frames @ 50fps, interpolated 2x → 100fps
        # 64 features → 32 HuBERT frames → 33*320 - 2*t_pad = 8960 samples
        min_audio_for_full_features = (
            (MIN_SYNTH_FEATURE_FRAMES // 2 + 1) * hubert_hop - 2 * t_pad
        )

        # --- Audio history for HuBERT (capped at hubert_context_sec) ---
        max_hubert_context_16k = max(
            min_audio_for_full_features,
            int(hubert_context_sec * 16000),
        )
        if self._streaming_audio_history is None:
            self._streaming_audio_history = audio_16k.copy()
        else:
            self._streaming_audio_history = np.concatenate([
                self._streaming_audio_history, new_hop_16k
            ])
        if len(self._streaming_audio_history) > max_hubert_context_16k:
            self._streaming_audio_history = (
                self._streaming_audio_history[-max_hubert_context_16k:]
            )
        hubert_history_full = (
            len(self._streaming_audio_history) >= max_hubert_context_16k
        )

        # Extend audio_16k with HuBERT history (capped at max_hubert_context_16k).
        pre_context_samples = 0
        if len(self._streaming_audio_history) > total_samples:
            pre_context_samples = len(self._streaming_audio_history) - total_samples
            # Treat pre-context as additional overlap (trimmed from output)
            audio_16k = self._streaming_audio_history.copy()
            total_samples = len(audio_16k)
            overlap_samples = overlap_samples + pre_context_samples

        # Apply high-pass filter
        audio_filtered = highpass_filter(audio_16k, sr=16000, cutoff=48)

        # Reflection padding for edge handling (fixed 50ms like batch infer)
        input_length = len(audio_filtered)

        # Ensure padded audio is multiple of hubert_hop
        padded_for_hubert = t_pad + input_length + t_pad
        remainder = padded_for_hubert % hubert_hop
        extra_pad = (hubert_hop - remainder) if remainder != 0 else 0

        audio_padded = np.pad(
            audio_filtered, (t_pad, t_pad + extra_pad), mode="reflect"
        )

        # Pad HuBERT input to a fixed size
        # As the audio history grows, the input size changes every chunk,
        # triggering expensive kernel compilations on Intel XPU.  By padding
        # to the maximum expected size, we get a single compilation on the
        # first chunk and stable performance thereafter.  The extra synthesis
        # output from the end padding is exactly consumed by the increased
        # trim_right, so net output length is unchanged.
        fixed_hubert_input = (
            (max_hubert_context_16k + 2 * t_pad + hubert_hop - 1)
            // hubert_hop * hubert_hop
        )
        if len(audio_padded) < fixed_hubert_input:
            end_pad = fixed_hubert_input - len(audio_padded)
            audio_padded = np.pad(audio_padded, (0, end_pad), mode="reflect")
            extra_pad += end_pad

        # Output-level trim amounts (at model sample rate)
        t_pad_tgt = int(t_pad * self.sample_rate / 16000)
        extra_pad_tgt = int(extra_pad * self.sample_rate / 16000)
        overlap_tgt = int(overlap_samples * self.sample_rate / 16000)
        # HuBERT produces (input / 320) - 1 frames, so the synthesizer output
        # is 1 HuBERT frame shorter than the naive expectation.  In output
        # space this is 2 feature-frames * samples_per_frame.
        hubert_deficit = 2 * (self.sample_rate // 100)
        # Left trim: padding + overlap (minus sola_extra to keep extra overlap
        # in the output for SOLA crossfade compensation).
        # Right trim: padding + extra - deficit.
        trim_left = max(0, t_pad_tgt + overlap_tgt - sola_extra_samples)
        trim_right = max(0, t_pad_tgt + extra_pad_tgt - hubert_deficit)

        # Convert to tensors
        audio_cpu_t = torch.from_numpy(audio_padded).float().unsqueeze(0)
        audio_t = audio_cpu_t.to(self.device)
        residual_f0_shift = float(pitch_shift)
        moe_strength = max(0.0, min(MAX_MOE_BOOST, float(moe_boost)))
        # Determine HuBERT output params
        if self.synthesizer.version == 1:
            output_dim = 256
            output_layer = 9
        else:
            output_dim = 768
            output_layer = 12

        # --- F0 extraction window (pitch is local) ---
        # Instead of running the F0 model over the full HuBERT context every
        # chunk, slice [f0_context | new_hop] plus t_pad of edge context on
        # both sides from the already-padded tensor (the left edge context is
        # real audio, the right edge is the existing reflect padding).  Frames
        # older than the window come from _streaming_f0_hz_history.
        f0_window_samples = 0
        f0_slice_samples = 0
        f0_source = (
            audio_cpu_t
            if f0_method == "swiftf0" and self.swiftf0 is not None
            else audio_t
        )
        f0_input = f0_source
        if f0_context_sec > 0:
            f0_ctx = int(round(f0_context_sec * 16000 / hubert_hop)) * hubert_hop
            # Fixed slice size: clamp to [F0_MIN_WINDOW_16K, hubert context].
            # Constant shape = single XPU kernel compilation, and never below
            # the F0 model's minimum input.
            f0_slice_samples = min(
                max(new_samples + max(0, f0_ctx), F0_MIN_WINDOW_16K),
                max_hubert_context_16k,
            )
            # Real-audio coverage; the shortfall (early chunks with a short
            # history) is taken from the existing right reflect padding.
            f0_window_samples = min(input_length, f0_slice_samples)
            pad_extend = f0_slice_samples - f0_window_samples
            win_start = input_length - f0_window_samples
            win_end = t_pad + input_length + t_pad + pad_extend
            f0_input = f0_source[:, win_start:win_end]

        # Parallel HuBERT + F0 extraction
        # HuBERT/F0 use the same source audio to preserve alignment.
        features = None
        f0_raw = None
        use_f0 = self.has_f0 and f0_method != "none"

        graph_capture_pending = self.hubert.graph_capture_pending(
            audio_t, output_layer, output_dim
        )
        if graph_capture_pending:
            logger.info("HuBERT Accelerator Graph capture will run before parallel F0")

        if use_parallel_extraction and use_f0 and not graph_capture_pending:
            def extract_hubert():
                tok = prof.start() if prof is not None else None
                with torch.autocast(device_type=self.device, dtype=self.dtype):
                    feats = self.hubert.extract(
                        audio_t, output_layer=output_layer, output_dim=output_dim
                    )
                if prof is not None:
                    prof.stop("hubert_ms", tok)
                return feats

            def extract_f0():
                # SwiftF0 runs synchronously on CPU — wall time is exact
                # there; the GPU methods use device timing events.
                on_device = f0_method != "swiftf0"
                tok = prof.start() if (prof is not None and on_device) else None
                wall0 = time.perf_counter()
                result = None
                if f0_method == "fcpe" and self.fcpe is not None:
                    with torch.autocast(device_type=self.device, dtype=self.dtype):
                        result = self.fcpe.infer(
                            f0_input, threshold=FCPE_VOICING_THRESHOLD
                        )
                elif f0_method == "swiftf0" and self.swiftf0 is not None:
                    result = self.swiftf0.infer(
                        f0_input, threshold=SWIFTF0_VOICING_THRESHOLD
                    )
                elif self.rmvpe is not None:
                    with torch.autocast(device_type=self.device, dtype=self.dtype):
                        result = self.rmvpe.infer(
                            f0_input, threshold=RMVPE_VOICING_THRESHOLD
                        )
                else:
                    logger.warning(
                        "[INFER] F0 parallel extraction returned None "
                        f"(method={f0_method}, fcpe={'loaded' if self.fcpe else 'None'}, "
                        f"swiftf0={'loaded' if self.swiftf0 else 'None'}, "
                        f"rmvpe={'loaded' if self.rmvpe else 'None'})"
                    )
                if prof is not None:
                    if on_device:
                        prof.stop("f0_ms", tok)
                    else:
                        prof.stop_wall("f0_ms", wall0)
                return result

            with ThreadPoolExecutor(max_workers=2) as executor:
                hubert_future = executor.submit(extract_hubert)
                f0_future = executor.submit(extract_f0)
                features = hubert_future.result()
                f0_raw = f0_future.result()
        else:
            tok = prof.start() if prof is not None else None
            with torch.autocast(device_type=self.device, dtype=self.dtype):
                features = self.hubert.extract(
                    audio_t, output_layer=output_layer, output_dim=output_dim
                )
            if prof is not None:
                prof.stop("hubert_ms", tok)

        # FAISS index retrieval (internally synced by the .cpu() transfer)
        if index_rate > 0 and self.faiss_index is not None:
            t0 = time.perf_counter()
            features = self._search_index(features, index_rate, k=index_k)
            if prof is not None:
                prof.stop_wall("faiss_ms", t0)

        # Interpolate features 2x (50fps -> 100fps for synthesizer)
        features = torch.nn.functional.interpolate(
            features.permute(0, 2, 1),
            scale_factor=2,
            mode="linear",
            align_corners=False,
        ).permute(0, 2, 1)

        feature_lengths = torch.tensor(
            [features.shape[1]], dtype=torch.long, device=self.device
        )

        # F0 processing (same audio_t as HuBERT for perfect frame alignment)
        pitch = None
        pitchf = None
        voiced_mask_for_gate = None

        if use_f0:
            f0 = f0_raw
            if f0 is None:
                logger.debug(
                    "[INFER] F0 parallel result was None, trying sequential fallback "
                    f"(method={f0_method})"
                )
                on_device = f0_method != "swiftf0"
                tok = prof.start() if (prof is not None and on_device) else None
                wall0 = time.perf_counter()
                if f0_method == "fcpe" and self.fcpe is not None:
                    with torch.autocast(device_type=self.device, dtype=self.dtype):
                        f0 = self.fcpe.infer(
                            f0_input, threshold=FCPE_VOICING_THRESHOLD
                        )
                elif f0_method == "swiftf0" and self.swiftf0 is not None:
                    f0 = self.swiftf0.infer(
                        f0_input, threshold=SWIFTF0_VOICING_THRESHOLD
                    )
                elif self.rmvpe is not None:
                    with torch.autocast(device_type=self.device, dtype=self.dtype):
                        f0 = self.rmvpe.infer(
                            f0_input, threshold=RMVPE_VOICING_THRESHOLD
                        )
                if prof is not None:
                    if on_device:
                        prof.stop("f0_ms", tok)
                    else:
                        prof.stop_wall("f0_ms", wall0)

            if f0 is not None and f0.numel() > 0:
                if f0.device != torch.device(self.device):
                    f0 = f0.to(self.device)

                # FCPE smoothing
                if f0_method == "fcpe":
                    f0 = _smooth_fcpe_f0(f0)

                # Pitch shift
                if abs(residual_f0_shift) > 0.01:
                    f0 = torch.where(
                        f0 > 0, f0 * (2 ** (residual_f0_shift / 12)), f0
                    )
                if moe_strength > 0.0:
                    f0 = apply_moe_f0_style(f0, moe_strength)

                if f0_window_samples > 0:
                    # Windowed extraction: place the fresh window frames at
                    # the timeline tail and serve older frames from the
                    # streaming F0 history cache.
                    f0 = self._assemble_windowed_f0(
                        f0,
                        target_frames=features.shape[1],
                        window_samples=f0_window_samples,
                        slice_samples=f0_slice_samples,
                        new_samples=new_samples,
                        input_length=input_length,
                        t_pad=t_pad,
                    )
                # Align F0 to feature length.  Same deficit-at-right
                # convention as the batch path: rate-normalize to the 100fps
                # grid, then crop — a uniform resize would squeeze the
                # timeline and misplace pitch at the output region.
                elif f0.shape[1] != features.shape[1]:
                    f0 = _align_f0_frames(f0, features.shape[1] + 2)
                    f0 = f0[:, : features.shape[1]]

                # --- F0 cross-chunk filter continuity ---
                # Cache the output-region tail BEFORE filtering, then prepend
                # previous chunk's tail so filters see continuous context.
                samples_per_frame_f0 = self.sample_rate // 100
                trim_right_feat_f0 = max(
                    0,
                    (t_pad_tgt + extra_pad_tgt - hubert_deficit)
                    // samples_per_frame_f0,
                )
                output_rightmost_feat = f0.shape[1] - trim_right_feat_f0
                cache_end = max(0, output_rightmost_feat)
                cache_start = max(0, cache_end - F0_HISTORY_FRAMES)
                new_f0_tail = f0[:, cache_start:cache_end].clone()

                # Prepend history for filter warmup
                f0_history_len = 0
                if self._streaming_f0_pre_filter_tail is not None:
                    history = self._streaming_f0_pre_filter_tail
                    f0_history_len = history.shape[1]
                    f0_extended = torch.cat([history, f0], dim=1)
                else:
                    f0_extended = f0

                # Apply the shared F0 post-processing chain on extended F0
                # (same median->lowpass->octave->slew chain as batch infer).
                f0_extended = apply_f0_filter_chain(
                    f0_extended,
                    f0_lowpass_cutoff_hz=f0_lowpass_cutoff_hz,
                    enable_octave_flip_suppress=enable_octave_flip_suppress,
                    enable_f0_slew_limit=enable_f0_slew_limit,
                    f0_slew_max_step_st=f0_slew_max_step_st,
                    f0_hole_fill_ms=f0_hole_fill_ms,
                )

                # Strip history prefix to restore original size
                f0 = f0_extended[:, f0_history_len:]

                # Update cache (pre-filter F0)
                self._streaming_f0_pre_filter_tail = new_f0_tail

                pitchf = f0.to(self.dtype)

                # Quantized pitch for embedding (shared with batch infer)
                pitch, voiced_mask_for_gate = quantize_f0_to_pitch(f0)

            if pitch is None:
                logger.warning(
                    "[INFER] F0 extraction failed — using flat pitch "
                    f"(method={f0_method}, f0_raw={'OK' if f0_raw is not None else 'None'}, "
                    f"audio_samples={audio_t.shape[-1]})"
                )
                # A missed chunk breaks the windowed-F0 history alignment.
                self._streaming_f0_hz_history = None
                pitch = torch.ones(
                    features.shape[0], features.shape[1],
                    dtype=torch.long, device=self.device,
                )
                pitchf = torch.zeros(
                    features.shape[0], features.shape[1],
                    dtype=self.dtype, device=self.device,
                )

        # Fallback: F0 model but f0_method="none" (e.g. overload protection)
        if self.has_f0 and pitch is None:
            logger.warning(
                "[INFER] F0 skipped — flat pitch fallback "
                f"(f0_method={f0_method}, use_f0={use_f0})"
            )
            self._streaming_f0_hz_history = None
            pitch = torch.ones(
                features.shape[0], features.shape[1],
                dtype=torch.long, device=self.device,
            )
            pitchf = torch.zeros(
                features.shape[0], features.shape[1],
                dtype=self.dtype, device=self.device,
            )

        # --- Feature cache: use real features from previous chunk instead
        # of reflect-padding when the current chunk is too short for the
        # synthesizer decoder.  This gives the decoder real context from
        # the preceding audio, dramatically improving output quality for
        # low-latency chunk sizes.
        cache_prepend_frames = 0
        new_features_for_cache = features.clone()
        new_pitch_for_cache = pitch.clone() if pitch is not None else None
        new_pitchf_for_cache = pitchf.clone() if pitchf is not None else None

        if (
            self._streaming_feat_cache is not None
            and features.shape[1] < MIN_SYNTH_FEATURE_FRAMES
        ):
            c_feat, c_pitch, c_pitchf = self._streaming_feat_cache
            need = MIN_SYNTH_FEATURE_FRAMES - features.shape[1]
            avail = c_feat.shape[1]
            cache_prepend_frames = min(need, avail)

            features = torch.cat(
                [c_feat[:, -cache_prepend_frames:], features], dim=1
            )
            feature_lengths = torch.tensor(
                [features.shape[1]], dtype=torch.long, device=self.device,
            )
            if pitch is not None and c_pitch is not None:
                pitch = torch.cat(
                    [c_pitch[:, -cache_prepend_frames:], pitch], dim=1
                )
            if pitchf is not None and c_pitchf is not None:
                pitchf = torch.cat(
                    [c_pitchf[:, -cache_prepend_frames:], pitchf], dim=1
                )
            # Extend voice gate mask to cover cached prefix (all-voiced,
            # since the cached portion is trimmed from the output anyway).
            if voiced_mask_for_gate is not None:
                cache_mask = torch.ones(
                    1, cache_prepend_frames,
                    dtype=voiced_mask_for_gate.dtype,
                    device=voiced_mask_for_gate.device,
                )
                voiced_mask_for_gate = torch.cat(
                    [cache_mask, voiced_mask_for_gate], dim=1
                )
            trim_left += cache_prepend_frames * (self.sample_rate // 100)

        # Fallback: reflect-pad if still too short (first chunk, no cache)
        synth_pad_frames = 0
        if features.shape[1] < MIN_SYNTH_FEATURE_FRAMES:
            synth_pad_frames = MIN_SYNTH_FEATURE_FRAMES - features.shape[1]
            pad_left = synth_pad_frames // 2
            pad_right = synth_pad_frames - pad_left

            current_size = features.shape[1]
            pad_mode = "replicate" if max(pad_left, pad_right) >= current_size else "reflect"

            features = torch.nn.functional.pad(
                features.permute(0, 2, 1), (pad_left, pad_right), mode=pad_mode,
            ).permute(0, 2, 1)
            feature_lengths = torch.tensor(
                [features.shape[1]], dtype=torch.long, device=self.device,
            )
            if pitch is not None:
                pitch = torch.nn.functional.pad(pitch, (pad_left, pad_right), mode=pad_mode)
            if pitchf is not None:
                pitchf = torch.nn.functional.pad(pitchf, (pad_left, pad_right), mode=pad_mode)

        # Save current chunk's features for next call's cache
        self._streaming_feat_cache = (
            new_features_for_cache,
            new_pitch_for_cache,
            new_pitchf_for_cache,
        )

        # --- Compute skip_head/return_length for streaming synthesis ---
        # TextEncoder processes ALL features for rich context, but Flow +
        # Decoder only synthesize the output region (new_hop + sola_extra).
        # This prevents SineGen phase accumulation through varying context,
        # which causes "a-na-na-" artifacts at chunk boundaries.
        samples_per_frame = self.sample_rate // 100

        # Total left context to skip (in model_sr samples)
        # trim_left already includes: t_pad_tgt + overlap_tgt - sola_extra + cache_prepend
        total_left_samples = trim_left
        if synth_pad_frames > 0:
            total_left_samples += (synth_pad_frames // 2) * samples_per_frame

        # Total right padding to skip (in model_sr samples)
        total_right_samples = trim_right
        if synth_pad_frames > 0:
            total_right_samples += (synth_pad_frames - synth_pad_frames // 2) * samples_per_frame

        skip_head_feat = total_left_samples // samples_per_frame
        trim_right_feat = total_right_samples // samples_per_frame

        # Clamp to valid range
        skip_head_feat = min(skip_head_feat, features.shape[1] - 1)
        return_length_feat = max(
            1, features.shape[1] - skip_head_feat - trim_right_feat
        )

        # Residual samples not covered by feature-level skip (sub-frame trim)
        residual_left = total_left_samples - skip_head_feat * samples_per_frame

        # Set SineGen harmonic phase mode / uv crossfade before synthesis
        self._set_fixed_harmonics(fixed_harmonics)
        self._set_uv_ramp(uv_ramp_ms)

        # Run synthesizer with skip_head/return_length
        t_synth = time.perf_counter()
        with torch.autocast(device_type=self.device, dtype=self.dtype):
            output = self.synthesizer.infer(
                features, feature_lengths, pitch=pitch, pitchf=pitchf,
                noise_scale=noise_scale,
                skip_head=skip_head_feat,
                return_length=return_length_feat,
                return_length2=return_length_feat,
                # Before history is full, skip_head moves every chunk. Capturing
                # those transient signatures wastes graph memory and can evict
                # the single steady-state graph needed by real-time inference.
                use_accelerator_graph=hubert_history_full,
            )

        # Voice gating (output is already the target region; shared with infer)
        if voice_gate_mode != "off" and voiced_mask_for_gate is not None:
            # Slice mask to match the synthesized output region
            gate_mask_src = voiced_mask_for_gate[
                :, skip_head_feat:skip_head_feat + return_length_feat
            ]
            output, _ = apply_voice_gate(
                output,
                gate_mask_src,
                voice_gate_mode=voice_gate_mode,
                energy_threshold=energy_threshold,
                sample_rate=self.sample_rate,
            )

        # Convert to numpy (the .cpu() transfer is the device sync point, so
        # synth_ms includes the real synthesis + gate execution time, and all
        # pending stage timing events are complete for finalize())
        output = output.cpu().float().numpy()
        if prof is not None:
            prof.stop_wall("synth_ms", t_synth)
            prof.finalize()
        if output.ndim == 2:
            output = output[0]

        # Trim sub-frame residual from left (skip_head rounds to feature boundaries)
        if residual_left > 0 and len(output) > residual_left:
            output = output[residual_left:]

        # Adjust to exact expected length.
        # With zc-aligned sola_extra_samples, residual_left should be 0 and
        # the output should be >= expected_output (right-trim conservative).
        # Trim excess; if short, pad tail with edge value (typically 0-2 samples
        # from rounding — not a quality concern).
        if len(output) > expected_output:
            output = output[:expected_output]
        elif len(output) < expected_output:
            output = np.pad(
                output, (0, expected_output - len(output)), mode="edge"
            )

        return output

    def get_info(self) -> dict:
        """Get information about the loaded pipeline."""
        if not self._loaded:
            return {"loaded": False}

        hubert_graph_stats = (
            self.hubert.graph_stats() if self.hubert is not None else None
        )
        synthesizer_graph_stats = (
            self.synthesizer.graph_stats()
            if self.synthesizer is not None
            else None
        )

        return {
            "loaded": True,
            "device": self.device,
            "dtype": str(self.dtype),
            "model_path": str(self.model_path),
            "index_path": str(self.index_path) if self.index_path else None,
            "has_index": self.faiss_index is not None,
            "index_vectors": self.faiss_index.ntotal if self.faiss_index else 0,
            "version": self.synthesizer.version if self.synthesizer else None,
            "has_f0": self.has_f0,
            "sample_rate": self.sample_rate,
            "use_compile": self.use_compile,
            "hubert_accelerator_graph": hubert_graph_stats,
            "synthesizer_accelerator_graph": synthesizer_graph_stats,
        }
