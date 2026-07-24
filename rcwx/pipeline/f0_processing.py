"""F0 (pitch contour) DSP: smoothing, stylization, filter chain, quantization.

Pure functions over [B, T] F0 tracks (Hz, 0 = unvoiced), shared by batch
``RVCPipeline.infer`` and streaming ``infer_streaming``.  Hot-path stages keep
the torch<->numpy round-trips to a single D2H/H2D pair (see
``apply_f0_filter_chain``).
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch
from scipy.signal import butter, filtfilt, medfilt

# Upper bound for moe_boost strength (F0-only stylization intensity).
MAX_MOE_BOOST = 1.0
# F0 correction hysteresis: corrections sustained this many consecutive
# frames are treated as genuine pitch transitions and accepted (the raw
# values are restored), so single-frame glitch suppression never latches
# onto real octave jumps or fast slides.
F0_CORRECTION_SUSTAIN_FRAMES = 3


def _f0_to_numpy(f0: torch.Tensor) -> np.ndarray:
    """Detach F0 to a contiguous float32 numpy array [B, T]."""
    return f0.detach().cpu().to(torch.float32).numpy()


def _f0_from_numpy(
    f0_np: np.ndarray,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Upload a float32 numpy F0 track back to the caller's device/dtype."""
    return torch.from_numpy(np.ascontiguousarray(f0_np, dtype=np.float32)).to(
        device=device, dtype=dtype
    )


@lru_cache(maxsize=16)
def _f0_lowpass_ba(cutoff_hz: float, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Cached Butterworth lowpass coefficients for the F0 contour filter."""
    nyquist = sample_rate / 2.0
    return butter(2, cutoff_hz / nyquist, btype="low")


def _smooth_f0_spikes_np(f0_np: np.ndarray, window: int = 3) -> np.ndarray:
    """Median-filter voiced regions of an F0 track (numpy, in-place safe copy)."""
    if f0_np.shape[1] < window:
        return f0_np
    result = np.empty_like(f0_np, dtype=np.float32)
    for b in range(f0_np.shape[0]):
        row = f0_np[b]
        voiced = row > 0
        if np.any(voiced):
            filtered = medfilt(row.astype(np.float64), window).astype(np.float32)
            result[b] = np.where(voiced, filtered, row)
        else:
            result[b] = row
    return result


def _lowpass_f0_np(
    f0_np: np.ndarray,
    cutoff_hz: float = 16.0,
    sample_rate: float = 100.0,
) -> np.ndarray:
    """Zero-phase Butterworth lowpass on voiced F0 (log2 domain, numpy)."""
    if f0_np.shape[1] < 10:
        return f0_np

    b, a = _f0_lowpass_ba(float(cutoff_hz), float(sample_rate))
    result = np.empty_like(f0_np, dtype=np.float32)
    idx = np.arange(f0_np.shape[1])

    for batch in range(f0_np.shape[0]):
        row = f0_np[batch]
        voiced = row > 0
        if int(np.sum(voiced)) > 10:
            voiced_indices = idx[voiced]
            log_contour = np.interp(
                idx, voiced_indices, np.log2(row[voiced_indices])
            )
            try:
                filtered = filtfilt(b, a, log_contour)
                result[batch] = np.where(
                    voiced, np.exp2(filtered).astype(np.float32), 0.0
                )
            except ValueError:
                result[batch] = row
        else:
            result[batch] = row
    return result


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
    return _f0_from_numpy(
        _smooth_f0_spikes_np(_f0_to_numpy(f0), window=window),
        device=f0.device,
        dtype=f0.dtype,
    )


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
    return _f0_from_numpy(
        _lowpass_f0_np(_f0_to_numpy(f0), cutoff_hz=cutoff_hz, sample_rate=sample_rate),
        device=f0.device,
        dtype=f0.dtype,
    )


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


def align_f0_to_feature_grid(f0: torch.Tensor, feature_frames: int) -> torch.Tensor:
    """Align F0 to feature length with the deficit-at-right convention.

    Features are ~2 frames short at the RIGHT edge (HuBERT frame deficit —
    the same convention as the right trim), so a uniform resize would squeeze
    the whole F0 timeline and misplace pitch by up to ~20ms at the tail.
    Rate-normalize to the 100fps grid (+2 frames), then crop the deficit on
    the right.
    """
    if f0.shape[1] == feature_frames:
        return f0
    f0 = _align_f0_frames(f0, feature_frames + 2)
    return f0[:, :feature_frames]


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


def _suppress_octave_flips_np(
    f0_np: np.ndarray,
    octave_ratio_center: float = 2.0,
    octave_ratio_tolerance: float = 0.16,
    sustain_frames: int = F0_CORRECTION_SUSTAIN_FRAMES,
) -> np.ndarray:
    """Suppress transient +-1 octave flips on a numpy F0 track."""
    if f0_np.size == 0 or f0_np.shape[1] < 2:
        return f0_np

    raw = f0_np
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
            cur = float(raw[b, i])
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
                corrected[b, streak_start : i + 1] = raw[b, streak_start : i + 1]
                streak_dir = 0
                streak_len = 0

    return corrected


def _limit_f0_slew_np(
    f0_np: np.ndarray,
    max_step_st: float = 2.8,
    sustain_frames: int = F0_CORRECTION_SUSTAIN_FRAMES,
) -> np.ndarray:
    """Clamp short frame-to-frame F0 flutter on a numpy F0 track."""
    if f0_np.size == 0 or f0_np.shape[1] < 2:
        return f0_np

    raw = f0_np
    corrected = f0_np.copy()
    max_ratio = float(2 ** (max_step_st / 12.0))

    for b in range(corrected.shape[0]):
        streak_dir = 0
        streak_start = 0
        streak_len = 0
        for i in range(1, corrected.shape[1]):
            prev = float(corrected[b, i - 1])
            cur = float(raw[b, i])
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
                corrected[b, streak_start : i + 1] = raw[b, streak_start : i + 1]
                streak_dir = 0
                streak_len = 0

    return corrected


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
    return _f0_from_numpy(
        _suppress_octave_flips_np(
            _f0_to_numpy(f0),
            octave_ratio_center=octave_ratio_center,
            octave_ratio_tolerance=octave_ratio_tolerance,
            sustain_frames=sustain_frames,
        ),
        device=f0.device,
        dtype=f0.dtype,
    )


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
    return _f0_from_numpy(
        _limit_f0_slew_np(
            _f0_to_numpy(f0),
            max_step_st=max_step_st,
            sustain_frames=sustain_frames,
        ),
        device=f0.device,
        dtype=f0.dtype,
    )


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


def _fill_short_unvoiced_gaps_np(
    f0_np: np.ndarray,
    max_gap_frames: int = 3,
) -> np.ndarray:
    """Fill short unvoiced holes by log2 interpolation (numpy, may copy)."""
    if max_gap_frames <= 0 or f0_np.shape[1] < 3:
        return f0_np

    # Work on a copy only if we actually write a fill.
    out = f0_np
    wrote = False
    for b in range(f0_np.shape[0]):
        row_src = f0_np[b]
        vidx = np.flatnonzero(row_src > 0)
        if vidx.size < 2:
            continue
        gaps = np.diff(vidx) - 1
        fillable = np.flatnonzero((gaps > 0) & (gaps <= max_gap_frames))
        if fillable.size == 0:
            continue
        if not wrote:
            out = f0_np.copy()
            wrote = True
        row = out[b]
        for gi in fillable:
            left = int(vidx[gi])
            right = int(vidx[gi + 1])
            n = right - left
            t = np.arange(1, n, dtype=np.float32) / n
            lv = np.log2(row[left])
            rv = np.log2(row[right])
            row[left + 1 : right] = np.exp2(lv * (1.0 - t) + rv * t)
    return out


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

    f0_np = _f0_to_numpy(f0)
    filled = _fill_short_unvoiced_gaps_np(f0_np, max_gap_frames=max_gap_frames)
    if filled is f0_np:
        return f0
    return _f0_from_numpy(filled, device=f0.device, dtype=f0.dtype)


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

    Hot path: one host transfer, all stages in numpy, one upload.  The previous
    staged implementation paid 4-5 torch<->numpy round-trips per hop.
    """
    if f0.numel() == 0:
        return f0

    max_gap_frames = int(round(f0_hole_fill_ms / 10.0))
    # Single D2H (no-op when already on CPU, as with SwiftF0).
    x = _f0_to_numpy(f0)
    x = _fill_short_unvoiced_gaps_np(x, max_gap_frames=max_gap_frames)
    x = _smooth_f0_spikes_np(x, window=3)
    x = _lowpass_f0_np(x, cutoff_hz=f0_lowpass_cutoff_hz, sample_rate=100.0)
    if enable_octave_flip_suppress:
        x = _suppress_octave_flips_np(x)
    if enable_f0_slew_limit:
        x = _limit_f0_slew_np(x, max_step_st=f0_slew_max_step_st)
    return _f0_from_numpy(x, device=f0.device, dtype=f0.dtype)


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
