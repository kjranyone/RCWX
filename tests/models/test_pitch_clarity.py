"""Test pitch clarity improvements — signal processing unit tests.

No model required. Tests lowpass_f0 cutoff behavior and FCPE smoothing kernel.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rcwx.pipeline.inference import lowpass_f0, suppress_octave_flips, limit_f0_slew


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_f0_with_modulation(
    base_hz: float = 200.0,
    mod_hz: float = 12.0,
    mod_depth: float = 20.0,
    length: int = 200,
    sample_rate: float = 100.0,
) -> torch.Tensor:
    """Create an F0 contour with a sinusoidal modulation component.

    Returns [1, length] tensor in Hz.
    """
    t = torch.arange(length, dtype=torch.float32) / sample_rate
    f0 = base_hz + mod_depth * torch.sin(2 * torch.pi * mod_hz * t)
    return f0.unsqueeze(0)


def _modulation_power_ratio(
    f0_original: torch.Tensor,
    f0_filtered: torch.Tensor,
    mod_hz: float,
    sample_rate: float = 100.0,
) -> float:
    """Measure how much of a specific modulation frequency survives filtering.

    Returns ratio of modulation amplitude after/before filtering (0.0-1.0).
    """
    n = f0_original.shape[1]
    freqs = torch.fft.rfftfreq(n, 1.0 / sample_rate)

    # Find bin closest to mod_hz
    idx = torch.argmin(torch.abs(freqs - mod_hz)).item()

    orig_spec = torch.abs(torch.fft.rfft(f0_original[0] - f0_original[0].mean()))
    filt_spec = torch.abs(torch.fft.rfft(f0_filtered[0] - f0_filtered[0].mean()))

    orig_power = orig_spec[idx].item()
    if orig_power < 1e-6:
        return 1.0  # No modulation to measure
    return filt_spec[idx].item() / orig_power


def _smooth_fcpe_f0(f0: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Reference FCPE smoothing (matches pipeline implementation)."""
    f0_smooth = torch.nn.functional.avg_pool1d(
        f0.unsqueeze(1), kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
    ).squeeze(1)
    return torch.where(f0 > 0, f0_smooth, f0)


# ---------------------------------------------------------------------------
# Tests: lowpass_f0 cutoff behavior
# ---------------------------------------------------------------------------

def test_lowpass_f0_default_preserves_12hz():
    """Default cutoff (16Hz) should preserve 12Hz modulation (>50% power)."""
    f0 = _make_f0_with_modulation(mod_hz=12.0, length=200)
    filtered = lowpass_f0(f0)  # uses default cutoff
    ratio = _modulation_power_ratio(f0, filtered, mod_hz=12.0)
    print(f"  Default cutoff: 12Hz power ratio = {ratio:.3f} (need >0.50)")
    assert ratio > 0.50, f"Default cutoff should preserve 12Hz; got ratio={ratio:.3f}"


def test_lowpass_f0_8hz_attenuates_12hz():
    """cutoff=8Hz should attenuate 12Hz modulation (<30% power)."""
    f0 = _make_f0_with_modulation(mod_hz=12.0, length=200)
    filtered = lowpass_f0(f0, cutoff_hz=8.0)
    ratio = _modulation_power_ratio(f0, filtered, mod_hz=12.0)
    print(f"  8Hz cutoff: 12Hz power ratio = {ratio:.3f} (need <0.30)")
    assert ratio < 0.30, f"8Hz cutoff should attenuate 12Hz; got ratio={ratio:.3f}"


def test_lowpass_f0_16hz_preserves_12hz():
    """cutoff=16Hz should preserve 12Hz modulation (>70% power)."""
    f0 = _make_f0_with_modulation(mod_hz=12.0, length=200)
    filtered = lowpass_f0(f0, cutoff_hz=16.0)
    ratio = _modulation_power_ratio(f0, filtered, mod_hz=12.0)
    print(f"  16Hz cutoff: 12Hz power ratio = {ratio:.3f} (need >0.70)")
    assert ratio > 0.70, f"16Hz cutoff should preserve 12Hz; got ratio={ratio:.3f}"


def test_lowpass_f0_preserves_unvoiced():
    """Unvoiced regions (f0=0) must remain 0 after filtering."""
    f0 = _make_f0_with_modulation(length=100)
    # Insert unvoiced gap
    f0[0, 30:50] = 0.0
    filtered = lowpass_f0(f0, cutoff_hz=16.0)
    unvoiced_after = filtered[0, 30:50]
    assert torch.all(unvoiced_after == 0.0), (
        f"Unvoiced region should stay 0; got max={unvoiced_after.abs().max().item():.6f}"
    )


def test_lowpass_f0_short_input_passthrough():
    """Input shorter than 10 frames should pass through unchanged."""
    f0 = torch.tensor([[200.0, 210.0, 220.0, 230.0, 240.0]])
    filtered = lowpass_f0(f0, cutoff_hz=16.0)
    assert torch.allclose(f0, filtered), "Short input should pass through"


# ---------------------------------------------------------------------------
# Tests: FCPE smoothing kernel
# ---------------------------------------------------------------------------

def test_fcpe_kernel3_sharper_than_kernel5():
    """kernel=3 should preserve step transitions better than kernel=5."""
    # Create a step function: low pitch -> high pitch
    f0 = torch.ones(1, 100) * 200.0
    f0[0, 50:] = 400.0

    smooth3 = _smooth_fcpe_f0(f0, kernel_size=3)
    smooth5 = _smooth_fcpe_f0(f0, kernel_size=5)

    # Measure transition sharpness: diff at step boundary
    diff3 = abs(smooth3[0, 50].item() - smooth3[0, 49].item())
    diff5 = abs(smooth5[0, 50].item() - smooth5[0, 49].item())
    print(f"  Step diff: kernel3={diff3:.1f}Hz, kernel5={diff5:.1f}Hz")
    assert diff3 > diff5, f"kernel=3 should be sharper: diff3={diff3:.1f} vs diff5={diff5:.1f}"


def test_fcpe_smoothing_preserves_voiced_mask():
    """Smoothing should preserve unvoiced=0 regions."""
    f0 = torch.ones(1, 50) * 250.0
    f0[0, 20:30] = 0.0  # unvoiced gap

    smoothed = _smooth_fcpe_f0(f0, kernel_size=3)
    assert torch.all(smoothed[0, 20:30] == 0.0), "Unvoiced gap should remain 0"


# ---------------------------------------------------------------------------
# Tests: octave flip suppression
# ---------------------------------------------------------------------------

def test_suppress_octave_flips_halves_obvious_double():
    """An isolated ~2x jump should be corrected back near the previous contour."""
    f0 = torch.tensor([[220.0, 224.0, 446.0, 228.0, 232.0]], dtype=torch.float32)
    fixed = suppress_octave_flips(f0)
    # Frame 2 should be corrected close to ~223Hz (not stay around 446Hz)
    assert abs(fixed[0, 2].item() - 223.0) < 8.0, (
        f"Expected octave correction near 223Hz, got {fixed[0, 2].item():.2f}Hz"
    )


def test_suppress_octave_flips_keeps_natural_contour():
    """Normal melodic movement must remain unchanged."""
    f0 = torch.tensor([[220.0, 226.0, 234.0, 246.0, 258.0]], dtype=torch.float32)
    fixed = suppress_octave_flips(f0)
    assert torch.allclose(f0, fixed), "Natural contour should not be modified"


def test_limit_f0_slew_clamps_large_step():
    """Large single-frame jump should be clamped by slew limiter."""
    f0 = torch.tensor([[220.0, 230.0, 420.0]], dtype=torch.float32)
    fixed = limit_f0_slew(f0, max_step_st=2.0)
    # 220->230 is preserved, 230->420 should be reduced significantly
    assert fixed[0, 1].item() == 230.0
    assert fixed[0, 2].item() < 320.0, f"Expected clamped jump, got {fixed[0,2].item():.2f}"


def test_limit_f0_slew_keeps_small_steps():
    """Small steps below threshold should pass unchanged."""
    f0 = torch.tensor([[220.0, 226.0, 232.0, 239.0]], dtype=torch.float32)
    fixed = limit_f0_slew(f0, max_step_st=3.0)
    assert torch.allclose(f0, fixed), "Small steps should not be modified"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_lowpass_f0_default_preserves_12hz,
        test_lowpass_f0_8hz_attenuates_12hz,
        test_lowpass_f0_16hz_preserves_12hz,
        test_lowpass_f0_preserves_unvoiced,
        test_lowpass_f0_short_input_passthrough,
        test_fcpe_kernel3_sharper_than_kernel5,
        test_fcpe_smoothing_preserves_voiced_mask,
        test_suppress_octave_flips_halves_obvious_double,
        test_suppress_octave_flips_keeps_natural_contour,
        test_limit_f0_slew_clamps_large_step,
        test_limit_f0_slew_keeps_small_steps,
    ]
    passed = 0
    failed = 0
    for t in tests:
        name = t.__name__
        try:
            print(f"Running {name}...")
            t()
            print(f"  PASS")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
