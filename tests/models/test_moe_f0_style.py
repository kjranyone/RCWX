from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rcwx.pipeline.inference import apply_moe_f0_style


def _male_like_f0(length: int = 240) -> torch.Tensor:
    t = torch.arange(length, dtype=torch.float32)
    base = 112.0 + 12.0 * torch.sin(2.0 * torch.pi * t / 54.0)
    base += 5.0 * torch.sin(2.0 * torch.pi * t / 17.0)
    # Short and long dropouts.
    base[60:63] = 0.0
    base[140:150] = 0.0
    return base.unsqueeze(0)


def _high_voice_f0(length: int = 240) -> torch.Tensor:
    t = torch.arange(length, dtype=torch.float32)
    base = 275.0 + 22.0 * torch.sin(2.0 * torch.pi * t / 48.0)
    base += 6.0 * torch.sin(2.0 * torch.pi * t / 19.0)
    base[120:124] = 0.0
    return base.unsqueeze(0)


def test_moe_strength_zero_is_identity() -> None:
    f0 = _male_like_f0()
    out = apply_moe_f0_style(f0, 0.0)
    assert torch.equal(out, f0), "strength=0 should return input unchanged"


def test_moe_lifts_male_register_and_floor() -> None:
    f0 = _male_like_f0()
    out = apply_moe_f0_style(f0, 1.0)

    in_voiced = f0[f0 > 0]
    out_voiced = out[out > 0]

    in_med = torch.median(in_voiced).item()
    out_med = torch.median(out_voiced).item()
    in_p10 = torch.quantile(in_voiced, 0.10).item()
    out_p10 = torch.quantile(out_voiced, 0.10).item()

    assert out_med > in_med * 1.35, (
        f"Expected strong register lift: in_med={in_med:.2f}, out_med={out_med:.2f}"
    )
    assert out_p10 > in_p10 * 1.40, (
        f"Expected low-floor lift: in_p10={in_p10:.2f}, out_p10={out_p10:.2f}"
    )


def test_moe_fills_short_gap_but_keeps_long_gap() -> None:
    f0 = _male_like_f0()
    out = apply_moe_f0_style(f0, 1.0)

    # short gap [60:63] should be interpolated and voiced
    assert torch.all(out[0, 60:63] > 0.0), "Short unvoiced gap should be filled"
    # long gap [140:150] should stay unvoiced at least in part
    assert torch.any(out[0, 140:150] == 0.0), "Long unvoiced gap should not be fully filled"


def test_moe_does_not_overboost_high_voice() -> None:
    f0 = _high_voice_f0()
    out = apply_moe_f0_style(f0, 1.0)

    in_voiced = f0[f0 > 0]
    out_voiced = out[out > 0]

    in_med = torch.median(in_voiced).item()
    out_med = torch.median(out_voiced).item()

    assert out_med < in_med * 1.20, (
        f"High voice should not be heavily boosted: in={in_med:.2f}, out={out_med:.2f}"
    )


def test_moe_strength_monotonic_for_male_register() -> None:
    f0 = _male_like_f0()
    out_0 = apply_moe_f0_style(f0, 0.0)[0]
    out_03 = apply_moe_f0_style(f0, 0.3)[0]
    out_08 = apply_moe_f0_style(f0, 0.8)[0]

    med_0 = torch.median(out_0[out_0 > 0]).item()
    med_03 = torch.median(out_03[out_03 > 0]).item()
    med_08 = torch.median(out_08[out_08 > 0]).item()
    assert med_0 < med_03 < med_08


if __name__ == "__main__":
    tests = [
        test_moe_strength_zero_is_identity,
        test_moe_lifts_male_register_and_floor,
        test_moe_fills_short_gap_but_keeps_long_gap,
        test_moe_does_not_overboost_high_voice,
        test_moe_strength_monotonic_for_male_register,
    ]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
