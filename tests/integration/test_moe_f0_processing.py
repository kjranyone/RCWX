"""Tests for F0-only moe processing.

Verifies:
1. moe_boost is identity at zero,
2. low-register F0 is lifted but bounded,
3. short unvoiced gaps are filled while longer gaps survive,
4. high-register F0 is not overboosted,
5. a marginally-voiced window blends toward raw F0 instead of snapping to the
   register floor (streaming chunk-boundary continuity),
6. config persistence keeps moe_boost and drops removed pre-HuBERT settings.
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rcwx.config import RCWXConfig
from rcwx.pipeline.inference import apply_moe_f0_style

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _male_like_f0(length: int = 240) -> torch.Tensor:
    t = torch.arange(length, dtype=torch.float32)
    base = 112.0 + 12.0 * torch.sin(2.0 * torch.pi * t / 54.0)
    base += 5.0 * torch.sin(2.0 * torch.pi * t / 17.0)
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
    assert torch.equal(out, f0)


def test_moe_lifts_low_register_but_stays_bounded() -> None:
    f0 = _male_like_f0()
    out = apply_moe_f0_style(f0, 1.0)

    in_med = torch.median(f0[f0 > 0]).item()
    out_med = torch.median(out[out > 0]).item()

    assert out_med > in_med * 1.25, (
        f"Expected low-register lift: in={in_med:.2f}Hz, out={out_med:.2f}Hz"
    )
    assert out_med < in_med * 1.75, (
        f"F0-only lift should stay bounded: in={in_med:.2f}Hz, out={out_med:.2f}Hz"
    )


def test_moe_fills_short_gap_but_keeps_long_gap() -> None:
    f0 = _male_like_f0()
    out = apply_moe_f0_style(f0, 1.0)

    assert torch.all(out[0, 60:63] > 0.0), "Short unvoiced gap should be filled"
    assert torch.any(out[0, 140:150] == 0.0), "Long unvoiced gap should not be fully filled"


def test_moe_does_not_overboost_high_voice() -> None:
    f0 = _high_voice_f0()
    out = apply_moe_f0_style(f0, 1.0)

    in_med = torch.median(f0[f0 > 0]).item()
    out_med = torch.median(out[out > 0]).item()

    assert out_med < in_med * 1.15, (
        f"High voice should not be heavily boosted: in={in_med:.2f}, out={out_med:.2f}"
    )


def test_moe_low_confidence_window_does_not_snap_to_floor() -> None:
    """A marginally-voiced window must blend toward raw F0 rather than force its
    deep frames up to the register floor.

    Regression guard for the streaming chunk-boundary pitch step: with the old
    per-coefficient ``effective_strength`` gate, a window whose voiced_ratio sat
    just above 0.12 still ran the full pipeline and floored every voiced frame
    (~111Hz at moe=0.45), while the adjacent sub-0.12 window returned raw F0 --
    an audible ~1.5 semitone step. The confidence blend must keep this window
    near its raw 95Hz.
    """
    length = 240
    row = torch.zeros(length)
    # ~13% voiced (just above the 0.12 confidence gate) at a deep 95 Hz.
    row[:32] = 95.0
    f0 = row.unsqueeze(0)

    out = apply_moe_f0_style(f0, 0.45)
    voiced_out = out[out > 0]

    assert float(voiced_out.max().item()) < 99.0, (
        "Low-confidence window snapped upward instead of blending toward raw F0: "
        f"max={float(voiced_out.max().item()):.1f}Hz (floor would impose ~111Hz)"
    )


def test_config_roundtrip_drops_removed_pre_hubert_key() -> None:
    config = RCWXConfig()
    config.inference.moe_boost = 0.60

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        tmp_path = Path(f.name)

    try:
        config.save(tmp_path)
        with open(tmp_path, encoding="utf-8") as f:
            data = json.load(f)
        assert "pre_hubert_pitch_ratio" not in data["inference"]
        assert data["inference"]["moe_boost"] == 0.60

        data["inference"]["pre_hubert_pitch_ratio"] = 0.75
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        loaded = RCWXConfig.load(tmp_path)
        assert not hasattr(loaded.inference, "pre_hubert_pitch_ratio")
        assert loaded.inference.moe_boost == 0.60
    finally:
        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    tests = [
        test_moe_strength_zero_is_identity,
        test_moe_lifts_low_register_but_stays_bounded,
        test_moe_fills_short_gap_but_keeps_long_gap,
        test_moe_does_not_overboost_high_voice,
        test_moe_low_confidence_window_does_not_snap_to_floor,
        test_config_roundtrip_drops_removed_pre_hubert_key,
    ]
    for test in tests:
        test()
        logger.info("PASS: %s", test.__name__)
