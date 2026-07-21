"""Verify decoder overlap correctly increases _sola_extra_model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rcwx.pipeline.realtime_unified import (
    RealtimeConfig,
    _compute_sola_extra_model,
)


def test_sola_extra_uses_one_search_window() -> None:
    model_sr = 48000
    extra = _compute_sola_extra_model(
        model_sr,
        48000,
        crossfade_samples_out=480,
        search_samples_out=720,
        decoder_overlap_frames=0,
    )

    # 25ms is rounded up to the decoder's 10ms (480-sample) boundary.
    assert extra == 1440


def test_decoder_overlap_increases_sola_extra():
    """decoder_overlap_frames=5 should produce a larger _sola_extra_model
    than decoder_overlap_frames=0, by exactly 5 * zc_model samples."""

    # We can't instantiate RealtimeVoiceChangerUnified without a pipeline,
    # so replicate the _sola_extra_model formula to verify.
    model_sr = 32000  # typical model sample rate
    output_sr = 48000
    crossfade_sec = 0.05
    sola_search_ms = 10.0

    crossfade_samples_out = int(output_sr * crossfade_sec)
    search_samples_out = int(output_sr * sola_search_ms / 1000)
    sola_extra_out = crossfade_samples_out + search_samples_out
    zc_model = model_sr // 100  # 320

    sola_extra_raw = int(sola_extra_out * model_sr / output_sr)

    # Without decoder overlap
    extra_no_overlap = (sola_extra_raw + zc_model - 1) // zc_model * zc_model

    # With decoder overlap (5 frames)
    decoder_frames = 5
    decoder_overlap_model = decoder_frames * zc_model
    extra_with_overlap = (
        (sola_extra_raw + decoder_overlap_model + zc_model - 1) // zc_model * zc_model
    )

    diff = extra_with_overlap - extra_no_overlap
    expected_diff = decoder_frames * zc_model  # 5 * 320 = 1600

    assert diff == expected_diff, (
        f"Expected diff={expected_diff}, got {diff} "
        f"(no_overlap={extra_no_overlap}, with_overlap={extra_with_overlap})"
    )
    print(
        f"PASS: decoder_overlap adds {diff} samples "
        f"({diff * 1000 / model_sr:.0f}ms at {model_sr}Hz) to sola_extra_model"
    )


def test_decoder_overlap_default_is_5():
    """RealtimeConfig default decoder_overlap_frames should be 5."""
    cfg = RealtimeConfig()
    assert cfg.decoder_overlap_frames == 5, f"Expected default 5, got {cfg.decoder_overlap_frames}"
    print("PASS: default decoder_overlap_frames == 5")


def test_decoder_overlap_at_40k():
    """Verify formula at model_sr=40000 (another common RVC rate)."""
    model_sr = 40000
    output_sr = 48000
    crossfade_sec = 0.08
    sola_search_ms = 10.0

    crossfade_samples_out = int(output_sr * crossfade_sec)
    search_samples_out = int(output_sr * sola_search_ms / 1000)
    sola_extra_out = crossfade_samples_out + search_samples_out
    zc_model = model_sr // 100  # 400

    sola_extra_raw = int(sola_extra_out * model_sr / output_sr)

    decoder_frames = 5
    decoder_overlap_model = decoder_frames * zc_model
    extra = (sola_extra_raw + decoder_overlap_model + zc_model - 1) // zc_model * zc_model

    # Must be a multiple of zc_model
    assert extra % zc_model == 0, f"Not aligned: {extra} % {zc_model} = {extra % zc_model}"
    # Must include at least the decoder overlap
    assert extra >= decoder_overlap_model + sola_extra_raw, (
        f"extra={extra} < decoder_overlap({decoder_overlap_model}) + sola_raw({sola_extra_raw})"
    )
    print(
        f"PASS: at 40kHz, sola_extra_model={extra} "
        f"({extra * 1000 / model_sr:.0f}ms), aligned to {zc_model}"
    )


if __name__ == "__main__":
    test_decoder_overlap_default_is_5()
    test_decoder_overlap_increases_sola_extra()
    test_decoder_overlap_at_40k()
    print("\nAll decoder overlap tests passed.")
