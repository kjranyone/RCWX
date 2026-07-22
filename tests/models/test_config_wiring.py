"""Test config wiring for pitch clarity fields.

No model required. Verifies new fields exist with correct defaults and
survive JSON round-trip.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rcwx.config import AudioConfig, DenoiseConfig, InferenceConfig, RCWXConfig
from rcwx.gui.widgets.latency_settings import _auto_params, _minimum_chunk_ms
from rcwx.pipeline.realtime_unified import RealtimeConfig

# ---------------------------------------------------------------------------
# RealtimeConfig tests
# ---------------------------------------------------------------------------


def test_realtime_config_has_noise_scale():
    """RealtimeConfig should have noise_scale=0.4 by default."""
    cfg = RealtimeConfig()
    assert hasattr(cfg, "noise_scale"), "RealtimeConfig missing noise_scale"
    assert cfg.noise_scale == 0.4, f"Expected 0.4, got {cfg.noise_scale}"


def test_realtime_config_has_f0_lowpass_cutoff():
    """RealtimeConfig should have f0_lowpass_cutoff_hz=16.0 by default."""
    cfg = RealtimeConfig()
    assert hasattr(cfg, "f0_lowpass_cutoff_hz"), "RealtimeConfig missing f0_lowpass_cutoff_hz"
    assert cfg.f0_lowpass_cutoff_hz == 16.0, f"Expected 16.0, got {cfg.f0_lowpass_cutoff_hz}"


def test_realtime_config_custom_values():
    """RealtimeConfig should accept custom noise_scale and f0_lowpass_cutoff_hz."""
    cfg = RealtimeConfig(noise_scale=0.2, f0_lowpass_cutoff_hz=12.0)
    assert cfg.noise_scale == 0.2, f"Expected 0.2, got {cfg.noise_scale}"
    assert cfg.f0_lowpass_cutoff_hz == 12.0, f"Expected 12.0, got {cfg.f0_lowpass_cutoff_hz}"


# ---------------------------------------------------------------------------
# InferenceConfig tests
# ---------------------------------------------------------------------------


def test_inference_config_has_noise_scale():
    """InferenceConfig should have noise_scale=0.45 by default."""
    cfg = InferenceConfig()
    assert hasattr(cfg, "noise_scale"), "InferenceConfig missing noise_scale"
    assert cfg.noise_scale == 0.45, f"Expected 0.45, got {cfg.noise_scale}"


def test_inference_config_has_f0_lowpass_cutoff():
    """InferenceConfig should have f0_lowpass_cutoff_hz=16.0 by default."""
    cfg = InferenceConfig()
    assert hasattr(cfg, "f0_lowpass_cutoff_hz"), "InferenceConfig missing f0_lowpass_cutoff_hz"
    assert cfg.f0_lowpass_cutoff_hz == 16.0, f"Expected 16.0, got {cfg.f0_lowpass_cutoff_hz}"


def test_asio_buffer_size_field():
    """asio_buffer_size should default to 0 (follow driver panel) on both configs."""
    assert AudioConfig().asio_buffer_size == 0, "AudioConfig.asio_buffer_size default"
    assert RealtimeConfig().asio_buffer_size == 0, "RealtimeConfig.asio_buffer_size default"


def test_denoise_strength_defaults_and_clamps():
    assert DenoiseConfig().strength == 1.0
    assert DenoiseConfig(strength=0.0).strength == 0.5
    assert DenoiseConfig(strength=3.0).strength == 2.0
    assert RealtimeConfig(denoise_strength=0.0).denoise_strength == 0.5
    assert RealtimeConfig(denoise_strength=3.0).denoise_strength == 2.0


def test_aggressive_latency_mode_parameters():
    balanced = _auto_params(0.24, "balanced")
    aggressive = _auto_params(0.24, "aggressive")

    assert balanced["crossfade_sec"] == 0.06
    assert balanced["buffer_margin"] == 0.5
    assert aggressive["crossfade_sec"] == 0.02
    assert _auto_params(0.1, "aggressive")["crossfade_sec"] == 0.01
    assert aggressive["buffer_margin"] == 0.25
    assert aggressive["prebuffer_chunks"] == 1
    assert _minimum_chunk_ms("swiftf0") == 40


def test_latency_mode_validation():
    assert AudioConfig(latency_mode="aggressive").latency_mode == "aggressive"
    assert AudioConfig(latency_mode="sub100").latency_mode == "sub100"
    assert AudioConfig(latency_mode="frontier").latency_mode == "frontier"
    assert AudioConfig(latency_mode="invalid").latency_mode == "balanced"
    assert RealtimeConfig(latency_mode="sub100").latency_mode == "sub100"
    assert RealtimeConfig(latency_mode="frontier").latency_mode == "frontier"
    assert RealtimeConfig(latency_mode="invalid").latency_mode == "balanced"


def test_hole_fill_and_uv_ramp_fields():
    """f0_hole_fill_ms / uv_ramp_ms should exist on both configs with matching defaults."""
    inf = InferenceConfig()
    rt = RealtimeConfig()
    for cfg, name in ((inf, "InferenceConfig"), (rt, "RealtimeConfig")):
        assert getattr(cfg, "f0_hole_fill_ms", None) == 30.0, (
            f"{name}.f0_hole_fill_ms expected 30.0, got {getattr(cfg, 'f0_hole_fill_ms', None)}"
        )
        assert getattr(cfg, "uv_ramp_ms", None) == 5.0, (
            f"{name}.uv_ramp_ms expected 5.0, got {getattr(cfg, 'uv_ramp_ms', None)}"
        )


# ---------------------------------------------------------------------------
# JSON round-trip tests
# ---------------------------------------------------------------------------


def test_config_roundtrip_new_fields():
    """New fields should survive JSON save -> load round-trip."""
    cfg = RCWXConfig()
    cfg.audio.latency_mode = "aggressive"
    cfg.inference.noise_scale = 0.3
    cfg.inference.f0_lowpass_cutoff_hz = 20.0
    cfg.inference.denoise.strength = 1.75

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        tmp_path = Path(f.name)

    try:
        cfg.save(tmp_path)
        loaded = RCWXConfig.load(tmp_path)
        assert loaded.audio.latency_mode == "aggressive"
        assert loaded.inference.noise_scale == 0.3, (
            f"noise_scale round-trip failed: {loaded.inference.noise_scale}"
        )
        assert loaded.inference.f0_lowpass_cutoff_hz == 20.0, (
            f"f0_lowpass_cutoff_hz round-trip failed: {loaded.inference.f0_lowpass_cutoff_hz}"
        )
        assert loaded.inference.denoise.strength == 1.75
    finally:
        tmp_path.unlink(missing_ok=True)


def test_config_backward_compat():
    """Loading old JSON without new fields should use defaults."""
    old_json = {
        "models_dir": "~/.cache/rcwx/models",
        "audio": {},
        "inference": {
            "pitch_shift": 5,
            "f0_method": "fcpe",
        },
    }

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json.dump(old_json, f)
        tmp_path = Path(f.name)

    try:
        loaded = RCWXConfig.load(tmp_path)
        assert loaded.inference.noise_scale == 0.45, (
            f"Expected default 0.45, got {loaded.inference.noise_scale}"
        )
        assert loaded.inference.f0_lowpass_cutoff_hz == 16.0, (
            f"Expected default 16.0, got {loaded.inference.f0_lowpass_cutoff_hz}"
        )
        assert loaded.inference.pitch_shift == 5, "Existing field should be preserved"
        assert loaded.audio.latency_mode == "balanced"
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_realtime_config_has_noise_scale,
        test_realtime_config_has_f0_lowpass_cutoff,
        test_realtime_config_custom_values,
        test_inference_config_has_noise_scale,
        test_inference_config_has_f0_lowpass_cutoff,
        test_asio_buffer_size_field,
        test_hole_fill_and_uv_ramp_fields,
        test_config_roundtrip_new_fields,
        test_config_backward_compat,
    ]
    passed = 0
    failed = 0
    for t in tests:
        name = t.__name__
        try:
            print(f"Running {name}...")
            t()
            print("  PASS")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1
    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
