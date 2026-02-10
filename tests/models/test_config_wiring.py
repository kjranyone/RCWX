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

from rcwx.config import InferenceConfig, RCWXConfig
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
    """InferenceConfig should have noise_scale=0.4 by default."""
    cfg = InferenceConfig()
    assert hasattr(cfg, "noise_scale"), "InferenceConfig missing noise_scale"
    assert cfg.noise_scale == 0.4, f"Expected 0.4, got {cfg.noise_scale}"


def test_inference_config_has_f0_lowpass_cutoff():
    """InferenceConfig should have f0_lowpass_cutoff_hz=16.0 by default."""
    cfg = InferenceConfig()
    assert hasattr(cfg, "f0_lowpass_cutoff_hz"), "InferenceConfig missing f0_lowpass_cutoff_hz"
    assert cfg.f0_lowpass_cutoff_hz == 16.0, f"Expected 16.0, got {cfg.f0_lowpass_cutoff_hz}"


# ---------------------------------------------------------------------------
# JSON round-trip tests
# ---------------------------------------------------------------------------

def test_config_roundtrip_new_fields():
    """New fields should survive JSON save -> load round-trip."""
    cfg = RCWXConfig()
    cfg.inference.noise_scale = 0.3
    cfg.inference.f0_lowpass_cutoff_hz = 20.0

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        tmp_path = Path(f.name)

    try:
        cfg.save(tmp_path)
        loaded = RCWXConfig.load(tmp_path)
        assert loaded.inference.noise_scale == 0.3, (
            f"noise_scale round-trip failed: {loaded.inference.noise_scale}"
        )
        assert loaded.inference.f0_lowpass_cutoff_hz == 20.0, (
            f"f0_lowpass_cutoff_hz round-trip failed: {loaded.inference.f0_lowpass_cutoff_hz}"
        )
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
        assert loaded.inference.noise_scale == 0.4, (
            f"Expected default 0.4, got {loaded.inference.noise_scale}"
        )
        assert loaded.inference.f0_lowpass_cutoff_hz == 16.0, (
            f"Expected default 16.0, got {loaded.inference.f0_lowpass_cutoff_hz}"
        )
        assert loaded.inference.pitch_shift == 5, "Existing field should be preserved"
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
            print(f"  PASS")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
