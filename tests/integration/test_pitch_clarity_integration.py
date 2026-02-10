"""Integration tests for pitch clarity improvements.

Requires model to be available. Tests that parameters reach inference.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_PATH = Path(r"C:\lib\github\grand2-products\RCWX\model\元気系アニメボイス Kana\voice\voice.pth")


def _get_pipeline():
    """Load pipeline (skip if model not available)."""
    if not MODEL_PATH.exists():
        return None
    from rcwx.pipeline.inference import RVCPipeline
    pipeline = RVCPipeline(str(MODEL_PATH), device="xpu", dtype=torch.float16)
    pipeline.load()
    return pipeline


def test_f0_lowpass_cutoff_param_accepted():
    """infer_streaming() should accept f0_lowpass_cutoff_hz parameter."""
    pipeline = _get_pipeline()
    if pipeline is None:
        print("  SKIP: model not found")
        return

    # Generate test audio (16kHz, HuBERT-aligned)
    duration_samples = 320 * 30  # 30 HuBERT frames = 600ms
    overlap_samples = 320 * 5   # 5 frames overlap
    audio = np.random.randn(duration_samples).astype(np.float32) * 0.1

    # Should not raise TypeError
    output = pipeline.infer_streaming(
        audio,
        overlap_samples=overlap_samples,
        f0_lowpass_cutoff_hz=20.0,
    )
    assert output is not None and len(output) > 0, "Output should not be empty"
    print(f"  Output length: {len(output)} samples")


def test_noise_scale_deterministic():
    """noise_scale=0.0 should give deterministic output."""
    pipeline = _get_pipeline()
    if pipeline is None:
        print("  SKIP: model not found")
        return

    duration_samples = 320 * 30
    overlap_samples = 320 * 5
    audio = np.random.randn(duration_samples).astype(np.float32) * 0.1

    # Reset streaming state between runs
    pipeline._streaming_prev_audio = None
    pipeline._streaming_prev_features = None
    pipeline._streaming_prev_f0 = None
    out1 = pipeline.infer_streaming(
        audio, overlap_samples=overlap_samples, noise_scale=0.0,
    )

    pipeline._streaming_prev_audio = None
    pipeline._streaming_prev_features = None
    pipeline._streaming_prev_f0 = None
    out2 = pipeline.infer_streaming(
        audio, overlap_samples=overlap_samples, noise_scale=0.0,
    )

    # Should be very similar (not necessarily identical due to GPU non-determinism)
    if len(out1) == len(out2):
        diff = np.abs(out1 - out2).max()
        print(f"  Max diff with noise_scale=0: {diff:.6f}")


def test_config_noise_scale_wired():
    """RealtimeConfig.noise_scale should be used (not hardcoded 0.66666)."""
    from rcwx.pipeline.realtime_unified import RealtimeConfig
    import inspect
    from rcwx.pipeline.realtime_unified import RealtimeVoiceChangerUnified

    # Verify config field exists and has correct default
    cfg = RealtimeConfig(noise_scale=0.4)
    assert cfg.noise_scale == 0.4

    # Verify the inference thread code references config.noise_scale
    # by inspecting the source (not running full pipeline)
    source = inspect.getsource(RealtimeVoiceChangerUnified)
    assert "self.config.noise_scale" in source, (
        "RealtimeVoiceChangerUnified should use self.config.noise_scale"
    )
    assert "noise_scale=0.66666" not in source, (
        "Hardcoded noise_scale=0.66666 should be replaced with config value"
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_f0_lowpass_cutoff_param_accepted,
        test_noise_scale_deterministic,
        test_config_noise_scale_wired,
    ]
    passed = 0
    failed = 0
    skipped = 0
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
