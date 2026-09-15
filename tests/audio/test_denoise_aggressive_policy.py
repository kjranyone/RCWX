"""Aggressive-mode denoise policy.

The Aggressive hop budget leaves no room for per-hop ML denoiser calls,
and GTCRN (learned, streaming, CPU ~2ms/hop) is the highest-quality
denoiser that fits that budget.  Two guarantees:

1. In Aggressive mode the denoise method is remapped to GTCRN
   (construction, ``set_denoise``, ``set_latency_mode``).
2. As a safety net for stateless callers, ``denoise(method="spectral")``
   passes audio shorter than one analysis window through unchanged instead
   of zeroing it (the realtime path uses the streaming gate instead).
"""

from __future__ import annotations

import numpy as np

import rcwx.pipeline.realtime_unified as ru
from rcwx.audio.denoise import denoise
from rcwx.pipeline.realtime_config import RealtimeConfig


class _FakePipeline:
    _loaded = True
    device = "cpu"
    sample_rate = 16000
    synthesizer = None
    accelerator_index = None
    stage_times: dict = {}


def test_aggressive_config_forces_gtcrn() -> None:
    for method in ("auto", "ml", "spectral"):
        cfg = RealtimeConfig(
            latency_mode="aggressive",
            denoise_enabled=True,
            denoise_method=method,
        )
        assert cfg.denoise_method == "gtcrn", method


def test_normal_config_keeps_method() -> None:
    for method in ("auto", "ml", "spectral"):
        cfg = RealtimeConfig(
            latency_mode="normal",
            denoise_enabled=True,
            denoise_method=method,
        )
        assert cfg.denoise_method == method


def test_unified_set_denoise_forces_gtcrn_in_aggressive() -> None:
    cfg = RealtimeConfig(
        mic_sample_rate=16000,
        output_sample_rate=16000,
        chunk_sec=0.02,
        latency_mode="aggressive",
        denoise_enabled=True,
        denoise_method="spectral",
        use_sola=False,
    )
    changer = ru.RealtimeVoiceChangerUnified(_FakePipeline(), cfg)
    assert changer.config.denoise_method == "gtcrn"
    changer.set_denoise(True, "ml", 1.0)
    assert changer.config.denoise_method == "gtcrn"


def test_unified_set_latency_mode_forces_gtcrn() -> None:
    cfg = RealtimeConfig(
        mic_sample_rate=16000,
        output_sample_rate=16000,
        chunk_sec=0.04,
        latency_mode="normal",
        denoise_enabled=True,
        denoise_method="spectral",
        use_sola=False,
    )
    changer = ru.RealtimeVoiceChangerUnified(_FakePipeline(), cfg)
    assert changer.config.denoise_method == "spectral"
    changer.set_latency_mode("aggressive")
    assert changer.config.denoise_method == "gtcrn"


def test_spectral_short_hop_passthrough() -> None:
    # 20ms @16k = 320 samples, far below the 2048-sample analysis window.
    rng = np.random.RandomState(0)
    hop = (rng.randn(320) * 0.05).astype(np.float32)
    out = denoise(hop, sample_rate=16000, method="spectral")
    assert out.shape == hop.shape
    assert float(np.max(np.abs(out))) > 0.0, "short hop must not be zeroed"
    assert np.allclose(out, hop), "short hop must pass through unchanged"


if __name__ == "__main__":
    test_aggressive_config_forces_gtcrn()
    test_normal_config_keeps_method()
    test_unified_set_denoise_forces_gtcrn_in_aggressive()
    test_unified_set_latency_mode_forces_gtcrn()
    test_spectral_short_hop_passthrough()
    print("OK: all denoise aggressive-policy tests passed")
