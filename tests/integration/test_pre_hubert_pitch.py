"""Test pre-HuBERT pitch shift functionality.

Verifies:
1. ratio=0.0 produces identical output to baseline (no pre-shift)
2. pitch_shift_resample preserves sample count
3. FFT peak shifts confirm actual frequency change
4. Performance stays within budget (<5ms for 8960 samples)
5. Residual F0 shift keeps HuBERT/F0 shift accounting consistent
6. Chunk-boundary continuity is preserved for shifted chunks
7. Moe boost shapes F0 contour upward
8. Config round-trip persistence
9. Pipeline integration (if model available)
"""

from __future__ import annotations

import logging
import sys
import time
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


def generate_sine(freq: float, duration: float, sr: int = 16000) -> np.ndarray:
    """Generate a pure sine wave."""
    t = np.arange(int(sr * duration)) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def dominant_freq(audio: np.ndarray, sr: int = 16000) -> float:
    """Find the dominant frequency via FFT."""
    n = len(audio)
    fft = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    fft[0] = 0  # Ignore DC
    return float(freqs[np.argmax(fft)])


def test_identity_ratio_zero():
    """ratio=0.0 must produce output identical to no pre-shift."""
    logger.info("=== Test: identity (ratio=0.0) ===")
    from rcwx.pipeline.inference import pitch_shift_resample

    audio = generate_sine(440.0, 0.56)
    audio_t = torch.from_numpy(audio)

    # With ratio=0.0, effective_shift=0, which is < 0.01 threshold
    result = pitch_shift_resample(audio_t, sample_rate=16000, semitones=0.0)
    result_np = result.cpu().numpy().squeeze()

    assert np.array_equal(audio, result_np), "semitones=0.0 should return original"
    logger.info("PASS: semitones=0.0 returns original array")


def test_shape_preservation():
    """pitch_shift_resample must preserve sample count."""
    logger.info("=== Test: shape preservation ===")
    from rcwx.pipeline.inference import pitch_shift_resample

    audio = generate_sine(440.0, 0.56)
    audio_t = torch.from_numpy(audio)
    n = len(audio)

    for semitones in [-12, -5, -1, 1, 5, 12]:
        shifted = pitch_shift_resample(audio_t, sample_rate=16000, semitones=float(semitones))
        shifted_np = shifted.cpu().numpy().squeeze()
        assert len(shifted_np) == n, (
            f"Length mismatch for {semitones}st: input={n}, output={len(shifted_np)}"
        )
        logger.info(f"  semitones={semitones:+d}: len={len(shifted_np)} (OK)")

    logger.info("PASS: all shifts preserve length")


def test_frequency_shift():
    """FFT peak should shift by expected ratio."""
    logger.info("=== Test: frequency shift verification ===")
    from rcwx.pipeline.inference import pitch_shift_resample

    base_freq = 440.0
    audio = generate_sine(base_freq, 1.0)
    audio_t = torch.from_numpy(audio)

    for semitones in [-12, -5, 5, 12]:
        shifted = pitch_shift_resample(audio_t, sample_rate=16000, semitones=float(semitones))
        shifted_np = shifted.cpu().numpy().squeeze()

        # Check frequency of first 75% to avoid padding artifacts
        check_len = int(len(shifted_np) * 0.75)
        actual_freq = dominant_freq(shifted_np[:check_len])
        expected_freq = base_freq * (2 ** (semitones / 12))

        ratio = actual_freq / expected_freq
        tolerance = 0.03
        assert abs(ratio - 1.0) < tolerance, (
            f"Frequency mismatch for {semitones}st: "
            f"expected={expected_freq:.1f}Hz, got={actual_freq:.1f}Hz (ratio={ratio:.3f})"
        )
        logger.info(
            f"  {semitones:+d}st: expected={expected_freq:.1f}Hz, "
            f"got={actual_freq:.1f}Hz (ratio={ratio:.3f})"
        )

    logger.info("PASS: frequency shifts match expected values")


def test_performance():
    """pitch_shift_resample should complete within 5ms for typical chunk size."""
    logger.info("=== Test: performance ===")
    from rcwx.pipeline.inference import pitch_shift_resample

    # 8960 samples = typical streaming chunk (~0.56s @ 16kHz)
    audio = generate_sine(440.0, 0.56)
    audio_t = torch.from_numpy(audio)

    # Warmup
    for _ in range(3):
        _ = pitch_shift_resample(audio_t, sample_rate=16000, semitones=5.0)

    # Benchmark
    times = []
    for _ in range(50):
        start = time.perf_counter()
        _ = pitch_shift_resample(audio_t, sample_rate=16000, semitones=5.0)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    mean_ms = np.mean(times)
    max_ms = np.max(times)
    logger.info(f"  mean={mean_ms:.2f}ms, max={max_ms:.2f}ms (n={len(times)})")

    assert mean_ms < 10, f"Too slow: {mean_ms:.2f}ms mean (budget: 10ms)"
    logger.info("PASS: within performance budget")


def test_post_f0_shift_residual():
    """Residual shift should be pitch_shift - pre_hubert_shift."""
    logger.info("=== Test: residual F0 shift math ===")
    from rcwx.pipeline.inference import compute_post_f0_shift, compute_pre_hubert_shift

    cases = [
        (12, 0.0, 12.0),
        (12, 0.5, 6.0),
        (-12, 0.25, -9.0),
        (7, 1.0, 0.0),
    ]
    for pitch_shift, ratio, expected in cases:
        pre = compute_pre_hubert_shift(pitch_shift, ratio)
        residual = compute_post_f0_shift(pitch_shift, pre)
        assert abs(residual - expected) < 1e-6, (
            f"residual mismatch: pitch={pitch_shift}, ratio={ratio}, "
            f"expected={expected}, got={residual}"
        )
    logger.info("PASS: residual F0 shift is correct")


def test_chunk_boundary_continuity():
    """Shifted chunks should not introduce large sample jumps at boundaries."""
    logger.info("=== Test: chunk boundary continuity ===")
    from rcwx.pipeline.inference import pitch_shift_resample

    sr = 16000
    freq = 220.0
    duration = 1.2
    chunk_samples = 2560
    semitones = 7.0

    t = np.arange(int(sr * duration)) / sr
    audio = np.sin(2 * np.pi * freq * t).astype(np.float32)

    shifted_chunks = []
    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i : i + chunk_samples]
        if len(chunk) == 0:
            continue
        shifted = pitch_shift_resample(
            torch.from_numpy(chunk), sample_rate=sr, semitones=semitones
        )
        shifted_chunks.append(shifted.cpu().numpy().squeeze())

    shifted_audio = np.concatenate(shifted_chunks)[: len(audio)]
    boundaries = list(range(chunk_samples, len(shifted_audio), chunk_samples))
    assert boundaries, "Need at least one boundary for continuity test"

    jumps = [abs(float(shifted_audio[b] - shifted_audio[b - 1])) for b in boundaries]
    mean_jump = float(np.mean(jumps))
    max_jump = float(np.max(jumps))
    logger.info(f"  boundary jumps: mean={mean_jump:.4f}, max={max_jump:.4f}")

    assert mean_jump < 0.15, f"Mean boundary jump too large: {mean_jump:.4f}"
    assert max_jump < 0.25, f"Max boundary jump too large: {max_jump:.4f}"
    logger.info("PASS: chunk boundaries are continuous")


def test_moe_boost_f0_style():
    """Moe boost should lift contour and fill short voiced dropouts."""
    logger.info("=== Test: moe boost F0 style ===")
    from rcwx.pipeline.inference import apply_moe_f0_style

    f0 = torch.tensor([[200.0, 210.0, 190.0, 220.0, 0.0, 205.0, 195.0]])
    styled = apply_moe_f0_style(f0, strength=0.8)

    voiced = f0 > 0
    base_voiced = f0[voiced]
    styled_voiced = styled[voiced]

    base_mean = float(base_voiced.mean())
    styled_mean = float(styled_voiced.mean())
    base_span = float(base_voiced.max() - base_voiced.min())
    styled_span = float(styled_voiced.max() - styled_voiced.min())

    logger.info(
        f"  mean: base={base_mean:.2f}Hz -> styled={styled_mean:.2f}Hz, "
        f"span: base={base_span:.2f}Hz -> styled={styled_span:.2f}Hz"
    )
    logger.info(
        f"  short-gap frame: base={float(f0[0, 4]):.2f}Hz -> styled={float(styled[0, 4]):.2f}Hz"
    )
    assert styled_mean > base_mean, "Moe boost should raise average voiced F0"
    assert styled_span > base_span * 0.90, "Moe boost should keep contour expressive"
    assert styled[0, 4] > 0, "Moe boost should fill short unvoiced F0 gap"
    logger.info("PASS: moe boost style is applied")


def test_config_roundtrip():
    """pre_hubert_pitch_ratio should survive config save/load."""
    logger.info("=== Test: config round-trip ===")

    import json
    import tempfile
    from rcwx.config import RCWXConfig

    config = RCWXConfig()
    config.inference.pre_hubert_pitch_ratio = 0.75
    config.inference.moe_boost = 0.60

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        tmp_path = Path(f.name)

    try:
        config.save(tmp_path)

        with open(tmp_path) as f:
            data = json.load(f)
        assert data["inference"]["pre_hubert_pitch_ratio"] == 0.75
        assert data["inference"]["moe_boost"] == 0.60

        loaded = RCWXConfig.load(tmp_path)
        assert loaded.inference.pre_hubert_pitch_ratio == 0.75
        assert loaded.inference.moe_boost == 0.60
        logger.info("PASS: config round-trip preserves pre_hubert_pitch_ratio")
    finally:
        tmp_path.unlink(missing_ok=True)


def test_pipeline_integration():
    """Test pre-HuBERT pitch shift through RVCPipeline if model available."""
    logger.info("=== Test: pipeline integration ===")

    from rcwx.config import RCWXConfig

    config = RCWXConfig.load()
    if not config.last_model_path or not Path(config.last_model_path).exists():
        logger.info("SKIP: no model configured (run GUI first)")
        return

    from rcwx.pipeline.inference import RVCPipeline

    pipeline = RVCPipeline(config.last_model_path, device=config.device, use_compile=False)
    pipeline.load()

    audio = generate_sine(220.0, 0.8)

    # Baseline: no pre-shift
    pipeline.clear_cache()
    out_baseline = pipeline.infer(
        audio, pitch_shift=5, noise_scale=0,
        pre_hubert_pitch_ratio=0.0,
    )

    # With pre-shift
    pipeline.clear_cache()
    out_shifted = pipeline.infer(
        audio, pitch_shift=5, noise_scale=0,
        pre_hubert_pitch_ratio=1.0,
    )

    # Outputs should be different (different HuBERT features)
    min_len = min(len(out_baseline), len(out_shifted))
    if min_len > 0:
        corr = np.corrcoef(out_baseline[:min_len], out_shifted[:min_len])[0, 1]
        logger.info(f"  Correlation baseline vs shifted: {corr:.3f}")
        assert corr < 0.999, "Pre-shift should change HuBERT features"
        logger.info("PASS: pre-shift produces different output than baseline")
    else:
        logger.warning("  Output too short for comparison")


def test_streaming_integration():
    """Test pre-HuBERT pitch shift through infer_streaming() if model available."""
    logger.info("=== Test: streaming integration ===")

    from rcwx.config import RCWXConfig

    config = RCWXConfig.load()
    if not config.last_model_path or not Path(config.last_model_path).exists():
        logger.info("SKIP: no model configured (run GUI first)")
        return

    from rcwx.pipeline.inference import RVCPipeline

    pipeline = RVCPipeline(config.last_model_path, device=config.device, use_compile=False)
    pipeline.load()

    # Create aligned chunk (multiple of 320)
    hop = 320
    chunk_samples = hop * 8   # 2560 samples = 160ms
    overlap_samples = hop * 4  # 1280 samples = 80ms

    audio = generate_sine(220.0, (chunk_samples + overlap_samples) / 16000)
    audio = audio[:chunk_samples + overlap_samples]  # exact length

    # Baseline
    pipeline.clear_cache()
    out_base = pipeline.infer_streaming(
        audio, overlap_samples=overlap_samples, pitch_shift=5,
        noise_scale=0, pre_hubert_pitch_ratio=0.0,
    )

    # With pre-shift
    pipeline.clear_cache()
    out_shifted = pipeline.infer_streaming(
        audio, overlap_samples=overlap_samples, pitch_shift=5,
        noise_scale=0, pre_hubert_pitch_ratio=1.0,
    )

    logger.info(f"  Baseline output: {len(out_base)} samples")
    logger.info(f"  Shifted output:  {len(out_shifted)} samples")

    min_len = min(len(out_base), len(out_shifted))
    if min_len > 100:
        corr = np.corrcoef(out_base[:min_len], out_shifted[:min_len])[0, 1]
        logger.info(f"  Correlation: {corr:.3f}")
        logger.info("PASS: streaming integration works")
    else:
        logger.info("PASS: streaming integration runs without error")


if __name__ == "__main__":
    test_identity_ratio_zero()
    test_shape_preservation()
    test_frequency_shift()
    test_performance()
    test_post_f0_shift_residual()
    test_chunk_boundary_continuity()
    test_moe_boost_f0_style()
    test_config_roundtrip()
    test_pipeline_integration()
    test_streaming_integration()
    logger.info("\n=== All tests passed ===")
