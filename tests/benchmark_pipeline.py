"""
Pipeline Benchmark - 各処理ステージの時間計測とボトルネック分析

Usage:
    uv run python tests/benchmark_pipeline.py
    uv run python tests/benchmark_pipeline.py --iterations 20
    uv run python tests/benchmark_pipeline.py --chunk-sec 0.15 0.20 0.35
"""
from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from rcwx.audio.resample import resample, StatefulResampler
from rcwx.audio.crossfade import SOLAState, apply_sola_crossfade
from rcwx.config import RCWXConfig
from rcwx.pipeline.inference import RVCPipeline

logging.basicConfig(
    level=logging.WARNING,  # Suppress verbose logs during benchmark
    format="%(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class StageTimings:
    """Timings for a single stage."""
    name: str
    times_ms: list[float] = field(default_factory=list)

    @property
    def mean_ms(self) -> float:
        return np.mean(self.times_ms) if self.times_ms else 0.0

    @property
    def std_ms(self) -> float:
        return np.std(self.times_ms) if self.times_ms else 0.0

    @property
    def min_ms(self) -> float:
        return np.min(self.times_ms) if self.times_ms else 0.0

    @property
    def max_ms(self) -> float:
        return np.max(self.times_ms) if self.times_ms else 0.0


@dataclass
class BenchmarkResult:
    """Complete benchmark result."""
    chunk_sec: float
    iterations: int
    stages: dict[str, StageTimings] = field(default_factory=dict)
    total_times_ms: list[float] = field(default_factory=list)

    @property
    def total_mean_ms(self) -> float:
        return np.mean(self.total_times_ms) if self.total_times_ms else 0.0

    @property
    def realtime_ratio(self) -> float:
        """Ratio of processing time to audio duration. <1.0 = realtime capable."""
        chunk_ms = self.chunk_sec * 1000
        return self.total_mean_ms / chunk_ms if chunk_ms > 0 else 0.0

    def add_timing(self, stage: str, time_ms: float):
        if stage not in self.stages:
            self.stages[stage] = StageTimings(name=stage)
        self.stages[stage].times_ms.append(time_ms)


def generate_test_audio(duration_sec: float, sample_rate: int = 48000) -> np.ndarray:
    """Generate test audio with voice-like characteristics."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), dtype=np.float32)
    # Simulate voice with fundamental + harmonics + noise
    f0 = 150  # Hz
    audio = (
        0.4 * np.sin(2 * np.pi * f0 * t) +
        0.2 * np.sin(2 * np.pi * f0 * 2 * t) +
        0.1 * np.sin(2 * np.pi * f0 * 3 * t) +
        0.05 * np.random.randn(len(t))
    )
    return audio.astype(np.float32)


def benchmark_individual_stages(
    pipeline: RVCPipeline,
    chunk_sec: float,
    iterations: int = 10,
    f0_method: str = "rmvpe",
) -> BenchmarkResult:
    """Benchmark each pipeline stage individually."""
    result = BenchmarkResult(chunk_sec=chunk_sec, iterations=iterations)

    mic_sr = 48000
    process_sr = 16000
    output_sr = 48000

    # Warmup
    print(f"  Warming up ({chunk_sec}s chunks)...")
    warmup_audio = generate_test_audio(chunk_sec, mic_sr)
    warmup_16k = resample(warmup_audio, mic_sr, process_sr)
    pipeline.clear_cache()
    try:
        _ = pipeline.infer(warmup_16k, input_sr=process_sr, f0_method=f0_method, allow_short_input=True)
    except Exception as e:
        print(f"  Warmup failed: {e}")
        return result

    # Prepare test data
    test_chunks_mic = [generate_test_audio(chunk_sec, mic_sr) for _ in range(iterations)]

    # Stage 1: Input Resampling (48kHz -> 16kHz)
    print("  Benchmarking: Input Resampling...")
    resampler_in = StatefulResampler(mic_sr, process_sr)
    for chunk in test_chunks_mic:
        gc.collect()
        t0 = time.perf_counter()
        _ = resampler_in.resample_chunk(chunk)
        result.add_timing("1_resample_in", (time.perf_counter() - t0) * 1000)

    # Prepare 16kHz chunks for next stages
    test_chunks_16k = [resample(c, mic_sr, process_sr) for c in test_chunks_mic]

    # Stage 2: F0 extraction (measured separately as it's a known bottleneck)
    print(f"  Benchmarking: F0 extraction ({f0_method})...")
    for chunk in test_chunks_16k:
        gc.collect()
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            torch.xpu.synchronize()
        t0 = time.perf_counter()
        chunk_tensor = torch.from_numpy(chunk).to(pipeline.device)
        if f0_method == "rmvpe" and pipeline.rmvpe is not None:
            _ = pipeline.rmvpe.infer(chunk_tensor)
        elif f0_method == "fcpe" and pipeline.fcpe is not None:
            _ = pipeline.fcpe.infer(chunk_tensor)
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            torch.xpu.synchronize()
        result.add_timing("2_f0", (time.perf_counter() - t0) * 1000)

    # Stage 5: Output Resampling (model_sr -> 48kHz)
    print("  Benchmarking: Output Resampling...")
    model_sr = pipeline.sample_rate
    test_output = np.random.randn(int(model_sr * chunk_sec)).astype(np.float32) * 0.5
    resampler_out = StatefulResampler(model_sr, output_sr)
    for _ in range(iterations):
        gc.collect()
        t0 = time.perf_counter()
        _ = resampler_out.resample_chunk(test_output)
        result.add_timing("5_resample_out", (time.perf_counter() - t0) * 1000)

    # Stage 6: SOLA Crossfade
    print("  Benchmarking: SOLA...")
    crossfade_samples = int(output_sr * 0.05)
    context_samples = int(output_sr * 0.10)
    sola_state = SOLAState.create(crossfade_samples, output_sr)
    test_sola_input = np.random.randn(int(output_sr * chunk_sec)).astype(np.float32) * 0.5
    for _ in range(iterations):
        gc.collect()
        t0 = time.perf_counter()
        _ = apply_sola_crossfade(test_sola_input, sola_state, wokada_mode=True, context_samples=context_samples)
        result.add_timing("6_sola", (time.perf_counter() - t0) * 1000)

    # Full Pipeline (end-to-end)
    print("  Benchmarking: Full Pipeline...")
    pipeline.clear_cache()
    for chunk in test_chunks_16k:
        gc.collect()
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            torch.xpu.synchronize()
        t0 = time.perf_counter()
        try:
            _ = pipeline.infer(
                chunk, input_sr=process_sr, f0_method=f0_method,
                use_feature_cache=True, use_parallel_extraction=True, allow_short_input=True
            )
        except Exception:
            pass
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            torch.xpu.synchronize()
        result.total_times_ms.append((time.perf_counter() - t0) * 1000)

    # Estimate HuBERT+Synth time from full pipeline - f0
    f0_avg = result.stages["2_f0"].mean_ms if "2_f0" in result.stages else 0
    total_avg = np.mean(result.total_times_ms) if result.total_times_ms else 0
    hubert_synth_est = max(0, total_avg - f0_avg)
    for _ in range(iterations):
        result.add_timing("3_hubert_synth", hubert_synth_est)

    return result


def print_results(results: list[BenchmarkResult], f0_method: str):
    """Print benchmark results in a table."""
    print("\n" + "=" * 80)
    print(f"BENCHMARK RESULTS (F0: {f0_method})")
    print("=" * 80)

    # Header
    print(f"\n{'Stage':<20} ", end="")
    for r in results:
        print(f"{r.chunk_sec*1000:.0f}ms chunk  ", end="")
    print()
    print("-" * 80)

    # Get all stage names
    all_stages = set()
    for r in results:
        all_stages.update(r.stages.keys())
    stage_names = sorted(all_stages)

    # Print each stage
    for stage in stage_names:
        print(f"{stage:<20} ", end="")
        for r in results:
            if stage in r.stages:
                s = r.stages[stage]
                print(f"{s.mean_ms:6.1f}±{s.std_ms:4.1f}ms  ", end="")
            else:
                print(f"{'N/A':>14}  ", end="")
        print()

    print("-" * 80)

    # Total pipeline time
    print(f"{'TOTAL PIPELINE':<20} ", end="")
    for r in results:
        mean = np.mean(r.total_times_ms) if r.total_times_ms else 0
        std = np.std(r.total_times_ms) if r.total_times_ms else 0
        print(f"{mean:6.1f}±{std:4.1f}ms  ", end="")
    print()

    # Realtime ratio
    print(f"{'REALTIME RATIO':<20} ", end="")
    for r in results:
        ratio = r.realtime_ratio
        status = "OK" if ratio < 1.0 else "SLOW"
        print(f"{ratio:6.2f}x ({status:4})  ", end="")
    print()

    # Budget analysis
    print("\n" + "=" * 80)
    print("BUDGET ANALYSIS")
    print("=" * 80)

    for r in results:
        chunk_ms = r.chunk_sec * 1000
        print(f"\nChunk: {chunk_ms:.0f}ms (budget: {chunk_ms:.0f}ms for realtime)")

        total_stage_time = sum(s.mean_ms for s in r.stages.values())
        overhead = r.total_mean_ms - total_stage_time if r.total_times_ms else 0

        # Sort by time
        sorted_stages = sorted(r.stages.items(), key=lambda x: x[1].mean_ms, reverse=True)

        for name, s in sorted_stages:
            pct = (s.mean_ms / chunk_ms) * 100
            bar = "#" * int(pct / 5) + "." * (20 - int(pct / 5))
            status = "[!]" if pct > 30 else "   "
            print(f"  {status}{name:<18} {s.mean_ms:6.1f}ms ({pct:5.1f}%) {bar}")

        if overhead > 0:
            pct = (overhead / chunk_ms) * 100
            print(f"    {'(overhead)':<18} {overhead:6.1f}ms ({pct:5.1f}%)")

        print(f"    {'TOTAL':<18} {r.total_mean_ms:6.1f}ms ({r.realtime_ratio*100:5.1f}%)")

        if r.realtime_ratio >= 1.0:
            excess = r.total_mean_ms - chunk_ms
            print(f"    [X] Exceeds budget by {excess:.1f}ms")
        else:
            margin = chunk_ms - r.total_mean_ms
            print(f"    [OK] Under budget by {margin:.1f}ms")


def analyze_bottlenecks(results: list[BenchmarkResult]) -> dict:
    """Analyze and identify bottlenecks."""
    analysis = {
        "primary_bottleneck": None,
        "recommendations": [],
    }

    # Find the slowest stage across all chunk sizes
    stage_totals = {}
    for r in results:
        for name, s in r.stages.items():
            if name not in stage_totals:
                stage_totals[name] = []
            stage_totals[name].append(s.mean_ms)

    # Average across chunk sizes
    stage_avgs = {name: np.mean(times) for name, times in stage_totals.items()}
    sorted_stages = sorted(stage_avgs.items(), key=lambda x: x[1], reverse=True)

    if sorted_stages:
        analysis["primary_bottleneck"] = sorted_stages[0][0]

        # Generate recommendations
        bottleneck = sorted_stages[0][0]
        if "f0" in bottleneck.lower():
            analysis["recommendations"].append(
                "F0 extraction is the primary bottleneck. "
                "Consider: (1) Use FCPE instead of RMVPE for lower latency, "
                "(2) Increase chunk size to amortize fixed costs, "
                "(3) Cache F0 values for similar audio segments."
            )
        elif "hubert" in bottleneck.lower():
            analysis["recommendations"].append(
                "HuBERT feature extraction is slow. "
                "Consider: (1) Use smaller HuBERT model, "
                "(2) Reduce feature extraction frequency, "
                "(3) Implement feature caching more aggressively."
            )
        elif "synth" in bottleneck.lower():
            analysis["recommendations"].append(
                "Synthesizer is the bottleneck. "
                "Consider: (1) Use torch.compile for optimization, "
                "(2) Reduce model complexity, "
                "(3) Batch multiple frames."
            )

    return analysis


def main():
    parser = argparse.ArgumentParser(description="Pipeline Benchmark")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations per test")
    parser.add_argument("--chunk-sec", type=float, nargs="+", default=[0.15, 0.20, 0.35],
                        help="Chunk sizes to test")

    args = parser.parse_args()

    # Load config and pipeline
    config = RCWXConfig.load()
    if not config.last_model_path:
        print("ERROR: No model configured. Run GUI first.")
        sys.exit(1)

    print(f"Loading model: {config.last_model_path}")
    print(f"Device: {config.device}")

    pipeline = RVCPipeline(config.last_model_path, device=config.device, use_compile=False)
    pipeline.load()

    # Detect F0 method
    try:
        import torchfcpe
        f0_method = "fcpe"
        print("F0 method: FCPE (low latency)")
    except ImportError:
        f0_method = "rmvpe"
        print("F0 method: RMVPE (higher latency)")

    # Filter chunk sizes for RMVPE compatibility
    if f0_method == "rmvpe":
        valid_chunks = [c for c in args.chunk_sec if c >= 0.32]
        if not valid_chunks:
            valid_chunks = [0.35]
        if len(valid_chunks) < len(args.chunk_sec):
            print(f"Note: RMVPE requires chunk >= 0.32s, using {valid_chunks}")
        args.chunk_sec = valid_chunks

    print(f"\nBenchmarking {args.iterations} iterations for each chunk size: {args.chunk_sec}")
    print("=" * 80)

    results = []
    for chunk_sec in args.chunk_sec:
        print(f"\n--- Chunk: {chunk_sec*1000:.0f}ms ---")
        result = benchmark_individual_stages(
            pipeline, chunk_sec, iterations=args.iterations, f0_method=f0_method
        )
        results.append(result)

    # Print results
    print_results(results, f0_method)

    # Analyze bottlenecks
    print("\n" + "=" * 80)
    print("BOTTLENECK ANALYSIS")
    print("=" * 80)

    analysis = analyze_bottlenecks(results)
    print(f"\nPrimary Bottleneck: {analysis['primary_bottleneck']}")
    print("\nRecommendations:")
    for i, rec in enumerate(analysis['recommendations'], 1):
        print(f"  {i}. {rec}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    realtime_capable = [r for r in results if r.realtime_ratio < 1.0]
    if realtime_capable:
        best = min(realtime_capable, key=lambda r: r.chunk_sec)
        print(f"\n[OK] Realtime capable at {best.chunk_sec*1000:.0f}ms chunks (ratio: {best.realtime_ratio:.2f}x)")
    else:
        print(f"\n[X] NOT realtime capable at any tested chunk size")
        closest = min(results, key=lambda r: r.realtime_ratio)
        print(f"   Closest: {closest.chunk_sec*1000:.0f}ms chunks at {closest.realtime_ratio:.2f}x realtime")
        print(f"   Need {closest.realtime_ratio:.1f}x speedup or {closest.chunk_sec * closest.realtime_ratio * 1000:.0f}ms chunks")


if __name__ == "__main__":
    main()
