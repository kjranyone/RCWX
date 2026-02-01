"""
Benchmark to test parallel HuBERT + F0 extraction.

Tests three approaches:
1. Sequential (current implementation)
2. ThreadPoolExecutor (Python threads)
3. CUDA/XPU streams (GPU parallel)
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from rcwx.config import RCWXConfig
from rcwx.pipeline.inference import RVCPipeline


def benchmark_sequential(pipeline: RVCPipeline, audio: torch.Tensor, iterations: int = 10):
    """Current sequential implementation."""
    times = []

    for _ in range(iterations):
        start = time.perf_counter()

        # HuBERT extraction
        with torch.autocast(device_type=pipeline.device, dtype=pipeline.dtype):
            features = pipeline.hubert.extract(audio, output_layer=12, output_dim=768)

        # F0 extraction
        if pipeline.fcpe is not None:
            with torch.autocast(device_type=pipeline.device, dtype=pipeline.dtype):
                f0 = pipeline.fcpe.infer(audio, threshold=0.006)
        elif pipeline.rmvpe is not None:
            with torch.autocast(device_type=pipeline.device, dtype=pipeline.dtype):
                f0 = pipeline.rmvpe.infer(audio)

        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    return np.mean(times), np.std(times)


def benchmark_threadpool(pipeline: RVCPipeline, audio: torch.Tensor, iterations: int = 10):
    """Parallel execution with ThreadPoolExecutor."""
    times = []

    def extract_hubert():
        with torch.autocast(device_type=pipeline.device, dtype=pipeline.dtype):
            return pipeline.hubert.extract(audio, output_layer=12, output_dim=768)

    def extract_f0():
        if pipeline.fcpe is not None:
            with torch.autocast(device_type=pipeline.device, dtype=pipeline.dtype):
                return pipeline.fcpe.infer(audio, threshold=0.006)
        elif pipeline.rmvpe is not None:
            with torch.autocast(device_type=pipeline.device, dtype=pipeline.dtype):
                return pipeline.rmvpe.infer(audio)
        return None

    for _ in range(iterations):
        start = time.perf_counter()

        with ThreadPoolExecutor(max_workers=2) as executor:
            hubert_future = executor.submit(extract_hubert)
            f0_future = executor.submit(extract_f0)

            features = hubert_future.result()
            f0 = f0_future.result()

        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    return np.mean(times), np.std(times)


def benchmark_streams(pipeline: RVCPipeline, audio: torch.Tensor, iterations: int = 10):
    """Parallel execution with CUDA/XPU streams."""
    times = []

    # Create streams for parallel execution
    if pipeline.device == "cuda":
        stream_hubert = torch.cuda.Stream()
        stream_f0 = torch.cuda.Stream()
    elif pipeline.device == "xpu":
        # XPU streams
        stream_hubert = torch.xpu.Stream()
        stream_f0 = torch.xpu.Stream()
    else:
        # CPU - streams not applicable
        return None, None

    for _ in range(iterations):
        start = time.perf_counter()

        # Run HuBERT and F0 in parallel streams
        if pipeline.device == "cuda":
            with torch.cuda.stream(stream_hubert):
                with torch.autocast(device_type=pipeline.device, dtype=pipeline.dtype):
                    features = pipeline.hubert.extract(audio, output_layer=12, output_dim=768)

            with torch.cuda.stream(stream_f0):
                if pipeline.fcpe is not None:
                    with torch.autocast(device_type=pipeline.device, dtype=pipeline.dtype):
                        f0 = pipeline.fcpe.infer(audio, threshold=0.006)
                elif pipeline.rmvpe is not None:
                    with torch.autocast(device_type=pipeline.device, dtype=pipeline.dtype):
                        f0 = pipeline.rmvpe.infer(audio)

            # Wait for both streams to complete
            torch.cuda.synchronize()

        elif pipeline.device == "xpu":
            with torch.xpu.stream(stream_hubert):
                with torch.autocast(device_type=pipeline.device, dtype=pipeline.dtype):
                    features = pipeline.hubert.extract(audio, output_layer=12, output_dim=768)

            with torch.xpu.stream(stream_f0):
                if pipeline.fcpe is not None:
                    with torch.autocast(device_type=pipeline.device, dtype=pipeline.dtype):
                        f0 = pipeline.fcpe.infer(audio, threshold=0.006)
                elif pipeline.rmvpe is not None:
                    with torch.autocast(device_type=pipeline.device, dtype=pipeline.dtype):
                        f0 = pipeline.rmvpe.infer(audio)

            # Wait for both streams to complete
            torch.xpu.synchronize()

        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    return np.mean(times), np.std(times)


def main():
    """Run benchmarks."""
    print("=" * 80)
    print("Parallel Inference Benchmark")
    print("=" * 80)
    print()

    # Load config and pipeline
    config = RCWXConfig.load()

    if config.last_model_path is None:
        print("No model configured. Please run the GUI first.")
        return

    print(f"Model: {config.last_model_path}")
    print(f"Device: {config.device}")
    print(f"F0 method: {config.inference.f0_method}")
    print()

    # Initialize pipeline
    pipeline = RVCPipeline(
        model_path=config.last_model_path,
        device=config.device,
        dtype=config.dtype,
        use_f0=config.inference.use_f0,
        use_compile=False,  # Disable compile for fair comparison
        models_dir=config.models_dir,
    )
    pipeline.load()

    # Create test audio (0.2 seconds @ 16kHz)
    audio_length = int(16000 * 0.2)
    audio_np = np.random.randn(audio_length).astype(np.float32) * 0.1
    audio = torch.from_numpy(audio_np).unsqueeze(0).to(pipeline.device)

    print(f"Test audio: {audio.shape} ({audio_length / 16000:.2f}s)")
    print()

    # Warmup
    print("Warming up...")
    for _ in range(3):
        with torch.autocast(device_type=pipeline.device, dtype=pipeline.dtype):
            _ = pipeline.hubert.extract(audio, output_layer=12, output_dim=768)
        if pipeline.fcpe is not None:
            with torch.autocast(device_type=pipeline.device, dtype=pipeline.dtype):
                _ = pipeline.fcpe.infer(audio, threshold=0.006)
        elif pipeline.rmvpe is not None:
            with torch.autocast(device_type=pipeline.device, dtype=pipeline.dtype):
                _ = pipeline.rmvpe.infer(audio)
    print("Warmup complete")
    print()

    # Run benchmarks
    iterations = 20

    print(f"Running benchmarks ({iterations} iterations)...")
    print("-" * 80)

    # Sequential
    mean_seq, std_seq = benchmark_sequential(pipeline, audio, iterations)
    print(f"1. Sequential:      {mean_seq:6.2f} ± {std_seq:5.2f} ms")

    # ThreadPool
    mean_tp, std_tp = benchmark_threadpool(pipeline, audio, iterations)
    print(f"2. ThreadPool:      {mean_tp:6.2f} ± {std_tp:5.2f} ms")
    speedup_tp = mean_seq / mean_tp
    print(f"   Speedup: {speedup_tp:.2f}x")

    # Streams (if GPU available)
    mean_st, std_st = benchmark_streams(pipeline, audio, iterations)
    if mean_st is not None:
        print(f"3. GPU Streams:     {mean_st:6.2f} ± {std_st:5.2f} ms")
        speedup_st = mean_seq / mean_st
        print(f"   Speedup: {speedup_st:.2f}x")
    else:
        print(f"3. GPU Streams:     N/A (CPU device)")

    print()
    print("=" * 80)
    print("Recommendation:")
    print("=" * 80)

    if mean_st is not None and mean_st < mean_tp:
        print(f"Use GPU Streams for best performance ({speedup_st:.2f}x speedup)")
    elif speedup_tp > 1.1:
        print(f"Use ThreadPool for {speedup_tp:.2f}x speedup")
    else:
        print("Sequential execution is optimal (minimal overhead)")


if __name__ == "__main__":
    main()
