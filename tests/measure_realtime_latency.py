"""
Measure actual inference time and end-to-end latency in realtime pipeline.

This simulates the full realtime voice changer flow and measures:
1. Pure inference time (HuBERT + F0 + Synthesizer)
2. Total latency (chunk buffering + inference + output buffering)
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from rcwx.config import RCWXConfig
from rcwx.pipeline.inference import RVCPipeline
from rcwx.pipeline.realtime import RealtimeConfig, RealtimeVoiceChanger


def measure_inference_time(
    pipeline: RVCPipeline,
    chunk_sec: float,
    input_sr: int = 16000,
    f0_method: str = "fcpe",
    iterations: int = 10,
):
    """Measure pure inference time (no buffering)."""
    chunk_samples = int(chunk_sec * input_sr)
    times = []

    print(f"\nMeasuring pure inference time...")
    print(f"  Chunk size: {chunk_sec*1000:.0f}ms ({chunk_samples} samples)")
    print(f"  F0 method: {f0_method}")
    print(f"  Iterations: {iterations}")
    print()

    # Warmup
    dummy_audio = np.random.randn(chunk_samples).astype(np.float32) * 0.1
    for _ in range(3):
        _ = pipeline.infer(
            dummy_audio,
            input_sr=input_sr,
            pitch_shift=0,
            f0_method=f0_method,
            index_rate=0.0,
            use_feature_cache=False,
        )

    # Clear cache to avoid cache effects
    pipeline.clear_cache()

    # Measure
    for i in range(iterations):
        # Generate test audio
        audio = np.random.randn(chunk_samples).astype(np.float32) * 0.1

        start = time.perf_counter()
        _ = pipeline.infer(
            audio,
            input_sr=input_sr,
            pitch_shift=0,
            f0_method=f0_method,
            index_rate=0.1,
            use_feature_cache=True,
            use_parallel_extraction=False,  # Disable for accurate measurement
        )
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

        if i < 5:
            print(f"  Iteration {i+1}: {elapsed:.1f}ms")

    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)

    print()
    print(f"Pure inference time:")
    print(f"  Mean: {mean_time:.1f} ± {std_time:.1f}ms")
    print(f"  Min:  {min_time:.1f}ms")
    print(f"  Max:  {max_time:.1f}ms")

    return mean_time, std_time


def measure_end_to_end_latency(
    model_path: str,
    config: RCWXConfig,
    chunk_sec: float,
    f0_method: str = "fcpe",
    duration_sec: float = 3.0,
):
    """Measure end-to-end latency in full realtime pipeline."""
    print(f"\nMeasuring end-to-end latency...")
    print(f"  Chunk size: {chunk_sec*1000:.0f}ms")
    print(f"  F0 method: {f0_method}")
    print(f"  Duration: {duration_sec}s")
    print()

    # Create realtime config
    rt_config = RealtimeConfig(
        mic_sample_rate=48000,
        input_sample_rate=16000,
        output_sample_rate=48000,
        chunk_sec=chunk_sec,
        pitch_shift=0,
        use_f0=True,
        f0_method=f0_method,
        prebuffer_chunks=0,
        buffer_margin=0.3,
        index_rate=0.1,
        index_k=4,
        resample_method="linear",
        use_parallel_extraction=False,  # Disable for accurate measurement
        use_feature_cache=True,
        context_sec=0.01,
        lookahead_sec=0.0,
        extra_sec=0.0,
        crossfade_sec=0.05,
        use_sola=True,
        voice_gate_mode="energy",
        energy_threshold=0.2,
    )

    # Initialize pipeline
    pipeline = RVCPipeline(
        model_path=str(model_path),
        device=config.device,
        dtype=config.dtype,
        use_f0=config.inference.use_f0,
        use_compile=False,  # Disabled for testing
        models_dir=config.models_dir,
    )
    pipeline.load()

    # Initialize voice changer
    changer = RealtimeVoiceChanger(
        pipeline=pipeline,
        config=rt_config,
    )

    # Generate test audio (simulate microphone input at 48kHz)
    mic_sr = 48000
    total_samples = int(duration_sec * mic_sr)
    test_audio = (np.random.randn(total_samples).astype(np.float32) * 0.1).astype(np.float32)

    # Process in chunks and measure
    mic_chunk_samples = int(chunk_sec * mic_sr)
    num_chunks = total_samples // mic_chunk_samples

    latencies = []
    inference_times = []

    print(f"Processing {num_chunks} chunks...")

    for i in range(num_chunks):
        start_idx = i * mic_chunk_samples
        end_idx = start_idx + mic_chunk_samples
        chunk = test_audio[start_idx:end_idx]

        # Add input and process (simulates the full pipeline)
        chunk_start = time.perf_counter()

        # Add chunk to input buffer
        changer.add_input(chunk)

        # Process next chunk from queue
        processed = changer.process_next_chunk()

        chunk_elapsed = (time.perf_counter() - chunk_start) * 1000

        if not processed:
            continue

        # Get stats from changer
        inference_time = changer.stats.inference_ms
        latency = changer.stats.latency_ms

        if i >= 3:  # Skip first few chunks (warmup)
            latencies.append(latency)
            inference_times.append(inference_time)

        if i < 5:
            print(
                f"  Chunk {i+1}: infer={inference_time:.1f}ms, "
                f"latency={latency:.1f}ms, total={chunk_elapsed:.1f}ms"
            )

    # Calculate statistics
    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    mean_inference = np.mean(inference_times)
    std_inference = np.std(inference_times)

    print()
    print(f"End-to-end measurements (excluding first 3 warmup chunks):")
    print(f"  Inference time: {mean_inference:.1f} ± {std_inference:.1f}ms")
    print(f"  Total latency:  {mean_latency:.1f} ± {std_latency:.1f}ms")
    print()
    print(f"Latency breakdown:")
    print(f"  Chunk buffering: {chunk_sec*1000:.0f}ms (theoretical)")
    print(f"  Inference: {mean_inference:.1f}ms (measured)")
    print(f"  Output buffer: {chunk_sec*1000*0.3:.0f}ms (theoretical, margin=0.3)")
    print(f"  Other overhead: {mean_latency - chunk_sec*1000 - mean_inference - chunk_sec*1000*0.3:.1f}ms")
    print()
    print(f"Latency ratio: {mean_latency / mean_inference:.1f}x inference time")

    return mean_latency, mean_inference


def main():
    """Run latency measurements."""
    print("=" * 80)
    print("Real-time Latency Measurement")
    print("=" * 80)

    # Load config
    config = RCWXConfig.load()

    if config.last_model_path is None:
        print("No model configured. Please run the GUI first.")
        return

    print()
    print(f"Model: {config.last_model_path}")
    print(f"Device: {config.device}")
    print(f"Chunk size: {config.audio.chunk_sec*1000:.0f}ms")
    print(f"F0 method: {config.inference.f0_method}")
    print(f"Prebuffer: {config.audio.prebuffer_chunks} chunks")
    print(f"Buffer margin: {config.audio.buffer_margin}")
    print()

    # Initialize pipeline for pure inference measurement
    # Disable compile for testing to avoid ThreadPool + compile issues
    pipeline = RVCPipeline(
        model_path=config.last_model_path,
        device=config.device,
        dtype=config.dtype,
        use_f0=config.inference.use_f0,
        use_compile=False,  # Disabled for testing
        models_dir=config.models_dir,
    )
    pipeline.load()

    # Measure pure inference time
    mean_infer, std_infer = measure_inference_time(
        pipeline,
        chunk_sec=config.audio.chunk_sec,
        f0_method=config.inference.f0_method,
        iterations=20,
    )

    # Measure end-to-end latency
    mean_latency, mean_e2e_infer = measure_end_to_end_latency(
        model_path=config.last_model_path,
        config=config,
        chunk_sec=config.audio.chunk_sec,
        f0_method=config.inference.f0_method,
        duration_sec=3.0,
    )

    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print()
    print(f"Pure inference:     {mean_infer:.1f}ms")
    print(f"E2E inference:      {mean_e2e_infer:.1f}ms")
    print(f"Total latency:      {mean_latency:.1f}ms")
    print()
    print(f"Latency overhead:   {mean_latency - mean_e2e_infer:.1f}ms ({(mean_latency / mean_e2e_infer - 1) * 100:.0f}%)")
    print(f"Main contributor:   Chunk buffering (~{config.audio.chunk_sec*1000:.0f}ms)")


if __name__ == "__main__":
    main()
