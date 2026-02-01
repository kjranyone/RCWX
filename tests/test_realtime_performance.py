"""
Test realtime voice changer performance: inference time and latency.

This test measures the actual performance in a realistic scenario:
- Uses RealtimeVoiceChanger with actual configuration
- Processes real audio chunks
- Measures inference time and end-to-end latency
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from rcwx.audio.resample import resample
from rcwx.config import RCWXConfig
from rcwx.pipeline.inference import RVCPipeline
from rcwx.pipeline.realtime import RealtimeConfig, RealtimeVoiceChanger

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def test_performance():
    """Test realtime performance with current configuration."""
    logger.info("=" * 70)
    logger.info("Realtime Performance Test")
    logger.info("=" * 70)
    logger.info("")

    # Load config
    config = RCWXConfig.load()

    if config.last_model_path is None:
        logger.error("No model configured. Please run the GUI first.")
        return

    logger.info(f"Model: {config.last_model_path}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Chunk size: {config.audio.chunk_sec*1000:.0f}ms")
    logger.info(f"F0 method: {config.inference.f0_method}")
    logger.info(f"Prebuffer: {config.audio.prebuffer_chunks} chunks")
    logger.info(f"Buffer margin: {config.audio.buffer_margin}")
    logger.info(f"Use compile: {config.inference.use_compile}")
    logger.info(f"Parallel extraction: {config.inference.use_parallel_extraction}")
    logger.info(f"Index k: {config.inference.index_k}")
    logger.info(f"Resample method: {config.inference.resample_method}")
    logger.info("")

    # Create realtime config from RCWXConfig
    rt_config = RealtimeConfig(
        mic_sample_rate=config.audio.output_sample_rate,
        input_sample_rate=config.audio.sample_rate,
        output_sample_rate=config.audio.output_sample_rate,
        chunk_sec=config.audio.chunk_sec,
        pitch_shift=config.inference.pitch_shift,
        use_f0=config.inference.use_f0,
        f0_method=config.inference.f0_method,
        prebuffer_chunks=config.audio.prebuffer_chunks,
        buffer_margin=config.audio.buffer_margin,
        index_rate=config.inference.index_ratio,
        index_k=config.inference.index_k,
        resample_method=config.inference.resample_method,
        use_parallel_extraction=config.inference.use_parallel_extraction,
        use_feature_cache=config.inference.use_feature_cache,
        context_sec=config.inference.context_sec,
        lookahead_sec=config.inference.lookahead_sec,
        extra_sec=config.inference.extra_sec,
        crossfade_sec=config.inference.crossfade_sec,
        use_sola=config.inference.use_sola,
        voice_gate_mode=config.inference.voice_gate_mode,
        energy_threshold=config.inference.energy_threshold,
        denoise_enabled=config.inference.denoise.enabled,
        denoise_method=config.inference.denoise.method,
    )

    # Initialize pipeline
    logger.info("Loading pipeline...")
    pipeline = RVCPipeline(
        model_path=config.last_model_path,
        device=config.device,
        dtype=config.dtype,
        use_f0=config.inference.use_f0,
        use_compile=config.inference.use_compile,
        models_dir=config.models_dir,
    )
    pipeline.load()
    logger.info("Pipeline loaded")
    logger.info("")

    # Initialize voice changer
    logger.info("Initializing RealtimeVoiceChanger...")
    changer = RealtimeVoiceChanger(
        pipeline=pipeline,
        config=rt_config,
    )
    logger.info("RealtimeVoiceChanger initialized")
    logger.info("")

    # Load test audio
    test_audio_path = Path("sample_data/seki.wav")
    if not test_audio_path.exists():
        logger.error(f"Test audio not found: {test_audio_path}")
        return

    sr, audio = wavfile.read(test_audio_path)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    logger.info(f"Test audio: {len(audio)/sr:.2f}s @ {sr}Hz")

    # Resample to mic rate if needed
    if sr != rt_config.mic_sample_rate:
        audio = resample(audio, sr, rt_config.mic_sample_rate)
        logger.info(f"Resampled to {rt_config.mic_sample_rate}Hz")

    # Process in chunks
    mic_chunk_samples = int(rt_config.chunk_sec * rt_config.mic_sample_rate)
    num_chunks = len(audio) // mic_chunk_samples

    logger.info(f"Processing {num_chunks} chunks...")
    logger.info(f"Chunk size: {mic_chunk_samples} samples @ {rt_config.mic_sample_rate}Hz")
    logger.info("")

    inference_times = []
    latencies = []

    import time

    for i in range(num_chunks):
        start_idx = i * mic_chunk_samples
        end_idx = start_idx + mic_chunk_samples
        chunk = audio[start_idx:end_idx]

        # Measure chunk processing time
        chunk_start = time.perf_counter()

        # Add input to buffer
        changer.process_input_chunk(chunk)

        # Process chunk from queue (this does the actual inference)
        processed = changer.process_next_chunk()

        chunk_elapsed = (time.perf_counter() - chunk_start) * 1000

        if processed:
            # Measure inference time (approximate - includes all processing)
            infer_ms = chunk_elapsed

            # Latency = chunk buffering + inference + output buffering
            # Approximate: chunk_sec + inference + buffer_margin * chunk_sec
            latency_ms = (
                rt_config.chunk_sec * 1000  # Input buffering
                + infer_ms  # Processing time
                + rt_config.chunk_sec * 1000 * rt_config.buffer_margin  # Output buffering
            )

            # Skip first few chunks (warmup)
            if i >= 3:
                inference_times.append(infer_ms)
                latencies.append(latency_ms)

            # Print first 10 chunks
            if i < 10:
                logger.info(
                    f"Chunk {i+1:3d}: infer={infer_ms:6.1f}ms, "
                    f"latency={latency_ms:6.1f}ms (estimated), "
                    f"buf_under={changer.stats.buffer_underruns}, "
                    f"buf_over={changer.stats.buffer_overruns}"
                )

    logger.info("")
    logger.info("=" * 70)
    logger.info("Performance Results (excluding first 3 warmup chunks)")
    logger.info("=" * 70)
    logger.info("")

    if inference_times:
        mean_infer = np.mean(inference_times)
        std_infer = np.std(inference_times)
        min_infer = np.min(inference_times)
        max_infer = np.max(inference_times)
        p50_infer = np.percentile(inference_times, 50)
        p95_infer = np.percentile(inference_times, 95)
        p99_infer = np.percentile(inference_times, 99)

        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)

        logger.info("Inference Time:")
        logger.info(f"  Mean:   {mean_infer:6.1f} ± {std_infer:5.1f} ms")
        logger.info(f"  Median: {p50_infer:6.1f} ms")
        logger.info(f"  Min:    {min_infer:6.1f} ms")
        logger.info(f"  Max:    {max_infer:6.1f} ms")
        logger.info(f"  P95:    {p95_infer:6.1f} ms")
        logger.info(f"  P99:    {p99_infer:6.1f} ms")
        logger.info("")

        logger.info("Total Latency:")
        logger.info(f"  Mean:   {mean_latency:6.1f} ± {std_latency:5.1f} ms")
        logger.info(f"  Median: {p50_latency:6.1f} ms")
        logger.info(f"  Min:    {min_latency:6.1f} ms")
        logger.info(f"  Max:    {max_latency:6.1f} ms")
        logger.info(f"  P95:    {p95_latency:6.1f} ms")
        logger.info(f"  P99:    {p99_latency:6.1f} ms")
        logger.info("")

        logger.info("Latency Breakdown:")
        logger.info(f"  Chunk buffering: {rt_config.chunk_sec*1000:6.1f} ms (theoretical)")
        logger.info(f"  Inference:       {mean_infer:6.1f} ms (measured)")
        logger.info(
            f"  Output buffer:   {rt_config.chunk_sec*1000*rt_config.buffer_margin:6.1f} ms (theoretical, margin={rt_config.buffer_margin})"
        )
        other_overhead = (
            mean_latency - rt_config.chunk_sec * 1000 - mean_infer - rt_config.chunk_sec * 1000 * rt_config.buffer_margin
        )
        logger.info(f"  Other overhead:  {other_overhead:6.1f} ms")
        logger.info("")

        logger.info("Performance:")
        logger.info(f"  Latency/Inference ratio: {mean_latency / mean_infer:.2f}x")
        logger.info(
            f"  Real-time factor: {mean_infer / (rt_config.chunk_sec*1000):.2f}x (1.0 = realtime)"
        )
        logger.info(f"  Buffer underruns: {changer.stats.buffer_underruns}")
        logger.info(f"  Buffer overruns:  {changer.stats.buffer_overruns}")
        logger.info("")

        # Assessment
        logger.info("=" * 70)
        logger.info("Assessment:")
        logger.info("=" * 70)
        logger.info("")

        if mean_infer < 80:
            logger.info("✓ Inference time is EXCELLENT (< 80ms)")
        elif mean_infer < 120:
            logger.info("✓ Inference time is GOOD (80-120ms)")
        elif mean_infer < 200:
            logger.info("⚠ Inference time is ACCEPTABLE (120-200ms)")
        else:
            logger.info("✗ Inference time is SLOW (> 200ms)")
            logger.info("  Possible causes:")
            logger.info("  - torch.compile disabled or not working")
            logger.info("  - CPU device instead of GPU")
            logger.info("  - Parallel extraction causing issues")

        logger.info("")

        if mean_latency < 150:
            logger.info("✓ Total latency is EXCELLENT (< 150ms)")
        elif mean_latency < 250:
            logger.info("✓ Total latency is GOOD (150-250ms)")
        elif mean_latency < 400:
            logger.info("⚠ Total latency is ACCEPTABLE (250-400ms)")
        else:
            logger.info("✗ Total latency is HIGH (> 400ms)")
            logger.info("  Possible optimizations:")
            logger.info(f"  - Reduce chunk size (current: {rt_config.chunk_sec*1000:.0f}ms)")
            logger.info(f"  - Reduce buffer margin (current: {rt_config.buffer_margin})")
            logger.info(f"  - Reduce prebuffer (current: {rt_config.prebuffer_chunks})")

        logger.info("")

        if changer.stats.buffer_underruns > 0:
            logger.info(f"⚠ Buffer underruns detected: {changer.stats.buffer_underruns}")
            logger.info("  Consider increasing buffer_margin or chunk_sec")
        else:
            logger.info("✓ No buffer underruns")

        logger.info("")

    else:
        logger.error("No chunks processed successfully")


if __name__ == "__main__":
    test_performance()
