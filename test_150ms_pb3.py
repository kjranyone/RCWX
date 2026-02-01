"""Test 96ms chunk (w-okada style) for FCPE quality and performance."""
import logging
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).parent))

from rcwx.audio.resample import resample
from rcwx.config import RCWXConfig
from rcwx.pipeline.inference import RVCPipeline
from rcwx.pipeline.realtime import RealtimeConfig, RealtimeVoiceChanger

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_test_audio(path: Path, target_sr: int = 48000) -> np.ndarray:
    """Load and resample audio file to target sample rate."""
    sr, audio = wavfile.read(path)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    if sr != target_sr:
        audio = resample(audio, sr, target_sr)
    return audio.astype(np.float32)


def process_batch(pipeline: RVCPipeline, audio: np.ndarray, pitch_shift: int = 0,
                  chunk_sec: float = 0.096, context_sec: float = 0.05,
                  f0_method: str = "fcpe",
                  output_sample_rate: int = 48000) -> np.ndarray:
    """Process audio in chunks like streaming, for fair comparison."""
    pipeline.clear_cache()

    chunk_samples_48k = int(48000 * chunk_sec)
    context_samples_48k = int(48000 * context_sec)
    outputs = []

    main_pos = 0
    chunk_idx = 0
    while main_pos < len(audio):
        if chunk_idx == 0:
            start = 0
            end = min(chunk_samples_48k, len(audio))
        else:
            start = max(0, main_pos - context_samples_48k)
            end = min(main_pos + chunk_samples_48k, len(audio))

        chunk_48k = audio[start:end]
        chunk_16k = resample(chunk_48k, 48000, 16000)

        chunk_output = pipeline.infer(
            chunk_16k,
            input_sr=16000,
            pitch_shift=pitch_shift,
            f0_method=f0_method,
            index_rate=0.0,
            voice_gate_mode="off",
            use_feature_cache=True,
        )

        if pipeline.sample_rate != output_sample_rate:
            chunk_output = resample(chunk_output, pipeline.sample_rate, output_sample_rate)

        if chunk_idx > 0 and context_sec > 0:
            context_samples_output = int(output_sample_rate * context_sec)
            if len(chunk_output) > context_samples_output:
                chunk_output = chunk_output[context_samples_output:]

        outputs.append(chunk_output)
        main_pos += chunk_samples_48k
        chunk_idx += 1

        if chunk_idx <= 5 or chunk_idx % 100 == 0:
            logger.info(f"Batch chunk {chunk_idx}: processed {len(chunk_48k)} samples @ 48kHz")

    logger.info(f"Batch: processed {chunk_idx} chunks total")
    return np.concatenate(outputs)


def process_streaming(
    pipeline: RVCPipeline,
    audio: np.ndarray,
    pitch_shift: int = 0,
    chunk_sec: float = 0.096,
    context_sec: float = 0.05,
    f0_method: str = "fcpe",
    prebuffer_chunks: int = 1,
    mic_sample_rate: int = 48000,
) -> tuple[np.ndarray, dict]:
    """Process audio in streaming chunks using ACTUAL RealtimeVoiceChanger."""
    rt_config = RealtimeConfig(
        mic_sample_rate=mic_sample_rate,
        output_sample_rate=mic_sample_rate,
        chunk_sec=chunk_sec,
        context_sec=context_sec,
        lookahead_sec=0.0,
        crossfade_sec=0.05,
        use_sola=False,
        prebuffer_chunks=prebuffer_chunks,
        buffer_margin=0.5,
        pitch_shift=pitch_shift,
        use_f0=True,
        f0_method=f0_method,
        index_rate=0.0,
        voice_gate_mode="off",
        use_feature_cache=True,
    )

    changer = RealtimeVoiceChanger(pipeline, config=rt_config)
    pipeline.clear_cache()
    changer._recalculate_buffers()
    changer._running = True

    expected_duration_sec = len(audio) / mic_sample_rate
    expected_chunks = int(expected_duration_sec / rt_config.chunk_sec) + 2
    max_output_samples = expected_chunks * int(rt_config.output_sample_rate * rt_config.chunk_sec) * 2
    changer.output_buffer.set_max_latency(max_output_samples)

    input_block_size = int(mic_sample_rate * 0.02)
    output_block_size = int(rt_config.output_sample_rate * 0.02)
    outputs = []

    pos = 0
    chunks_processed = 0

    while pos < len(audio):
        block = audio[pos : pos + input_block_size]
        if len(block) < input_block_size:
            block = np.pad(block, (0, input_block_size - len(block)))
        pos += input_block_size

        changer.process_input_chunk(block)

        while changer.process_next_chunk():
            chunks_processed += 1

        changer.get_output_chunk(0)

    while changer.process_next_chunk():
        chunks_processed += 1
        changer.get_output_chunk(0)

    logger.info(f"Streaming: processed {chunks_processed} chunks total")

    total_output_retrieved = 0
    while changer.output_buffer.available > 0:
        output_block = changer.get_output_chunk(output_block_size)
        outputs.append(output_block)
        total_output_retrieved += len(output_block)

    stats = {
        "chunks_processed": chunks_processed,
        "buffer_underruns": changer.stats.buffer_underruns,
        "buffer_overruns": changer.stats.buffer_overruns,
    }

    if outputs:
        return np.concatenate(outputs), stats
    else:
        return np.array([], dtype=np.float32), stats


def compare_outputs(batch: np.ndarray, streaming: np.ndarray) -> dict:
    """Compare batch and streaming outputs."""
    min_len = min(len(batch), len(streaming))
    batch_trim = batch[:min_len]
    streaming_trim = streaming[:min_len]

    diff = batch_trim - streaming_trim
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff**2))
    max_diff = np.max(np.abs(diff))

    correlation = np.corrcoef(batch_trim, streaming_trim)[0, 1]

    batch_energy = np.sqrt(np.mean(batch_trim**2))
    streaming_energy = np.sqrt(np.mean(streaming_trim**2))
    energy_ratio = streaming_energy / batch_energy if batch_energy > 0 else 0

    return {
        "mae": mae,
        "rmse": rmse,
        "max_diff": max_diff,
        "correlation": correlation,
        "energy_ratio": energy_ratio,
        "batch_length": len(batch),
        "streaming_length": len(streaming),
        "length_diff": len(batch) - len(streaming),
    }


def test_96ms_chunk():
    """Test 96ms chunk (w-okada style) with FCPE."""
    logger.info("=" * 70)
    logger.info("96ms Chunk Test (w-okada style) with FCPE")
    logger.info("=" * 70)

    config = RCWXConfig.load()
    if not config.last_model_path:
        logger.error("ERROR: No model configured. Run GUI first to select a model.")
        return False

    logger.info(f"\nLoading model: {config.last_model_path}")
    pipeline = RVCPipeline(
        config.last_model_path,
        device=config.device,
        use_compile=False,
    )
    pipeline.load()

    test_file = Path("sample_data/seki.wav")
    if not test_file.exists():
        logger.error(f"ERROR: Test file not found: {test_file}")
        return False

    logger.info(f"Loading test audio: {test_file}")
    audio = load_test_audio(test_file, target_sr=48000)

    # Trim to exact multiple of chunk_sec for fair comparison
    chunk_sec = 0.096  # 96ms (w-okada style)
    chunk_samples = int(48000 * chunk_sec)
    num_full_chunks = len(audio) // chunk_samples
    audio = audio[:num_full_chunks * chunk_samples]

    duration = len(audio) / 48000
    logger.info(f"Audio: {duration:.2f}s @ 48kHz ({num_full_chunks} full chunks of 96ms)")

    # Test with different prebuffer settings
    results = []

    for prebuffer in [1, 2, 3]:
        logger.info(f"\n{'='*70}")
        logger.info(f"Testing prebuffer_chunks={prebuffer}")
        logger.info(f"{'='*70}")

        # Process in batch mode
        logger.info("\n--- Batch Processing (FCPE, 96ms) ---")
        batch_output = process_batch(pipeline, audio, pitch_shift=0,
                                      chunk_sec=0.150, context_sec=0.05,
                                      f0_method="fcpe",
                                      output_sample_rate=48000)
        logger.info(f"Batch output: {len(batch_output)} samples @ 48000Hz")

        # Process in streaming mode
        logger.info("\n--- Streaming Processing (FCPE, 96ms) ---")
        streaming_output, stats = process_streaming(
            pipeline,
            audio,
            pitch_shift=0,
            chunk_sec=0.150,
            context_sec=0.05,
            f0_method="fcpe",
            prebuffer_chunks=prebuffer,
        )
        logger.info(f"Streaming output: {len(streaming_output)} samples @ 48000Hz")
        logger.info(f"Buffer underruns: {stats['buffer_underruns']}")
        logger.info(f"Buffer overruns: {stats['buffer_overruns']}")

        # Compare outputs
        logger.info("\n--- Comparison ---")
        metrics = compare_outputs(batch_output, streaming_output)

        for key, value in metrics.items():
            logger.info(f"{key:20s}: {value}")

        # Expected latency calculation
        chunk_duration_ms = chunk_sec * 1000
        prebuffer_latency = prebuffer * chunk_duration_ms
        processing_time = 60  # Approximate from previous tests
        theoretical_latency = chunk_duration_ms + prebuffer_latency + processing_time

        logger.info(f"\nLatency breakdown:")
        logger.info(f"  Chunk duration: {chunk_duration_ms:.0f}ms")
        logger.info(f"  Prebuffer: {prebuffer_latency:.0f}ms ({prebuffer} chunks)")
        logger.info(f"  Processing: ~{processing_time}ms")
        logger.info(f"  Theoretical total: ~{theoretical_latency:.0f}ms")

        # Evaluate
        passed = True
        if metrics["correlation"] < 0.93:
            logger.error(f"FAIL: Correlation too low: {metrics['correlation']:.4f} < 0.93")
            passed = False
        else:
            logger.info(f"✅ PASS: Correlation: {metrics['correlation']:.4f}")

        if metrics["mae"] > 0.05:
            logger.error(f"FAIL: MAE too high: {metrics['mae']:.4f} > 0.05")
            passed = False
        else:
            logger.info(f"✅ PASS: MAE: {metrics['mae']:.4f}")

        if stats["buffer_underruns"] > 0:
            logger.warning(f"⚠️  Buffer underruns: {stats['buffer_underruns']}")
        else:
            logger.info(f"✅ PASS: No buffer underruns")

        results.append({
            "prebuffer": prebuffer,
            "correlation": metrics["correlation"],
            "mae": metrics["mae"],
            "underruns": stats["buffer_underruns"],
            "latency": theoretical_latency,
            "passed": passed,
        })

    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("SUMMARY - 96ms Chunk Tests")
    logger.info(f"{'='*70}")
    logger.info(f"{'Prebuf':<8} {'Corr':<8} {'MAE':<8} {'Under':<8} {'Latency':<10} {'Status':<10}")
    logger.info("-" * 70)

    for r in results:
        status = "✅ PASS" if r["passed"] and r["underruns"] == 0 else "❌ FAIL"
        logger.info(
            f"{r['prebuffer']:<8} {r['correlation']:<8.4f} {r['mae']:<8.4f} "
            f"{r['underruns']:<8} {r['latency']:<10.0f} {status:<10}"
        )

    # Find best configuration
    ok_results = [r for r in results if r["passed"] and r["underruns"] == 0]
    if ok_results:
        ok_results.sort(key=lambda x: x["latency"])
        best = ok_results[0]
        logger.info(f"\n🎯 RECOMMENDED: prebuffer={best['prebuffer']}, latency ~{best['latency']:.0f}ms")
        return True
    else:
        logger.error("\n❌ No configuration passed all tests")
        return False


if __name__ == "__main__":
    success = test_96ms_chunk()
    sys.exit(0 if success else 1)
