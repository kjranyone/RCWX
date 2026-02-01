"""Profile RTF for different chunk sizes to find optimal setting."""
import logging
import sys
import time
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

# Load config and pipeline
config = RCWXConfig.load()
pipeline = RVCPipeline(config.last_model_path, device=config.device, use_compile=True)  # ENABLE COMPILE!
pipeline.load()

# Load test audio (first 10 seconds)
test_file = Path("sample_data/seki.wav")
sr, audio_raw = wavfile.read(test_file)
if audio_raw.dtype == np.int16:
    audio_raw = audio_raw.astype(np.float32) / 32768.0
if len(audio_raw.shape) > 1:
    audio_raw = audio_raw[:, 0]

audio = resample(audio_raw, sr, 48000)[:int(48000 * 10)]  # First 10 seconds
logger.info(f"Audio: {len(audio)} samples @ 48kHz (10 seconds)")

# Test chunk sizes from 80ms to 200ms
chunk_sizes_ms = [80, 90, 96, 100, 110, 120, 130, 140, 150, 160, 180, 200]

results = []

for chunk_ms in chunk_sizes_ms:
    chunk_sec = chunk_ms / 1000.0
    logger.info(f"\n{'='*70}")
    logger.info(f"Testing chunk_sec={chunk_sec:.3f} ({chunk_ms}ms)")
    logger.info(f"{'='*70}")

    try:
        rt_config = RealtimeConfig(
            mic_sample_rate=48000,
            output_sample_rate=48000,
            chunk_sec=chunk_sec,
            context_sec=0.05,
            lookahead_sec=0.0,
            crossfade_sec=0.05,
            use_sola=False,
            prebuffer_chunks=2,  # Fixed prebuffer for comparison
            buffer_margin=0.5,
            pitch_shift=0,
            use_f0=True,
            f0_method="fcpe",
            index_rate=0.0,
            voice_gate_mode="off",
            use_feature_cache=True,
        )

        changer = RealtimeVoiceChanger(pipeline, config=rt_config)
        pipeline.clear_cache()
        changer._recalculate_buffers()
        changer._running = True
        changer.output_buffer.set_max_latency(1000000)

        # Feed audio and measure time
        input_block_size = int(48000 * 0.02)
        chunks_processed = 0
        inference_times = []

        for pos in range(0, len(audio), input_block_size):
            block = audio[pos:pos+input_block_size]
            if len(block) < input_block_size:
                block = np.pad(block, (0, input_block_size - len(block)))
            changer.process_input_chunk(block)

            # Process chunks and measure time
            while True:
                start_chunk = time.perf_counter()
                processed = changer.process_next_chunk()
                if not processed:
                    break
                chunk_time = (time.perf_counter() - start_chunk) * 1000  # ms

                chunks_processed += 1
                inference_times.append(chunk_time)

            changer.get_output_chunk(0)

        # Statistics
        if inference_times:
            avg_inference = np.mean(inference_times)
            median_inference = np.median(inference_times)
            p95_inference = np.percentile(inference_times, 95)
            p99_inference = np.percentile(inference_times, 99)
            max_inference = np.max(inference_times)
            min_inference = np.min(inference_times)
            std_inference = np.std(inference_times)

            # Calculate RTF
            audio_duration_ms = (len(audio) / 48000) * 1000
            total_inference_time = np.sum(inference_times)
            rtf = total_inference_time / audio_duration_ms

            # Calculate margin (how much faster than realtime)
            # For stable realtime, need avg processing < chunk duration
            margin_avg = (chunk_ms - avg_inference) / chunk_ms
            margin_p95 = (chunk_ms - p95_inference) / chunk_ms
            margin_p99 = (chunk_ms - p99_inference) / chunk_ms

            # Check underruns
            underruns = changer.stats.buffer_underruns

            logger.info(f"\nResults for {chunk_ms}ms chunks:")
            logger.info(f"  Chunks processed: {chunks_processed}")
            logger.info(f"  Avg inference: {avg_inference:.1f}ms")
            logger.info(f"  Median: {median_inference:.1f}ms")
            logger.info(f"  P95: {p95_inference:.1f}ms")
            logger.info(f"  P99: {p99_inference:.1f}ms")
            logger.info(f"  Max: {max_inference:.1f}ms")
            logger.info(f"  Std: {std_inference:.1f}ms")
            logger.info(f"  RTF: {rtf:.3f}x")
            logger.info(f"  Margin (avg): {margin_avg*100:.1f}%")
            logger.info(f"  Margin (P95): {margin_p95*100:.1f}%")
            logger.info(f"  Margin (P99): {margin_p99*100:.1f}%")
            logger.info(f"  Buffer underruns: {underruns}")

            # Determine if stable
            stable = (
                rtf < 0.7 and  # Total RTF safety margin
                margin_p99 > 0 and  # Even P99 should finish within chunk time
                underruns == 0  # No underruns
            )

            status = "✅ STABLE" if stable else "⚠️  UNSTABLE"
            logger.info(f"  Status: {status}")

            results.append({
                "chunk_ms": chunk_ms,
                "avg": avg_inference,
                "median": median_inference,
                "p95": p95_inference,
                "p99": p99_inference,
                "max": max_inference,
                "rtf": rtf,
                "margin_avg": margin_avg,
                "margin_p95": margin_p95,
                "margin_p99": margin_p99,
                "underruns": underruns,
                "stable": stable,
            })

    except Exception as e:
        logger.error(f"Error with {chunk_ms}ms: {e}")
        import traceback
        traceback.print_exc()

# Summary
logger.info(f"\n{'='*70}")
logger.info("SUMMARY - Chunk Size Optimization")
logger.info(f"{'='*70}")
logger.info(f"{'Chunk':<8} {'Avg':<8} {'P95':<8} {'P99':<8} {'RTF':<8} {'Margin':<10} {'Under':<7} {'Status'}")
logger.info("-" * 100)

for r in results:
    margin_str = f"{r['margin_p99']*100:.1f}%"
    status = "✅" if r["stable"] else "⚠️"
    logger.info(
        f"{r['chunk_ms']:<8} {r['avg']:<8.1f} {r['p95']:<8.1f} {r['p99']:<8.1f} "
        f"{r['rtf']:<8.3f} {margin_str:<10} {r['underruns']:<7} {status}"
    )

# Find optimal configuration
stable_results = [r for r in results if r["stable"]]
if stable_results:
    # Sort by chunk size (lowest latency first)
    stable_results.sort(key=lambda x: x["chunk_ms"])
    best = stable_results[0]

    logger.info(f"\n🎯 OPTIMAL CONFIGURATION:")
    logger.info(f"   Chunk size: {best['chunk_ms']}ms")
    logger.info(f"   Avg processing: {best['avg']:.1f}ms")
    logger.info(f"   P99 processing: {best['p99']:.1f}ms")
    logger.info(f"   RTF: {best['rtf']:.3f}x")
    logger.info(f"   Safety margin (P99): {best['margin_p99']*100:.1f}%")
    logger.info(f"   Theoretical latency: ~{best['chunk_ms'] * 3 + best['avg']:.0f}ms (chunk + 2x prebuffer + processing)")
else:
    logger.warning("\n⚠️  No stable configuration found with prebuffer=2")
    logger.warning("   Consider increasing prebuffer or chunk size")
