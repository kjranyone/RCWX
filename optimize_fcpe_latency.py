"""Optimize FCPE for minimum latency with no buffer underruns."""
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
pipeline = RVCPipeline(config.last_model_path, device=config.device, use_compile=False)
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

# Test different chunk sizes with FCPE
# FCPE is 37% faster than RMVPE, so we can use smaller chunks
chunk_sizes = [0.10, 0.12, 0.15, 0.18, 0.20, 0.25]
prebuffer_values = [1, 2, 3]

results = []

for chunk_sec in chunk_sizes:
    for prebuffer_chunks in prebuffer_values:
        logger.info(f"\n{'='*70}")
        logger.info(f"Testing: chunk={chunk_sec*1000:.0f}ms, prebuffer={prebuffer_chunks}")
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
                prebuffer_chunks=prebuffer_chunks,
                buffer_margin=0.5,  # Default
                pitch_shift=0,
                use_f0=True,
                f0_method="fcpe",  # Use FCPE for low latency
                index_rate=0.0,
                voice_gate_mode="off",
                use_feature_cache=True,
            )

            changer = RealtimeVoiceChanger(pipeline, config=rt_config)
            pipeline.clear_cache()
            changer._recalculate_buffers()
            changer._running = True

            # Set large buffer for testing (to collect all output)
            changer.output_buffer.set_max_latency(1000000)

            # Feed audio and measure time
            input_block_size = int(48000 * 0.02)  # 20ms blocks
            chunks_processed = 0
            total_inference_time = 0.0
            inference_times = []

            start_total = time.perf_counter()

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
                    total_inference_time += chunk_time
                    inference_times.append(chunk_time)

                changer.get_output_chunk(0)  # Drain to buffer

            total_time = (time.perf_counter() - start_total) * 1000  # ms

            # Statistics
            avg_inference = total_inference_time / chunks_processed if chunks_processed > 0 else 0
            max_inference = max(inference_times) if inference_times else 0
            min_inference = min(inference_times) if inference_times else 0

            # Calculate RTF (Real-Time Factor)
            audio_duration_ms = (len(audio) / 48000) * 1000
            rtf = total_inference_time / audio_duration_ms

            # Theoretical latency
            chunk_duration_ms = chunk_sec * 1000
            prebuffer_latency = prebuffer_chunks * chunk_duration_ms
            theoretical_latency = chunk_duration_ms + avg_inference + prebuffer_latency

            # Check for underruns
            underruns = changer.stats.buffer_underruns
            overruns = changer.stats.buffer_overruns

            logger.info(f"\nResults:")
            logger.info(f"  Chunks processed: {chunks_processed}")
            logger.info(f"  Avg inference: {avg_inference:.1f}ms (min: {min_inference:.1f}, max: {max_inference:.1f})")
            logger.info(f"  RTF: {rtf:.2f}x")
            logger.info(f"  Can run real-time: {'YES' if rtf < 0.7 else 'NO'}")
            logger.info(f"  Theoretical latency: {theoretical_latency:.0f}ms")
            logger.info(f"    - Chunk duration: {chunk_duration_ms:.0f}ms")
            logger.info(f"    - Prebuffer: {prebuffer_latency:.0f}ms ({prebuffer_chunks} chunks)")
            logger.info(f"    - Processing: {avg_inference:.0f}ms")
            logger.info(f"  Buffer underruns: {underruns}")
            logger.info(f"  Buffer overruns: {overruns}")

            status = "✅ OK" if rtf < 0.7 and underruns == 0 else "❌ FAIL"
            logger.info(f"  Status: {status}")

            results.append({
                "chunk_ms": chunk_sec * 1000,
                "prebuffer": prebuffer_chunks,
                "rtf": rtf,
                "latency": theoretical_latency,
                "underruns": underruns,
                "overruns": overruns,
                "avg_inference": avg_inference,
                "max_inference": max_inference,
                "status": status,
            })

        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()

# Summary
logger.info(f"\n{'='*70}")
logger.info("SUMMARY - Optimal Configurations")
logger.info(f"{'='*70}")
logger.info(f"{'Chunk':<8} {'Pre':<5} {'RTF':<7} {'Latency':<10} {'Under':<7} {'Status':<10}")
logger.info("-" * 70)

# Find best configurations (no underruns, lowest latency)
ok_results = [r for r in results if r["underruns"] == 0 and r["rtf"] < 0.7]
if ok_results:
    ok_results.sort(key=lambda x: x["latency"])
    logger.info("\nConfigurations with NO underruns (sorted by latency):")
    for r in ok_results[:5]:  # Top 5
        logger.info(
            f"{r['chunk_ms']:<8.0f} {r['prebuffer']:<5} {r['rtf']:<7.2f} {r['latency']:<10.0f} "
            f"{r['underruns']:<7} {r['status']:<10}"
        )

    best = ok_results[0]
    logger.info(f"\n🎯 RECOMMENDED CONFIGURATION:")
    logger.info(f"   chunk_sec: {best['chunk_ms']/1000:.2f} ({best['chunk_ms']:.0f}ms)")
    logger.info(f"   prebuffer_chunks: {best['prebuffer']}")
    logger.info(f"   Expected latency: ~{best['latency']:.0f}ms")
    logger.info(f"   RTF: {best['rtf']:.2f}x (margin: {(0.7 - best['rtf']):.2f}x)")
else:
    logger.warning("No configuration met the criteria (RTF < 0.7 and no underruns)")

# Show all results
logger.info("\nAll results:")
for r in results:
    logger.info(
        f"{r['chunk_ms']:<8.0f} {r['prebuffer']:<5} {r['rtf']:<7.2f} {r['latency']:<10.0f} "
        f"{r['underruns']:<7} {r['status']:<10}"
    )
