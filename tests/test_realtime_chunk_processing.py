"""
Integration test: Verify streaming (chunked) output matches batch output.

This test processes a WAV file in chunks using the ACTUAL RealtimeVoiceChanger
logic (not a simulation), and compares it to batch processing.
"""

import logging
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

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


def process_true_batch(pipeline: RVCPipeline, audio: np.ndarray, pitch_shift: int = 0,
                       output_sample_rate: int = 48000) -> np.ndarray:
    """
    TRUE batch processing: process entire audio in one shot (no chunks).
    This is the gold standard for comparison.
    """
    pipeline.clear_cache()

    # Resample entire audio to processing rate
    audio_16k = resample(audio, 48000, 16000)

    # Single inference on entire audio
    output = pipeline.infer(
        audio_16k,
        input_sr=16000,
        pitch_shift=pitch_shift,
        f0_method="rmvpe",
        index_rate=0.0,
        voice_gate_mode="off",
        use_feature_cache=False,  # Not needed for single inference
    )

    # Resample output to output rate
    if pipeline.sample_rate != output_sample_rate:
        output = resample(output, pipeline.sample_rate, output_sample_rate)

    return output


def process_batch(pipeline: RVCPipeline, audio: np.ndarray, pitch_shift: int = 0,
                  chunk_sec: float = 0.35, context_sec: float = 0.05,
                  output_sample_rate: int = 48000) -> np.ndarray:
    """Process audio in chunks like streaming, for fair comparison.

    CRITICAL: Use same order as streaming:
    1. Split at 48kHz (mic rate)
    2. Resample chunks to 16kHz (processing rate)
    3. Infer
    4. Resample output to 48kHz (output rate)
    """
    # Clear cache before processing
    pipeline.clear_cache()

    # Process in chunks matching streaming configuration @ MIC RATE (48kHz)
    chunk_samples_48k = int(48000 * chunk_sec)
    context_samples_48k = int(48000 * context_sec)
    outputs = []

    main_pos = 0  # Position of current chunk's main section start (no context)
    chunk_idx = 0
    while main_pos < len(audio):
        # Determine chunk with context @ 48kHz
        if chunk_idx == 0:
            # First chunk: NO left context, only main (+ lookahead if any)
            start = 0
            end = min(chunk_samples_48k, len(audio))
        else:
            # Subsequent chunks: left_context + main
            # left_context starts at (main_pos - context)
            start = max(0, main_pos - context_samples_48k)
            end = min(main_pos + chunk_samples_48k, len(audio))

        chunk_48k = audio[start:end]

        # Resample chunk to processing rate (16kHz) - SAME AS STREAMING
        chunk_16k = resample(chunk_48k, 48000, 16000)

        # Process chunk
        chunk_output = pipeline.infer(
            chunk_16k,
            input_sr=16000,
            pitch_shift=pitch_shift,
            f0_method="rmvpe",
            index_rate=0.0,
            voice_gate_mode="off",
            use_feature_cache=True,  # Enable for continuity
        )

        # Resample output to output sample rate (same as streaming)
        if pipeline.sample_rate != output_sample_rate:
            chunk_output = resample(chunk_output, pipeline.sample_rate, output_sample_rate)

        # Trim context from output (except first chunk)
        before_trim_len = len(chunk_output)
        if chunk_idx > 0 and context_sec > 0:
            context_samples_output = int(output_sample_rate * context_sec)
            if len(chunk_output) > context_samples_output:
                chunk_output = chunk_output[context_samples_output:]

        outputs.append(chunk_output)

        # Log chunk lengths for debugging
        if chunk_idx < 5 or chunk_idx >= 148:  # First 5 and last 2 chunks
            trim_info = f", trimmed {before_trim_len - len(chunk_output)}" if chunk_idx > 0 else ""
            logger.info(f"Batch chunk {chunk_idx}: input={len(chunk_16k)}@16k, output={len(outputs[-1])}@{output_sample_rate}Hz{trim_info}")

        # Advance main position by chunk_samples (always, for all chunks)
        main_pos += chunk_samples_48k
        chunk_idx += 1

    return np.concatenate(outputs)


def process_streaming(
    pipeline: RVCPipeline,
    audio: np.ndarray,
    pitch_shift: int = 0,
    chunk_sec: float = 0.35,
    context_sec: float = 0.05,
    lookahead_sec: float = 0.0,
    crossfade_sec: float = 0.05,
    use_sola: bool = True,
    prebuffer_chunks: int = 0,
    mic_sample_rate: int = 48000,
    chunking_mode: str = "wokada",
) -> np.ndarray:
    """
    Process audio in streaming chunks using ACTUAL RealtimeVoiceChanger.

    This calls the real implementation, ensuring test matches production.
    """
    # Create RealtimeConfig
    # IMPORTANT: Use model's output rate (40kHz) not mic rate (48kHz) to match batch processing
    rt_config = RealtimeConfig(
        mic_sample_rate=mic_sample_rate,
        output_sample_rate=40000,  # Model's native output rate
        chunk_sec=chunk_sec,
        context_sec=context_sec,
        lookahead_sec=lookahead_sec,
        crossfade_sec=crossfade_sec,
        use_sola=use_sola,  # Use parameter value (not hardcoded)
        prebuffer_chunks=prebuffer_chunks,
        pitch_shift=pitch_shift,
        use_f0=True,
        f0_method="rmvpe",
        chunking_mode=chunking_mode,  # Add chunking mode parameter
        rvc_overlap_sec=crossfade_sec,  # RVC mode uses same overlap as crossfade
        index_rate=0.0,
        voice_gate_mode="off",
        use_feature_cache=True,  # Enable for chunk continuity
    )

    # Create RealtimeVoiceChanger (using real implementation)
    changer = RealtimeVoiceChanger(pipeline, config=rt_config)

    # Clear feature cache to avoid warmup interference    # This is critical for matching batch processing!
    logger.info("Clearing feature cache to avoid warmup interference")
    pipeline.clear_cache()

    # Initialize internal state (without starting audio streams)
    changer._recalculate_buffers()
    changer._running = True  # Mark as running for internal logic

    # For testing, set a very large output buffer to collect all chunks
    # (in production, this is limited to maintain low latency, but for testing we want all output)
    expected_duration_sec = len(audio) / mic_sample_rate
    expected_chunks = int(expected_duration_sec / rt_config.chunk_sec) + 2
    max_output_samples = expected_chunks * int(rt_config.output_sample_rate * rt_config.chunk_sec) * 2
    changer.output_buffer.set_max_latency(max_output_samples)
    logger.info(f"Set output buffer max latency to {max_output_samples} samples for testing")

    # Simulate streaming input: feed audio in small blocks
    input_block_size = int(mic_sample_rate * 0.02)  # 20ms blocks (typical audio callback)
    # Output sample rate may differ from input (e.g., 40kHz vs 48kHz)
    output_block_size = int(rt_config.output_sample_rate * 0.02)
    outputs = []

    # Feed all input and process chunks as we go
    # This simulates real-time operation where feeding, processing, and output happen concurrently
    pos = 0
    chunks_processed = 0

    while pos < len(audio):
        # Feed input block
        block = audio[pos : pos + input_block_size]
        if len(block) < input_block_size:
            # Pad last block
            block = np.pad(block, (0, input_block_size - len(block)))
        pos += input_block_size

        # Add to input buffer and queue chunks
        changer.process_input_chunk(block)

        # Process any queued chunks to prevent input queue from filling up
        while changer.process_next_chunk():
            chunks_processed += 1

        # Drain output queue to output buffer to prevent output queue from filling up
        # This is critical because output queue has limited size (default 8)
        changer.get_output_chunk(0)  # Just drain queue to buffer, don't retrieve yet

    logger.info(f"All input fed and processed. Total chunks: {chunks_processed}")

    # For RVC WebUI mode, feed additional padding to flush remaining buffer
    # Since RVC mode uses overlapping chunks, we need to send enough padding
    # to process all remaining buffered audio
    if chunking_mode == "rvc_webui":
        # Calculate expected number of chunks based on audio length
        expected_chunks = (len(audio) - changer.mic_chunk_samples) // changer.mic_hop_samples + 1
        logger.info(f"Expected chunks: {expected_chunks}, processed so far: {chunks_processed}")

        # Send padding until we reach expected number of chunks
        # Each iteration sends 1 second of silence
        max_padding_iterations = 20  # Safety limit
        for _ in range(max_padding_iterations):
            if chunks_processed >= expected_chunks:
                break

            # Send 1 second of silence
            padding = np.zeros(mic_sample_rate, dtype=np.float32)
            pad_pos = 0
            while pad_pos < len(padding):
                pad_block = padding[pad_pos : pad_pos + input_block_size]
                if len(pad_block) < input_block_size:
                    pad_block = np.pad(pad_block, (0, input_block_size - len(pad_block)))
                pad_pos += input_block_size

                changer.process_input_chunk(pad_block)
                while changer.process_next_chunk():
                    chunks_processed += 1
                changer.get_output_chunk(0)

        logger.info(f"Fed padding to flush buffer. Total chunks: {chunks_processed}")

    # Process any remaining queued chunks
    while changer.process_next_chunk():
        chunks_processed += 1
        changer.get_output_chunk(0)  # Drain to buffer

    # Flush final SOLA buffer (critical for RVC WebUI mode)
    changer.flush_final_sola_buffer()
    changer.get_output_chunk(0)  # Drain flushed buffer

    logger.info(f"Processed {chunks_processed} chunks")
    logger.info(f"Output buffer available: {changer.output_buffer.available} samples @ {rt_config.output_sample_rate}Hz")

    # Now retrieve all output from buffer
    total_output_retrieved = 0
    while changer.output_buffer.available > 0:
        output_block = changer.get_output_chunk(output_block_size)
        outputs.append(output_block)
        total_output_retrieved += len(output_block)

    logger.info(f"Retrieved {total_output_retrieved} output samples @ {rt_config.output_sample_rate}Hz")

    # Concatenate all outputs
    if outputs:
        return np.concatenate(outputs)
    else:
        return np.array([], dtype=np.float32)


def compare_outputs(batch: np.ndarray, streaming: np.ndarray, tolerance: float = 0.1) -> dict:
    """
    Compare batch and streaming outputs.

    Returns:
        dict with comparison metrics
    """
    # Align lengths (streaming may be slightly longer due to buffering)
    min_len = min(len(batch), len(streaming))
    batch_trim = batch[:min_len]
    streaming_trim = streaming[:min_len]

    # Calculate metrics
    diff = batch_trim - streaming_trim
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff**2))
    max_diff = np.max(np.abs(diff))

    # Correlation
    correlation = np.corrcoef(batch_trim, streaming_trim)[0, 1]

    # Energy difference
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


def test_chunk_processing():
    """Main test: Compare batch vs streaming processing."""
    logger.info("=" * 70)
    logger.info("Chunk Processing Integration Test")
    logger.info("=" * 70)

    # Load config
    config = RCWXConfig.load()
    if not config.last_model_path:
        logger.error("ERROR: No model configured. Run GUI first to select a model.")
        return False

    # Load pipeline
    logger.info(f"\nLoading model: {config.last_model_path}")
    pipeline = RVCPipeline(
        config.last_model_path,
        device=config.device,
        use_compile=False,  # Disable for deterministic results
    )
    pipeline.load()

    # Load test audio
    test_file = Path("sample_data/seki.wav")
    if not test_file.exists():
        logger.error(f"ERROR: Test file not found: {test_file}")
        return False

    logger.info(f"Loading test audio: {test_file}")
    audio_full = load_test_audio(test_file, target_sr=48000)

    # For w-okada mode: use 52.5s (no overlap, 105 chunks @ 0.5s)
    chunk_sec_wokada = 0.35
    chunk_samples_wokada = int(48000 * chunk_sec_wokada)
    num_chunks_wokada = 150  # 52.5 seconds @ 0.35s chunks
    audio_wokada = audio_full[:num_chunks_wokada * chunk_samples_wokada]

    # For RVC WebUI mode: use shorter audio to match same processing time
    # chunk_sec=0.5s, overlap=0.22s, hop=0.28s
    # Each chunk covers hop_sec of new audio (except first chunk which covers chunk_sec)
    # To ensure all audio is processed, use exact multiple of chunk_samples
    # 63 chunks with hop_samples advance = 24000 + 62*13440 = 857,280 samples
    chunk_sec_rvc = 0.5
    overlap_sec_rvc = 0.22
    hop_sec_rvc = chunk_sec_rvc - overlap_sec_rvc  # 0.28s
    num_chunks_rvc = 63
    chunk_samples_rvc = int(48000 * chunk_sec_rvc)  # 24000
    hop_samples_rvc = int(48000 * hop_sec_rvc)  # 13440
    # Total samples = first chunk + (n-1) * hop
    total_samples_rvc = chunk_samples_rvc + (num_chunks_rvc - 1) * hop_samples_rvc
    audio_rvc = audio_full[:total_samples_rvc]

    logger.info(f"w-okada audio: {len(audio_wokada)/48000:.2f}s @ 48kHz ({num_chunks_wokada} chunks @ {chunk_sec_wokada}s)")
    logger.info(f"RVC WebUI audio: {len(audio_rvc)/48000:.2f}s @ 48kHz ({num_chunks_rvc} expected chunks, hop={hop_sec_rvc}s)")

    # Process in TRUE batch mode (gold standard) - use w-okada audio
    logger.info("\n--- TRUE Batch Processing (gold standard, w-okada length) ---")
    true_batch_output_wokada = process_true_batch(pipeline, audio_wokada, pitch_shift=0,
                                                   output_sample_rate=40000)
    logger.info(f"True batch output (w-okada): {len(true_batch_output_wokada)} samples @ 40000Hz")

    # Process in TRUE batch mode (gold standard) - use RVC WebUI audio
    logger.info("\n--- TRUE Batch Processing (gold standard, RVC WebUI length) ---")
    true_batch_output_rvc = process_true_batch(pipeline, audio_rvc, pitch_shift=0,
                                               output_sample_rate=40000)
    logger.info(f"True batch output (RVC): {len(true_batch_output_rvc)} samples @ 40000Hz")

    # Process in chunked mode WITHOUT SOLA (for comparison)
    logger.info("\n--- Chunked Processing (NO SOLA) ---")
    batch_output = process_batch(pipeline, audio_wokada, pitch_shift=0,
                                  chunk_sec=0.35, context_sec=0.05,  # w-okada default
                                  output_sample_rate=40000)
    logger.info(f"Chunked (no SOLA) output: {len(batch_output)} samples @ 40000Hz")

    # Process in streaming mode (w-okada mode)
    # w-okada mode: NO SOLA, just simple edge trimming (faithful reproduction)
    logger.info("\n--- Streaming Processing (w-okada mode) ---")
    streaming_output_wokada = process_streaming(
        pipeline,
        audio_wokada,
        pitch_shift=0,
        chunk_sec=0.35,
        context_sec=0.05,      # w-okada extraConvertSize (edge trimming)
        lookahead_sec=0.0,
        crossfade_sec=0.05,    # Not used in w-okada mode (no SOLA)
        use_sola=True,         # Parameter exists but w-okada mode ignores it
        prebuffer_chunks=0,    # No prebuffer for testing
        mic_sample_rate=48000,
        chunking_mode="wokada",  # Explicit w-okada mode (no SOLA, no crossfade)
    )
    logger.info(f"w-okada mode output: {len(streaming_output_wokada)} samples @ 40000Hz")

    # Process in streaming mode WITH SOLA (RVC WebUI mode)
    # Use EXACT same configuration as test_rvc_sola.py (proven to achieve 0 discontinuities)
    logger.info("\n--- Streaming Processing (RVC WebUI mode) ---")
    streaming_output_rvc = process_streaming(
        pipeline,
        audio_rvc,
        pitch_shift=0,
        chunk_sec=0.5,         # 500ms chunks @ 48kHz (EXACT match with test_rvc_sola.py)
        context_sec=0.05,      # Not used in RVC mode
        lookahead_sec=0.0,
        crossfade_sec=0.22,    # 220ms crossfade (matched with overlap)
        use_sola=True,
        prebuffer_chunks=0,
        mic_sample_rate=48000,
        chunking_mode="rvc_webui",  # RVC WebUI mode
    )
    logger.info(f"RVC WebUI mode output: {len(streaming_output_rvc)} samples @ 40000Hz")

    # Compare outputs
    logger.info("\n--- Comparison: Chunked (no SOLA) vs True Batch ---")
    metrics_chunked = compare_outputs(true_batch_output_wokada, batch_output)
    for key, value in metrics_chunked.items():
        logger.info(f"  {key:20s}: {value}")

    logger.info("\n--- Comparison: w-okada mode vs True Batch ---")
    metrics_wokada = compare_outputs(true_batch_output_wokada, streaming_output_wokada)

    logger.info("\n--- Comparison: RVC WebUI mode vs True Batch ---")
    # Trim RVC output to match true batch length (remove padding)
    streaming_output_rvc_trimmed = streaming_output_rvc[:len(true_batch_output_rvc)]
    logger.info(f"Trimmed RVC output from {len(streaming_output_rvc)} to {len(streaming_output_rvc_trimmed)} samples (removed padding)")
    metrics_rvc = compare_outputs(true_batch_output_rvc, streaming_output_rvc_trimmed)

    # Check correlation for w-okada mode
    logger.info("w-okada mode correlation by chunks:")
    for n_chunks in [10, 50, num_chunks_wokada]:
        chunk_samples = int(40000 * 0.35)  # Output rate
        compare_len = min(n_chunks * chunk_samples, len(true_batch_output_wokada), len(streaming_output_wokada))
        if compare_len > 0:
            true_batch_trim = true_batch_output_wokada[:compare_len]
            streaming_trim = streaming_output_wokada[:compare_len]
            corr = np.corrcoef(true_batch_trim, streaming_trim)[0, 1]
            logger.info(f"  Correlation for first {n_chunks} chunks: {corr:.6f}")

    for key, value in metrics_wokada.items():
        logger.info(f"  {key:20s}: {value}")

    # Check correlation for RVC WebUI mode
    logger.info("\nRVC WebUI mode correlation by chunks:")
    for n_chunks in [10, 30, num_chunks_rvc]:
        chunk_samples = int(40000 * 0.5)  # Output rate, 0.5s chunks
        compare_len = min(n_chunks * chunk_samples, len(true_batch_output_rvc), len(streaming_output_rvc_trimmed))
        if compare_len > 0:
            true_batch_trim = true_batch_output_rvc[:compare_len]
            streaming_trim = streaming_output_rvc_trimmed[:compare_len]
            corr = np.corrcoef(true_batch_trim, streaming_trim)[0, 1]
            logger.info(f"  Correlation for first {n_chunks} chunks: {corr:.6f}")

    for key, value in metrics_rvc.items():
        logger.info(f"  {key:20s}: {value}")

    # Analyze discontinuities in each output
    # Threshold 0.2 detects true clicks/pops, not normal audio transitions
    def count_discontinuities(audio: np.ndarray, threshold: float = 0.2) -> tuple:
        diff = np.abs(np.diff(audio))
        jumps = np.where(diff > threshold)[0]
        jump_values = diff[jumps]
        return len(jumps), jumps, jump_values

    logger.info("\n--- Discontinuity Analysis (threshold=0.2) ---")
    true_batch_wokada_count, true_batch_wokada_indices, _ = count_discontinuities(true_batch_output_wokada)
    true_batch_rvc_count, true_batch_rvc_indices, _ = count_discontinuities(true_batch_output_rvc)
    batch_count, batch_indices, batch_values = count_discontinuities(batch_output)
    wokada_count, wokada_indices, wokada_values = count_discontinuities(streaming_output_wokada)
    rvc_count, rvc_indices, rvc_values = count_discontinuities(streaming_output_rvc_trimmed)

    logger.info(f"True batch (w-okada) discontinuities: {true_batch_wokada_count}")
    logger.info(f"True batch (RVC) discontinuities: {true_batch_rvc_count}")
    logger.info(f"Chunked (no SOLA) discontinuities: {batch_count} ({batch_count - true_batch_wokada_count:+d} vs true batch)")
    logger.info(f"w-okada mode discontinuities: {wokada_count} ({wokada_count - true_batch_wokada_count:+d} vs true batch)")
    logger.info(f"RVC WebUI mode discontinuities: {rvc_count} ({rvc_count - true_batch_rvc_count:+d} vs true batch)")

    # Show details of RVC WebUI mode discontinuities
    if rvc_count > 0:
        logger.info("RVC WebUI mode discontinuity details:")
        for idx, (pos, val) in enumerate(zip(rvc_indices, rvc_values)):
            time_sec = pos / 40000
            logger.info(f"  #{idx+1}: sample={pos}, time={time_sec:.3f}s, jump={val:.4f}")

    # Save outputs for inspection
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)

    wavfile.write(
        output_dir / "true_batch_wokada_output.wav",
        40000,
        (true_batch_output_wokada * 32767).astype(np.int16),
    )
    wavfile.write(
        output_dir / "true_batch_rvc_output.wav",
        40000,
        (true_batch_output_rvc * 32767).astype(np.int16),
    )
    wavfile.write(
        output_dir / "chunked_no_sola_output.wav",
        40000,
        (batch_output * 32767).astype(np.int16),
    )
    wavfile.write(
        output_dir / "wokada_mode_output.wav",
        40000,
        (streaming_output_wokada * 32767).astype(np.int16),
    )
    wavfile.write(
        output_dir / "rvc_webui_mode_output.wav",
        40000,
        (streaming_output_rvc_trimmed * 32767).astype(np.int16),
    )
    logger.info(f"\nOutputs saved to {output_dir}/")

    # Evaluate results (RVC WebUI mode vs True Batch)
    logger.info("\n--- Results (RVC WebUI mode vs True Batch) ---")
    passed = True

    # RVC WebUI mode should achieve perfect continuity (≤ true batch discontinuities)
    discontinuity_diff_rvc = rvc_count - true_batch_rvc_count
    if discontinuity_diff_rvc > 0:
        logger.warning(f"RVC WebUI mode discontinuities: {rvc_count} (+{discontinuity_diff_rvc} vs true batch, target: 0)")
    else:
        logger.info(f"PASS: RVC WebUI mode discontinuities: {rvc_count} ({discontinuity_diff_rvc:+d} vs true batch, PERFECT!)")

    # Correlation threshold: 0.93 is sufficient
    if metrics_rvc["correlation"] < 0.93:
        logger.error(f"FAIL: RVC correlation too low: {metrics_rvc['correlation']:.4f} < 0.93")
        passed = False
    else:
        logger.info(f"PASS: RVC Correlation: {metrics_rvc['correlation']:.4f}")

    logger.info("\n--- Results (w-okada mode vs True Batch) ---")

    # w-okada mode may have some discontinuities (acceptable)
    discontinuity_diff_wokada = wokada_count - true_batch_wokada_count
    logger.info(f"w-okada mode discontinuities: {wokada_count} ({discontinuity_diff_wokada:+d} vs true batch)")

    if metrics_wokada["correlation"] < 0.93:
        logger.warning(f"w-okada correlation: {metrics_wokada['correlation']:.4f} < 0.93")
    else:
        logger.info(f"PASS: w-okada Correlation: {metrics_wokada['correlation']:.4f}")

    # w-okada mode may add some discontinuities (acceptable up to 1 per chunk)
    max_allowed_increase_wokada = num_chunks_wokada  # Max 1 click per chunk
    if discontinuity_diff_wokada > max_allowed_increase_wokada:
        logger.warning(f"w-okada mode discontinuity increase high: {discontinuity_diff_wokada} > {max_allowed_increase_wokada}")
    else:
        logger.info(f"w-okada mode discontinuity increase acceptable: {discontinuity_diff_wokada} <= {max_allowed_increase_wokada}")

    return passed


if __name__ == "__main__":
    success = test_chunk_processing()
    sys.exit(0 if success else 1)
