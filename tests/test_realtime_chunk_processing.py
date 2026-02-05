"""
Integration test: Verify streaming (chunked) output matches batch output.

This test processes a WAV file in chunks using the ACTUAL RealtimeVoiceChanger
logic (not a simulation), and compares it to batch processing.
"""

import logging
import os
import sys
from pathlib import Path
from queue import Empty

import numpy as np
import torch
from scipy.io import wavfile

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rcwx.audio.resample import StatefulResampler, resample
from rcwx.config import RCWXConfig
from rcwx.pipeline.inference import RVCPipeline
from rcwx.pipeline.realtime import RealtimeConfig, RealtimeVoiceChanger
from rcwx.pipeline.realtime_v2 import RealtimeVoiceChangerV2

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
QUICK_TEST = os.getenv("RCWX_QUICK_TEST", "0") == "1"


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


def tune_pipeline_cache(
    pipeline: RVCPipeline,
    chunk_sec: float,
    mic_sample_rate: int = 48000,
    feature_cache_frames: int | None = None,
    f0_cache_frames: int | None = None,
) -> None:
    """Match RealtimeVoiceChanger cache sizing for a given chunk size."""
    cfg = RealtimeConfig(chunk_sec=chunk_sec)
    feature_cache_frames = cfg.feature_cache_frames if feature_cache_frames is None else feature_cache_frames
    f0_cache_frames = cfg.f0_cache_frames if f0_cache_frames is None else f0_cache_frames

    mic_chunk_samples = int(mic_sample_rate * chunk_sec)
    chunk_at_16k = mic_chunk_samples * 16000 / mic_sample_rate
    hubert_frames = max(1, int(round(chunk_at_16k / 320)))
    max_feature_cache = max(2, int(hubert_frames * 0.5))
    f0_frames = hubert_frames * 2
    max_f0_cache = max(4, int(f0_frames * 0.5))

    pipeline._feature_cache_frames = min(feature_cache_frames, max_feature_cache)
    pipeline._f0_cache_frames = min(f0_cache_frames, max_f0_cache)


def build_full_hubert_slices(
    pipeline: RVCPipeline,
    audio_48k: np.ndarray,
    chunk_sec: float,
    context_sec: float,
) -> list[torch.Tensor]:
    """Precompute HuBERT features on full audio and slice per chunk."""
    audio_16k = resample(audio_48k, 48000, 16000)
    audio_tensor = torch.from_numpy(audio_16k).float().unsqueeze(0).to(pipeline.device)

    if pipeline.synthesizer.version == 1:
        output_dim = 256
        output_layer = 9
    else:
        output_dim = 768
        output_layer = 12

    with torch.autocast(device_type=pipeline.device, dtype=pipeline.dtype):
        full_features = pipeline.hubert.extract(
            audio_tensor, output_layer=output_layer, output_dim=output_dim
        )

    total_frames = full_features.shape[1]
    chunk_samples_48k = int(48000 * chunk_sec)
    context_samples_48k = int(48000 * context_sec)

    slices: list[torch.Tensor] = []
    main_pos = 0
    chunk_idx = 0
    while main_pos < len(audio_48k):
        if chunk_idx == 0:
            main_len_48k = min(chunk_samples_48k, len(audio_48k))
            if main_len_48k < chunk_samples_48k:
                break
            start_48k = 0
            length_48k = main_len_48k + context_samples_48k
        else:
            start_48k = max(0, main_pos - context_samples_48k)
            end_48k = min(main_pos + chunk_samples_48k, len(audio_48k))
            expected_len = chunk_samples_48k + context_samples_48k
            if end_48k - start_48k < expected_len:
                break
            length_48k = end_48k - start_48k

        start_16k = int(round(start_48k * 16000 / 48000))
        length_16k = int(round(length_48k * 16000 / 48000))
        frames_needed = max(1, (length_16k - 1) // 320)
        start_frame = max(0, start_16k // 320)
        end_frame = start_frame + frames_needed

        if end_frame <= total_frames:
            slice_feat = full_features[:, start_frame:end_frame, :]
        else:
            slice_feat = full_features[:, start_frame:total_frames, :]
            pad_frames = end_frame - total_frames
            if pad_frames > 0 and slice_feat.shape[1] > 0:
                last = slice_feat[:, -1:, :].repeat(1, pad_frames, 1)
                slice_feat = torch.cat([slice_feat, last], dim=1)

        slices.append(slice_feat)
        main_pos += chunk_samples_48k
        chunk_idx += 1

    return slices


def process_batch_with_full_hubert(
    pipeline: RVCPipeline,
    audio: np.ndarray,
    pitch_shift: int = 0,
    chunk_sec: float = 0.35,
    context_sec: float = 0.05,
    output_sample_rate: int = 48000,
    f0_method: str = "rmvpe",
) -> np.ndarray:
    """Chunked processing but using full-audio HuBERT features per chunk."""
    pipeline.clear_cache()
    tune_pipeline_cache(pipeline, chunk_sec)

    slices = build_full_hubert_slices(pipeline, audio, chunk_sec, context_sec)

    class HubertStub:
        def __init__(self, feats: list[torch.Tensor]) -> None:
            self._feats = feats
            self._idx = 0

        def extract(self, *args, **kwargs):
            if self._idx >= len(self._feats):
                return self._feats[-1]
            feat = self._feats[self._idx]
            self._idx += 1
            return feat

    original_hubert = pipeline.hubert
    pipeline.hubert = HubertStub(slices)  # type: ignore[assignment]

    try:
        return process_batch(
            pipeline,
            audio,
            pitch_shift=pitch_shift,
            chunk_sec=chunk_sec,
            context_sec=context_sec,
            output_sample_rate=output_sample_rate,
            f0_method=f0_method,
            use_feature_cache=False,
        )
    finally:
        pipeline.hubert = original_hubert


def process_true_batch(
    pipeline: RVCPipeline,
    audio: np.ndarray,
    pitch_shift: int = 0,
    output_sample_rate: int = 48000,
    f0_method: str = "rmvpe",
) -> np.ndarray:
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
        f0_method=f0_method,
        index_rate=0.0,
        voice_gate_mode="off",
        use_feature_cache=False,  # Not needed for single inference
        allow_short_input=False,
        pad_mode="batch",
    )

    # Resample output to output rate
    if pipeline.sample_rate != output_sample_rate:
        output = resample(output, pipeline.sample_rate, output_sample_rate)

    return output


def process_batch(
    pipeline: RVCPipeline,
    audio: np.ndarray,
    pitch_shift: int = 0,
    chunk_sec: float = 0.35,
    context_sec: float = 0.05,
    output_sample_rate: int = 48000,
    f0_method: str = "rmvpe",
    use_feature_cache: bool = True,
    synth_min_frames: int | None = None,
    history_sec: float = 0.0,
) -> np.ndarray:
    """Process audio in chunks like streaming, for fair comparison.

    CRITICAL: Use same order as streaming:
    1. Split at 48kHz (mic rate)
    2. Resample chunks to 16kHz (processing rate)
    3. Infer
    4. Resample output to 48kHz (output rate)
    """
    # Clear cache before processing
    pipeline.clear_cache()
    tune_pipeline_cache(pipeline, chunk_sec)

    # Process in chunks matching streaming configuration @ MIC RATE (48kHz)
    chunk_samples_48k = int(48000 * chunk_sec)
    context_samples_48k = int(48000 * context_sec)
    outputs = []
    input_resampler = StatefulResampler(48000, 16000)
    output_resampler = StatefulResampler(pipeline.sample_rate, output_sample_rate)

    main_pos = 0  # Position of current chunk's main section start (no context)
    chunk_idx = 0
    total_output = 0
    while main_pos < len(audio):
        # Determine chunk with context @ 48kHz
        if chunk_idx == 0:
            # First chunk: main only, with reflection padding for left context
            start = 0
            end = min(chunk_samples_48k, len(audio))
            main_chunk = audio[start:end]
            if len(main_chunk) < chunk_samples_48k:
                break

            if context_samples_48k > 0:
                reflect_len = min(context_samples_48k, len(main_chunk))
                reflection = main_chunk[:reflect_len][::-1].copy()
                if len(reflection) < context_samples_48k:
                    reflection = np.pad(
                        reflection,
                        (context_samples_48k - len(reflection), 0),
                        mode="constant",
                    )
                chunk_48k = np.concatenate([reflection, main_chunk])
            else:
                chunk_48k = main_chunk
        else:
            # Subsequent chunks: left_context + main (no padding, match ChunkBuffer)
            start = max(0, main_pos - context_samples_48k)
            end = min(main_pos + chunk_samples_48k, len(audio))
            expected_len = chunk_samples_48k + context_samples_48k
            if end - start < expected_len:
                break
            chunk_48k = audio[start:end]

        # Resample chunk to processing rate (16kHz) - SAME AS STREAMING (stateful)
        chunk_16k = input_resampler.resample_chunk(chunk_48k)

        # Process chunk
        chunk_output = pipeline.infer(
            chunk_16k,
            input_sr=16000,
            pitch_shift=pitch_shift,
            f0_method=f0_method,
            index_rate=0.0,
            voice_gate_mode="off",
            use_feature_cache=use_feature_cache,
            allow_short_input=True,
            pad_mode="none",
            synth_min_frames=synth_min_frames,
            history_sec=history_sec,
        )

        # Resample output to output sample rate (same as streaming)
        if pipeline.sample_rate != output_sample_rate:
            chunk_output = output_resampler.resample_chunk(chunk_output)

        # Trim context from output (including first chunk, matches reflection padding)
        before_trim_len = len(chunk_output)
        if context_sec > 0:
            context_samples_output = int(output_sample_rate * context_sec)
            if len(chunk_output) > context_samples_output:
                chunk_output = chunk_output[context_samples_output:]

        outputs.append(chunk_output)
        total_output += len(chunk_output)

        # Log chunk lengths for debugging
        if chunk_idx < 5 or chunk_idx >= 148:  # First 5 and last 2 chunks
            trim_info = f", trimmed {before_trim_len - len(chunk_output)}" if chunk_idx > 0 else ""
            logger.info(f"Batch chunk {chunk_idx}: input={len(chunk_16k)}@16k, output={len(outputs[-1])}@{output_sample_rate}Hz{trim_info}")

        # Advance main position by chunk size (main-only progression)
        main_pos += chunk_samples_48k
        chunk_idx += 1

    return np.concatenate(outputs)


def process_batch_rvc_webui(
    pipeline: RVCPipeline,
    audio: np.ndarray,
    pitch_shift: int = 0,
    chunk_sec: float = 0.5,
    overlap_sec: float = 0.22,
    output_sample_rate: int = 48000,
    f0_method: str = "rmvpe",
) -> np.ndarray:
    """Process audio in RVC WebUI-style overlapping chunks (baseline for comparison)."""
    pipeline.clear_cache()
    tune_pipeline_cache(pipeline, chunk_sec)

    chunk_samples_48k = int(48000 * chunk_sec)
    hop_samples_48k = int(48000 * (chunk_sec - overlap_sec))
    outputs = []
    input_resampler = StatefulResampler(48000, 16000)
    output_resampler = StatefulResampler(pipeline.sample_rate, output_sample_rate)

    pos = 0
    chunk_idx = 0
    while pos < len(audio):
        end = min(pos + chunk_samples_48k, len(audio))
        chunk_48k = audio[pos:end]
        if len(chunk_48k) < chunk_samples_48k:
            chunk_48k = np.pad(chunk_48k, (0, chunk_samples_48k - len(chunk_48k)))

        chunk_16k = input_resampler.resample_chunk(chunk_48k)
        chunk_output = pipeline.infer(
            chunk_16k,
            input_sr=16000,
            pitch_shift=pitch_shift,
            f0_method=f0_method,
            index_rate=0.0,
            voice_gate_mode="off",
            use_feature_cache=True,
            allow_short_input=True,
            pad_mode="chunk",
        )

        if pipeline.sample_rate != output_sample_rate:
            chunk_output = output_resampler.resample_chunk(chunk_output)

        # Trim overlap to hop size (match streaming length behavior)
        hop_sec = chunk_sec - overlap_sec
        hop_samples_out = int(output_sample_rate * hop_sec)
        if chunk_idx == 0:
            first_chunk_sec = chunk_sec - overlap_sec
            first_chunk_samples = int(output_sample_rate * first_chunk_sec)
            if len(chunk_output) > first_chunk_samples:
                chunk_output = chunk_output[:first_chunk_samples]
        else:
            if len(chunk_output) > hop_samples_out:
                chunk_output = chunk_output[:hop_samples_out]

        outputs.append(chunk_output)

        if chunk_idx < 5 or chunk_idx >= 148:
            logger.info(
                f"RVC batch chunk {chunk_idx}: input={len(chunk_16k)}@16k, output={len(chunk_output)}@{output_sample_rate}Hz"
            )

        pos += hop_samples_48k
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
    use_f0: bool = True,
    f0_method: str = "rmvpe",
    use_feature_cache: bool = True,
    synth_min_frames: int | None = None,
    history_sec: float = 0.0,
    engine: str = "v1",
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
        use_f0=use_f0,
        f0_method=f0_method,
        chunking_mode=chunking_mode,  # Add chunking mode parameter
        rvc_overlap_sec=crossfade_sec,  # RVC mode uses same overlap as crossfade
        index_rate=0.0,
        voice_gate_mode="off",
        use_feature_cache=use_feature_cache,
        synth_min_frames=64 if synth_min_frames is None else synth_min_frames,
        history_sec=history_sec,
    )

    # Create RealtimeVoiceChanger (using real implementation)
    if engine == "v2":
        changer = RealtimeVoiceChangerV2(pipeline, config=rt_config)
    else:
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
    queued_lengths: list[int] = []
    total_queued = 0

    # Feed all input and process chunks as we go
    # This simulates real-time operation where feeding, processing, and output happen concurrently
    pos = 0
    chunks_processed = 0

    def drain_output_queue(tag: str) -> None:
        nonlocal total_queued
        drained = 0
        while True:
            try:
                audio_chunk = changer._output_queue.get_nowait()
            except Empty:
                break
            changer.output_buffer.add(audio_chunk)
            changer._chunks_ready += 1
            queued_lengths.append(len(audio_chunk))
            total_queued += len(audio_chunk)
            drained += len(audio_chunk)
        if drained > 0:
            logger.info(f"[DRAIN] {tag}: drained {drained} samples, total_queued={total_queued}")

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
        drain_output_queue("feed")

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
                drain_output_queue("pad")

        logger.info(f"Fed padding to flush buffer. Total chunks: {chunks_processed}")

    # Process any remaining queued chunks
    while changer.process_next_chunk():
        chunks_processed += 1
        drain_output_queue("flush")

    # Flush final SOLA buffer (critical for RVC WebUI mode)
    changer.flush_final_sola_buffer()
    drain_output_queue("final")

    logger.info(f"Processed {chunks_processed} chunks")
    logger.info(f"Total queued samples: {total_queued}")
    logger.info(f"Output buffer available: {changer.output_buffer.available} samples @ {rt_config.output_sample_rate}Hz")

    # Now retrieve all output from buffer
    total_output_retrieved = 0
    while changer.output_buffer.available > 0:
        output_block = changer.get_output_chunk(output_block_size)
        outputs.append(output_block)
        total_output_retrieved += len(output_block)

    logger.info(f"Retrieved {total_output_retrieved} output samples @ {rt_config.output_sample_rate}Hz")
    expected_out = int(len(audio) * rt_config.output_sample_rate / mic_sample_rate)
    logger.info(f"Expected output (rate-only): {expected_out} samples @ {rt_config.output_sample_rate}Hz")

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
    if QUICK_TEST:
        logger.info("Quick test mode: streaming discontinuity check only (no baseline)")
    run_v2 = os.getenv("RCWX_TEST_V2", "0") == "1"

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
    test_file = Path("sample_data/kakita.wav")
    if not test_file.exists():
        logger.error(f"ERROR: Test file not found: {test_file}")
        return False

    logger.info(f"Loading test audio: {test_file}")
    audio_full = load_test_audio(test_file, target_sr=48000)
    # Shorten to 1/4 length to keep tests fast in quick mode
    if QUICK_TEST:
        audio_full = audio_full[: max(1, len(audio_full) // 4)]

    # For w-okada mode: align chunk_sec to RealtimeConfig rounding (20ms boundary)
    chunk_sec_wokada = RealtimeConfig(chunk_sec=0.35).chunk_sec
    chunk_samples_wokada = int(48000 * chunk_sec_wokada)
    if QUICK_TEST:
        num_chunks_wokada = 2  # ~0.70 seconds @ 0.35s chunks (short test)
    else:
        num_chunks_wokada = min(
            20, max(4, len(audio_full) // chunk_samples_wokada)
        )  # Longer test for continuity
    audio_wokada = audio_full[:num_chunks_wokada * chunk_samples_wokada]

    # For RVC WebUI mode: use shorter audio to match same processing time
    # chunk_sec=0.5s, overlap=0.22s, hop=0.28s
    # Each chunk covers hop_sec of new audio (except first chunk which covers chunk_sec)
    # To ensure all audio is processed, use exact multiple of chunk_samples
    # 63 chunks with hop_samples advance = 24000 + 62*13440 = 857,280 samples
    chunk_sec_rvc = 0.5
    overlap_sec_rvc = 0.22
    hop_sec_rvc = chunk_sec_rvc - overlap_sec_rvc  # 0.28s
    chunk_samples_rvc = int(48000 * chunk_sec_rvc)  # 24000
    hop_samples_rvc = int(48000 * hop_sec_rvc)  # 13440
    if QUICK_TEST:
        num_chunks_rvc = 2  # shorter test
    else:
        max_chunks_rvc = max(
            1, (len(audio_full) - chunk_samples_rvc) // hop_samples_rvc + 1
        )
        num_chunks_rvc = min(20, max(4, max_chunks_rvc))
    # Total samples = first chunk + (n-1) * hop
    total_samples_rvc = chunk_samples_rvc + (num_chunks_rvc - 1) * hop_samples_rvc
    audio_rvc = audio_full[:total_samples_rvc]

    logger.info(f"w-okada audio: {len(audio_wokada)/48000:.2f}s @ 48kHz ({num_chunks_wokada} chunks @ {chunk_sec_wokada}s)")
    logger.info(f"RVC WebUI audio: {len(audio_rvc)/48000:.2f}s @ 48kHz ({num_chunks_rvc} expected chunks, hop={hop_sec_rvc}s)")

    if not QUICK_TEST:
        # True batch (one-shot) for reference
        logger.info("\n--- True Batch Processing (one-shot) ---")
        true_batch_wokada = process_true_batch(
            pipeline,
            audio_wokada,
            pitch_shift=0,
            output_sample_rate=40000,
        )
        logger.info(f"True batch (w-okada audio) output: {len(true_batch_wokada)} samples @ 40000Hz")

        logger.info("\n--- True Batch Processing (one-shot, F0 disabled) ---")
        true_batch_wokada_nof0 = process_true_batch(
            pipeline,
            audio_wokada,
            pitch_shift=0,
            output_sample_rate=40000,
            f0_method="none",
        )
        logger.info(
            f"True batch (w-okada audio, no F0) output: {len(true_batch_wokada_nof0)} samples @ 40000Hz"
        )

        true_batch_rvc = process_true_batch(
            pipeline,
            audio_rvc,
            pitch_shift=0,
            output_sample_rate=40000,
        )
        logger.info(f"True batch (RVC audio) output: {len(true_batch_rvc)} samples @ 40000Hz")

        # Process in chunked mode WITHOUT SOLA (baseline for comparison)
        logger.info("\n--- Chunked Processing (NO SOLA, w-okada baseline) ---")
        batch_output = process_batch(
            pipeline,
            audio_wokada,
            pitch_shift=0,
            chunk_sec=chunk_sec_wokada,
            context_sec=0.05,  # w-okada default
            output_sample_rate=40000,
        )
        logger.info(f"Chunked (no SOLA) output: {len(batch_output)} samples @ 40000Hz")

        logger.info("\n--- Chunked Processing (NO SOLA, history context) ---")
        batch_output_history = process_batch(
            pipeline,
            audio_wokada,
            pitch_shift=0,
            chunk_sec=chunk_sec_wokada,
            context_sec=0.05,  # w-okada default
            output_sample_rate=40000,
            history_sec=0.2,
        )
        logger.info(
            f"Chunked (no SOLA, history=0.2s) output: {len(batch_output_history)} samples @ 40000Hz"
        )

        logger.info("\n--- Chunked Processing (NO SOLA, full-audio HuBERT features) ---")
        batch_output_fullhubert = process_batch_with_full_hubert(
            pipeline,
            audio_wokada,
            pitch_shift=0,
            chunk_sec=chunk_sec_wokada,
            context_sec=0.05,  # w-okada default
            output_sample_rate=40000,
        )
        logger.info(
            f"Chunked (no SOLA, full HuBERT) output: {len(batch_output_fullhubert)} samples @ 40000Hz"
        )

        logger.info("\n--- Chunked Processing (NO SOLA, synth min frames disabled) ---")
        batch_output_nosynthpad = process_batch(
            pipeline,
            audio_wokada,
            pitch_shift=0,
            chunk_sec=chunk_sec_wokada,
            context_sec=0.05,  # w-okada default
            output_sample_rate=40000,
            synth_min_frames=0,
        )
        logger.info(
            f"Chunked (no SOLA, no synth padding) output: {len(batch_output_nosynthpad)} samples @ 40000Hz"
        )

        logger.info("\n--- Chunked Processing (NO SOLA, Feature Cache disabled) ---")
        batch_output_nocache = process_batch(
            pipeline,
            audio_wokada,
            pitch_shift=0,
            chunk_sec=chunk_sec_wokada,
            context_sec=0.05,  # w-okada default
            output_sample_rate=40000,
            use_feature_cache=False,
        )
        logger.info(
            f"Chunked (no SOLA, no feature cache) output: {len(batch_output_nocache)} samples @ 40000Hz"
        )

        logger.info("\n--- Chunked Processing (NO SOLA, F0 disabled) ---")
        batch_output_nof0 = process_batch(
            pipeline,
            audio_wokada,
            pitch_shift=0,
            chunk_sec=chunk_sec_wokada,
            context_sec=0.05,  # w-okada default
            output_sample_rate=40000,
            f0_method="none",
        )
        logger.info(
            f"Chunked (no SOLA, no F0) output: {len(batch_output_nof0)} samples @ 40000Hz"
        )

        logger.info("\n--- RVC WebUI Baseline (batch-style, SOLA disabled) ---")
        batch_output_rvc = process_batch_rvc_webui(
            pipeline,
            audio_rvc,
            pitch_shift=0,
            chunk_sec=0.5,
            overlap_sec=0.22,
            output_sample_rate=40000,
        )
        logger.info(f"RVC baseline (batch, no SOLA) output: {len(batch_output_rvc)} samples @ 40000Hz")

    # Process in streaming mode (w-okada mode, SOLA disabled for baseline alignment)
    logger.info("\n--- Streaming Processing (w-okada mode, NO SOLA) ---")
    streaming_output_wokada_nosola = process_streaming(
        pipeline,
        audio_wokada,
        pitch_shift=0,
        chunk_sec=chunk_sec_wokada,
        context_sec=0.05,      # w-okada extraConvertSize (edge trimming)
        lookahead_sec=0.0,
        crossfade_sec=0.05,
        use_sola=False,
        prebuffer_chunks=0,    # No prebuffer for testing
        mic_sample_rate=48000,
        chunking_mode="wokada",  # Explicit w-okada mode (no SOLA, no crossfade)
    )
    logger.info(f"w-okada mode (no SOLA) output: {len(streaming_output_wokada_nosola)} samples @ 40000Hz")

    if not QUICK_TEST:
        logger.info("\n--- Streaming Processing (w-okada mode, NO SOLA, history context) ---")
        streaming_output_wokada_nosola_history = process_streaming(
            pipeline,
            audio_wokada,
            pitch_shift=0,
            chunk_sec=chunk_sec_wokada,
            context_sec=0.05,
            lookahead_sec=0.0,
            crossfade_sec=0.05,
            use_sola=False,
            prebuffer_chunks=0,
            mic_sample_rate=48000,
            chunking_mode="wokada",
            history_sec=0.2,
        )
        logger.info(
            f"w-okada mode (no SOLA, history=0.2s) output: {len(streaming_output_wokada_nosola_history)} samples @ 40000Hz"
        )

        logger.info("\n--- Streaming Processing (w-okada mode, NO SOLA, synth min frames disabled) ---")
        streaming_output_wokada_nosola_nosynthpad = process_streaming(
            pipeline,
            audio_wokada,
            pitch_shift=0,
            chunk_sec=chunk_sec_wokada,
            context_sec=0.05,
            lookahead_sec=0.0,
            crossfade_sec=0.05,
            use_sola=False,
            prebuffer_chunks=0,
            mic_sample_rate=48000,
            chunking_mode="wokada",
            synth_min_frames=0,
        )
        logger.info(
            f"w-okada mode (no SOLA, no synth padding) output: {len(streaming_output_wokada_nosola_nosynthpad)} samples @ 40000Hz"
        )

        logger.info("\n--- Streaming Processing (w-okada mode, NO SOLA, Feature Cache disabled) ---")
        streaming_output_wokada_nosola_nocache = process_streaming(
            pipeline,
            audio_wokada,
            pitch_shift=0,
            chunk_sec=chunk_sec_wokada,
            context_sec=0.05,
            lookahead_sec=0.0,
            crossfade_sec=0.05,
            use_sola=False,
            prebuffer_chunks=0,
            mic_sample_rate=48000,
            chunking_mode="wokada",
            use_feature_cache=False,
        )
        logger.info(
            f"w-okada mode (no SOLA, no feature cache) output: {len(streaming_output_wokada_nosola_nocache)} samples @ 40000Hz"
        )

    if not QUICK_TEST:
        logger.info("\n--- Streaming Processing (w-okada mode, NO SOLA, F0 disabled) ---")
        streaming_output_wokada_nosola_nof0 = process_streaming(
            pipeline,
            audio_wokada,
            pitch_shift=0,
            chunk_sec=chunk_sec_wokada,
            context_sec=0.05,
            lookahead_sec=0.0,
            crossfade_sec=0.05,
            use_sola=False,
            prebuffer_chunks=0,
            mic_sample_rate=48000,
            chunking_mode="wokada",
            use_f0=False,
            f0_method="none",
        )
        logger.info(
            f"w-okada mode (no SOLA, no F0) output: {len(streaming_output_wokada_nosola_nof0)} samples @ 40000Hz"
        )

    # Process in streaming mode WITH SOLA (w-okada quality mode)
    logger.info("\n--- Streaming Processing (w-okada mode, SOLA enabled) ---")
    streaming_output_wokada = process_streaming(
        pipeline,
        audio_wokada,
        pitch_shift=0,
        chunk_sec=chunk_sec_wokada,
        context_sec=0.05,
        lookahead_sec=0.0,
        crossfade_sec=0.05,
        use_sola=True,
        prebuffer_chunks=0,
        mic_sample_rate=48000,
        chunking_mode="wokada",
    )
    logger.info(f"w-okada mode (SOLA) output: {len(streaming_output_wokada)} samples @ 40000Hz")

    # Process in streaming mode WITHOUT SOLA (RVC baseline alignment)
    logger.info("\n--- Streaming Processing (RVC WebUI mode, NO SOLA) ---")
    streaming_output_rvc_nosola = process_streaming(
        pipeline,
        audio_rvc,
        pitch_shift=0,
        chunk_sec=0.5,
        context_sec=0.05,
        lookahead_sec=0.0,
        crossfade_sec=0.22,
        use_sola=False,
        prebuffer_chunks=0,
        mic_sample_rate=48000,
        chunking_mode="rvc_webui",
    )
    logger.info(f"RVC WebUI mode (no SOLA) output: {len(streaming_output_rvc_nosola)} samples @ 40000Hz")

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

    if run_v2:
        logger.info("\n--- Streaming Processing (w-okada mode, SOLA enabled, V2) ---")
        streaming_output_wokada_v2 = process_streaming(
            pipeline,
            audio_wokada,
            pitch_shift=0,
            chunk_sec=chunk_sec_wokada,
            context_sec=0.05,
            lookahead_sec=0.0,
            crossfade_sec=0.05,
            use_sola=True,
            prebuffer_chunks=0,
            mic_sample_rate=48000,
            chunking_mode="wokada",
            engine="v2",
        )
        logger.info(
            f"w-okada mode (SOLA, V2) output: {len(streaming_output_wokada_v2)} samples @ 40000Hz"
        )

        logger.info("\n--- Streaming Processing (RVC WebUI mode, V2) ---")
        streaming_output_rvc_v2 = process_streaming(
            pipeline,
            audio_rvc,
            pitch_shift=0,
            chunk_sec=0.5,
            context_sec=0.05,
            lookahead_sec=0.0,
            crossfade_sec=0.22,
            use_sola=True,
            prebuffer_chunks=0,
            mic_sample_rate=48000,
            chunking_mode="rvc_webui",
            engine="v2",
        )
        logger.info(
            f"RVC WebUI mode (V2) output: {len(streaming_output_rvc_v2)} samples @ 40000Hz"
        )

    if not QUICK_TEST:
        # Compare outputs (baseline vs streaming, SOLA disabled)
        logger.info("\n--- Comparison: w-okada streaming (NO SOLA) vs chunked baseline ---")
        metrics_wokada = compare_outputs(batch_output, streaming_output_wokada_nosola)

        logger.info("\n--- Comparison: RVC WebUI streaming (NO SOLA) vs batch baseline ---")
        streaming_output_rvc_trimmed = streaming_output_rvc_nosola[: len(batch_output_rvc)]
        logger.info(
            f"Trimmed RVC output from {len(streaming_output_rvc_nosola)} to {len(streaming_output_rvc_trimmed)} samples (baseline length)"
        )
        metrics_rvc = compare_outputs(batch_output_rvc, streaming_output_rvc_trimmed)

        # True batch comparisons (one-shot)
        logger.info("\n--- Comparison: w-okada true batch vs chunked baseline ---")
        metrics_wokada_true_vs_chunked = compare_outputs(true_batch_wokada, batch_output)

        logger.info("\n--- Comparison: w-okada true batch vs streaming (NO SOLA) ---")
        metrics_wokada_true_vs_stream = compare_outputs(true_batch_wokada, streaming_output_wokada_nosola)

        logger.info("\n--- Comparison: w-okada true batch vs chunked (NO F0) ---")
        metrics_wokada_true_vs_chunked_nof0 = compare_outputs(
            true_batch_wokada_nof0, batch_output_nof0
        )

        logger.info("\n--- Comparison: w-okada true batch vs streaming (NO SOLA, NO F0) ---")
        metrics_wokada_true_vs_stream_nof0 = compare_outputs(
            true_batch_wokada_nof0, streaming_output_wokada_nosola_nof0
        )

        logger.info("\n--- Comparison: w-okada true batch vs chunked (NO synth padding) ---")
        metrics_wokada_true_vs_chunked_nosynthpad = compare_outputs(
            true_batch_wokada, batch_output_nosynthpad
        )

        logger.info(
            "\n--- Comparison: w-okada true batch vs streaming (NO SOLA, NO synth padding) ---"
        )
        metrics_wokada_true_vs_stream_nosynthpad = compare_outputs(
            true_batch_wokada, streaming_output_wokada_nosola_nosynthpad
        )

        logger.info("\n--- Comparison: w-okada true batch vs chunked (history context) ---")
        metrics_wokada_true_vs_chunked_history = compare_outputs(
            true_batch_wokada, batch_output_history
        )

        logger.info("\n--- Comparison: w-okada true batch vs streaming (NO SOLA, history context) ---")
        metrics_wokada_true_vs_stream_history = compare_outputs(
            true_batch_wokada, streaming_output_wokada_nosola_history
        )

        logger.info("\n--- Comparison: w-okada true batch vs chunked (full HuBERT) ---")
        metrics_wokada_true_vs_chunked_fullhubert = compare_outputs(
            true_batch_wokada, batch_output_fullhubert
        )

        logger.info("\n--- Comparison: w-okada true batch vs chunked (NO feature cache) ---")
        metrics_wokada_true_vs_chunked_nocache = compare_outputs(
            true_batch_wokada, batch_output_nocache
        )

        logger.info("\n--- Comparison: w-okada true batch vs streaming (NO SOLA, NO feature cache) ---")
        metrics_wokada_true_vs_stream_nocache = compare_outputs(
            true_batch_wokada, streaming_output_wokada_nosola_nocache
        )

        logger.info("\n--- Comparison: RVC true batch vs streaming (NO SOLA) ---")
        metrics_rvc_true_vs_stream = compare_outputs(true_batch_rvc, streaming_output_rvc_nosola)
    else:
        streaming_output_rvc_trimmed = streaming_output_rvc

    if not QUICK_TEST:
        # Check correlation for w-okada mode
        logger.info("w-okada (no SOLA) correlation by chunks:")
        for n_chunks in [10, 50, num_chunks_wokada]:
            chunk_samples = int(40000 * chunk_sec_wokada)  # Output rate
            compare_len = min(
                n_chunks * chunk_samples,
                len(batch_output),
                len(streaming_output_wokada_nosola),
            )
            if compare_len > 0:
                batch_trim = batch_output[:compare_len]
                streaming_trim = streaming_output_wokada_nosola[:compare_len]
                corr = np.corrcoef(batch_trim, streaming_trim)[0, 1]
                logger.info(f"  Correlation for first {n_chunks} chunks: {corr:.6f}")

        for key, value in metrics_wokada.items():
            logger.info(f"  {key:20s}: {value}")

        logger.info("\nTrue batch vs chunked (w-okada):")
        for key, value in metrics_wokada_true_vs_chunked.items():
            logger.info(f"  {key:20s}: {value}")

        logger.info("\nTrue batch vs streaming (w-okada, no SOLA):")
        for key, value in metrics_wokada_true_vs_stream.items():
            logger.info(f"  {key:20s}: {value}")

        logger.info("\nTrue batch vs chunked (w-okada, no F0):")
        for key, value in metrics_wokada_true_vs_chunked_nof0.items():
            logger.info(f"  {key:20s}: {value}")

        logger.info("\nTrue batch vs streaming (w-okada, no SOLA, no F0):")
        for key, value in metrics_wokada_true_vs_stream_nof0.items():
            logger.info(f"  {key:20s}: {value}")

        logger.info("\nTrue batch vs chunked (w-okada, no synth padding):")
        for key, value in metrics_wokada_true_vs_chunked_nosynthpad.items():
            logger.info(f"  {key:20s}: {value}")

        logger.info("\nTrue batch vs streaming (w-okada, no SOLA, no synth padding):")
        for key, value in metrics_wokada_true_vs_stream_nosynthpad.items():
            logger.info(f"  {key:20s}: {value}")

        logger.info("\nTrue batch vs chunked (w-okada, history context):")
        for key, value in metrics_wokada_true_vs_chunked_history.items():
            logger.info(f"  {key:20s}: {value}")

        logger.info("\nTrue batch vs streaming (w-okada, no SOLA, history context):")
        for key, value in metrics_wokada_true_vs_stream_history.items():
            logger.info(f"  {key:20s}: {value}")

        logger.info("\nTrue batch vs chunked (w-okada, full HuBERT):")
        for key, value in metrics_wokada_true_vs_chunked_fullhubert.items():
            logger.info(f"  {key:20s}: {value}")

        logger.info("\nTrue batch vs chunked (w-okada, no feature cache):")
        for key, value in metrics_wokada_true_vs_chunked_nocache.items():
            logger.info(f"  {key:20s}: {value}")

        logger.info("\nTrue batch vs streaming (w-okada, no SOLA, no feature cache):")
        for key, value in metrics_wokada_true_vs_stream_nocache.items():
            logger.info(f"  {key:20s}: {value}")

        # Check correlation for RVC WebUI mode
        logger.info("\nRVC WebUI (no SOLA) correlation by chunks:")
        for n_chunks in [10, 30, num_chunks_rvc]:
            chunk_samples = int(40000 * 0.5)  # Output rate, 0.5s chunks
            compare_len = min(
                n_chunks * chunk_samples,
                len(batch_output_rvc),
                len(streaming_output_rvc_trimmed),
            )
            if compare_len > 0:
                baseline_trim = batch_output_rvc[:compare_len]
                streaming_trim = streaming_output_rvc_trimmed[:compare_len]
                corr = np.corrcoef(baseline_trim, streaming_trim)[0, 1]
                logger.info(f"  Correlation for first {n_chunks} chunks: {corr:.6f}")

        for key, value in metrics_rvc.items():
            logger.info(f"  {key:20s}: {value}")

        logger.info("\nTrue batch vs streaming (RVC, no SOLA):")
        for key, value in metrics_rvc_true_vs_stream.items():
            logger.info(f"  {key:20s}: {value}")

    # Analyze discontinuities in each output
    # Threshold 0.2 detects true clicks/pops, not normal audio transitions
    def count_discontinuities(audio: np.ndarray, threshold: float = 0.2) -> tuple:
        diff = np.abs(np.diff(audio))
        jumps = np.where(diff > threshold)[0]
        jump_values = diff[jumps]
        return len(jumps), jumps, jump_values

    logger.info("\n--- Discontinuity Analysis (threshold=0.2) ---")
    if not QUICK_TEST:
        batch_count, batch_indices, batch_values = count_discontinuities(batch_output)
        batch_rvc_count, batch_rvc_indices, batch_rvc_values = count_discontinuities(batch_output_rvc)
    wokada_nosola_count, wokada_nosola_indices, wokada_nosola_values = count_discontinuities(streaming_output_wokada_nosola)
    wokada_count, wokada_indices, wokada_values = count_discontinuities(streaming_output_wokada)
    rvc_nosola_count, rvc_nosola_indices, rvc_nosola_values = count_discontinuities(streaming_output_rvc_nosola)
    rvc_count, rvc_indices, rvc_values = count_discontinuities(streaming_output_rvc)
    if run_v2:
        wokada_v2_count, _, _ = count_discontinuities(streaming_output_wokada_v2)
        rvc_v2_count, rvc_v2_indices, rvc_v2_values = count_discontinuities(streaming_output_rvc_v2)

    if not QUICK_TEST:
        logger.info(f"Chunked baseline (w-okada, no SOLA) discontinuities: {batch_count}")
        logger.info(f"RVC baseline (batch, no SOLA) discontinuities: {batch_rvc_count}")
        logger.info(f"w-okada streaming (no SOLA) discontinuities: {wokada_nosola_count}")
        logger.info(f"w-okada streaming (SOLA) discontinuities: {wokada_count}")
        logger.info(f"RVC streaming (no SOLA) discontinuities: {rvc_nosola_count}")
        logger.info(f"RVC WebUI mode (SOLA) discontinuities: {rvc_count}")
        if run_v2:
            logger.info(f"w-okada streaming (SOLA, V2) discontinuities: {wokada_v2_count}")
            logger.info(f"RVC WebUI mode (SOLA, V2) discontinuities: {rvc_v2_count}")
    else:
        logger.info(f"w-okada mode discontinuities: {wokada_count}")
        logger.info(f"RVC WebUI mode discontinuities: {rvc_count}")

    # Show details of RVC WebUI mode discontinuities
    if rvc_count > 0:
        logger.info("RVC WebUI mode discontinuity details:")
        for idx, (pos, val) in enumerate(zip(rvc_indices, rvc_values)):
            time_sec = pos / 40000
            logger.info(f"  #{idx+1}: sample={pos}, time={time_sec:.3f}s, jump={val:.4f}")
    if run_v2 and rvc_v2_count > 0:
        logger.info("RVC WebUI mode discontinuity details (V2):")
        for idx, (pos, val) in enumerate(zip(rvc_v2_indices, rvc_v2_values)):
            time_sec = pos / 40000
            logger.info(f"  #{idx+1}: sample={pos}, time={time_sec:.3f}s, jump={val:.4f}")

    # Save outputs for inspection
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)

    if QUICK_TEST:
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
        if run_v2:
            wavfile.write(
                output_dir / "wokada_mode_v2_output.wav",
                40000,
                (streaming_output_wokada_v2 * 32767).astype(np.int16),
            )
            wavfile.write(
                output_dir / "rvc_webui_mode_v2_output.wav",
                40000,
                (streaming_output_rvc_v2 * 32767).astype(np.int16),
            )
    else:
        wavfile.write(
            output_dir / "true_batch_wokada_output.wav",
            40000,
            (true_batch_wokada * 32767).astype(np.int16),
        )
        wavfile.write(
            output_dir / "true_batch_wokada_nof0_output.wav",
            40000,
            (true_batch_wokada_nof0 * 32767).astype(np.int16),
        )
        wavfile.write(
            output_dir / "true_batch_rvc_output.wav",
            40000,
            (true_batch_rvc * 32767).astype(np.int16),
        )
        wavfile.write(
            output_dir / "baseline_wokada_output.wav",
            40000,
            (batch_output * 32767).astype(np.int16),
        )
        wavfile.write(
            output_dir / "baseline_wokada_history_output.wav",
            40000,
            (batch_output_history * 32767).astype(np.int16),
        )
        wavfile.write(
            output_dir / "baseline_wokada_fullhubert_output.wav",
            40000,
            (batch_output_fullhubert * 32767).astype(np.int16),
        )
        wavfile.write(
            output_dir / "baseline_wokada_nosynthpad_output.wav",
            40000,
            (batch_output_nosynthpad * 32767).astype(np.int16),
        )
        wavfile.write(
            output_dir / "baseline_wokada_nocache_output.wav",
            40000,
            (batch_output_nocache * 32767).astype(np.int16),
        )
        wavfile.write(
            output_dir / "baseline_wokada_nof0_output.wav",
            40000,
            (batch_output_nof0 * 32767).astype(np.int16),
        )
        wavfile.write(
            output_dir / "baseline_rvc_output.wav",
            40000,
            (batch_output_rvc * 32767).astype(np.int16),
        )
        wavfile.write(
            output_dir / "wokada_mode_no_sola_output.wav",
            40000,
            (streaming_output_wokada_nosola * 32767).astype(np.int16),
        )
        wavfile.write(
            output_dir / "wokada_mode_no_sola_history_output.wav",
            40000,
            (streaming_output_wokada_nosola_history * 32767).astype(np.int16),
        )
        wavfile.write(
            output_dir / "wokada_mode_no_sola_nosynthpad_output.wav",
            40000,
            (streaming_output_wokada_nosola_nosynthpad * 32767).astype(np.int16),
        )
        wavfile.write(
            output_dir / "wokada_mode_no_sola_nocache_output.wav",
            40000,
            (streaming_output_wokada_nosola_nocache * 32767).astype(np.int16),
        )
        wavfile.write(
            output_dir / "wokada_mode_no_sola_nof0_output.wav",
            40000,
            (streaming_output_wokada_nosola_nof0 * 32767).astype(np.int16),
        )
        wavfile.write(
            output_dir / "wokada_mode_output.wav",
            40000,
            (streaming_output_wokada * 32767).astype(np.int16),
        )
        wavfile.write(
            output_dir / "rvc_webui_mode_no_sola_output.wav",
            40000,
            (streaming_output_rvc_nosola * 32767).astype(np.int16),
        )
        wavfile.write(
            output_dir / "rvc_webui_mode_output.wav",
            40000,
            (streaming_output_rvc * 32767).astype(np.int16),
        )
        if run_v2:
            wavfile.write(
                output_dir / "wokada_mode_v2_output.wav",
                40000,
                (streaming_output_wokada_v2 * 32767).astype(np.int16),
            )
            wavfile.write(
                output_dir / "rvc_webui_mode_v2_output.wav",
                40000,
                (streaming_output_rvc_v2 * 32767).astype(np.int16),
            )
    logger.info(f"\nOutputs saved to {output_dir}/")

    if QUICK_TEST:
        return True

    # Evaluate results (RVC WebUI mode vs baseline)
    logger.info("\n--- Results (RVC WebUI mode vs baseline) ---")
    passed = True

    # RVC WebUI SOLA should not exceed baseline discontinuities
    discontinuity_diff_rvc = rvc_count - batch_rvc_count
    if discontinuity_diff_rvc > 0:
        logger.warning(f"RVC WebUI (SOLA) discontinuities: {rvc_count} (+{discontinuity_diff_rvc} vs baseline, target: 0)")
    else:
        logger.info(f"PASS: RVC WebUI (SOLA) discontinuities: {rvc_count} ({discontinuity_diff_rvc:+d} vs baseline, PERFECT!)")

    # Correlation threshold: 0.93 is sufficient (no SOLA vs batch baseline)
    if metrics_rvc["correlation"] < 0.93:
        logger.error(f"FAIL: RVC (no SOLA) correlation too low: {metrics_rvc['correlation']:.4f} < 0.93")
        passed = False
    else:
        logger.info(f"PASS: RVC (no SOLA) Correlation: {metrics_rvc['correlation']:.4f}")

    logger.info("\n--- Results (w-okada mode vs baseline) ---")

    # w-okada SOLA may have some discontinuities (acceptable)
    discontinuity_diff_wokada = wokada_count - batch_count
    logger.info(f"w-okada (SOLA) discontinuities: {wokada_count} ({discontinuity_diff_wokada:+d} vs baseline)")

    if metrics_wokada["correlation"] < 0.93:
        logger.warning(f"w-okada (no SOLA) correlation: {metrics_wokada['correlation']:.4f} < 0.93")
    else:
        logger.info(f"PASS: w-okada (no SOLA) Correlation: {metrics_wokada['correlation']:.4f}")

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
