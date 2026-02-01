"""Test real-time processing SOLA (w-okada mode)."""

import logging
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from rcwx.audio.buffer import ChunkBuffer, OutputBuffer
from rcwx.audio.crossfade import SOLAState, apply_sola_crossfade, flush_sola_buffer
from rcwx.audio.resample import resample
from rcwx.config import RCWXConfig
from rcwx.pipeline.inference import RVCPipeline

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout
)
logger = logging.getLogger(__name__)


def process_with_realtime_sola(
    pipeline: RVCPipeline,
    audio: np.ndarray,
    pitch_shift: int = 0,
    chunk_sec: float = 0.5,
    context_sec: float = 0.05,
    crossfade_sec: float = 0.05,
    output_sample_rate: int = 40000,
) -> np.ndarray:
    """
    Process using real-time processing logic (w-okada mode).

    This mimics the actual RealtimeVoiceChanger behavior:
    1. ChunkBuffer with left context
    2. w-okada mode SOLA
    3. Context trimming after inference
    """
    pipeline.clear_cache()

    # Chunk buffer @ 48kHz
    mic_sr = 48000
    chunk_samples = int(mic_sr * chunk_sec)
    context_samples = int(mic_sr * context_sec)
    crossfade_samples = int(output_sample_rate * crossfade_sec)

    chunk_buffer = ChunkBuffer(
        chunk_samples=chunk_samples,
        crossfade_samples=crossfade_samples,
        context_samples=context_samples,
        lookahead_samples=0,
    )

    # SOLA state @ output rate
    sola_state = SOLAState.create(
        int(output_sample_rate * crossfade_sec),
        output_sample_rate,
    )

    logger.info(
        f"Realtime SOLA: chunk={chunk_sec}s, context={context_sec}s, "
        f"crossfade={crossfade_sec}s"
    )
    logger.info(
        f"ChunkBuffer: chunk_samples={chunk_buffer.chunk_samples}, "
        f"context_samples={chunk_buffer.context_samples}"
    )
    logger.info(
        f"SOLA: buffer_frame={sola_state.sola_buffer_frame}, "
        f"search_frame={sola_state.sola_search_frame}"
    )

    outputs = []
    chunk_idx = 0

    # Feed audio in small increments (simulating real-time input)
    input_chunk_size = int(mic_sr * 0.02)  # 20ms increments
    pos = 0

    while pos < len(audio) or chunk_buffer.has_chunk():
        # Add input
        if pos < len(audio):
            end = min(pos + input_chunk_size, len(audio))
            chunk_buffer.add_input(audio[pos:end])
            pos = end

        # Process available chunks
        while chunk_buffer.has_chunk():
            # Get chunk (includes context on left for chunk 1+)
            chunk_48k = chunk_buffer.get_chunk()

            # Resample to 16kHz
            chunk_16k = resample(chunk_48k, mic_sr, 16000)

            # Infer (WITH feature cache for continuity)
            chunk_output = pipeline.infer(
                chunk_16k,
                input_sr=16000,
                pitch_shift=pitch_shift,
                f0_method="rmvpe",
                index_rate=0.0,
                voice_gate_mode="off",
                use_feature_cache=True,  # Enable for chunk continuity
            )

            # Resample to output rate
            if pipeline.sample_rate != output_sample_rate:
                chunk_output = resample(chunk_output, pipeline.sample_rate, output_sample_rate)

            logger.info(f"Chunk {chunk_idx}: output={len(chunk_output)}@{output_sample_rate}Hz")

            # Apply w-okada mode SOLA
            cf_result = apply_sola_crossfade(
                chunk_output,
                sola_state,
                wokada_mode=True,  # w-okada mode (real-time)
                context_samples=int(output_sample_rate * context_sec),
            )

            # Log SOLA details
            sola_info = f"  After SOLA: output={len(cf_result.audio)}, offset={cf_result.sola_offset}"
            if hasattr(cf_result, 'correlation'):
                sola_info += f", corr={cf_result.correlation:.4f}"
            logger.info(sola_info)

            outputs.append(cf_result.audio)
            chunk_idx += 1

    # Flush final buffer
    final_buffer = flush_sola_buffer(sola_state)
    if len(final_buffer) > 0:
        logger.info(f"Flushed final buffer: {len(final_buffer)} samples")
        outputs.append(final_buffer)

    result = np.concatenate(outputs)
    return result


def main():
    # Load config and pipeline
    config = RCWXConfig.load()
    pipeline = RVCPipeline(
        config.last_model_path, device=config.device, use_compile=False
    )
    pipeline.load()

    # Load test audio
    test_file = Path("sample_data/seki.wav")
    sr, audio = wavfile.read(test_file)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0

    # Trim to 52.5s (same as other tests)
    chunk_sec = 0.5
    chunk_samples = int(48000 * chunk_sec)
    num_chunks = 105  # 52.5s
    audio = audio[: num_chunks * chunk_samples]

    logger.info(f"Input: {len(audio)} samples @ 48kHz ({len(audio)/48000:.2f}s)")
    logger.info(f"Expected chunks: ~{num_chunks}")

    # Load true batch reference
    true_batch_file = Path("test_output/true_batch_output.wav")
    if true_batch_file.exists():
        _, true_batch = wavfile.read(true_batch_file)
        if true_batch.dtype == np.int16:
            true_batch = true_batch.astype(np.float32) / 32768.0
        logger.info(f"True batch: {len(true_batch)} samples @ 40kHz")
    else:
        logger.warning("True batch reference not found")
        true_batch = None

    # Process with real-time SOLA (w-okada mode)
    logger.info("\n--- Real-time SOLA (w-okada mode with default settings) ---")
    realtime_output = process_with_realtime_sola(
        pipeline,
        audio,
        pitch_shift=0,
        chunk_sec=0.5,
        context_sec=0.05,  # Default setting
        crossfade_sec=0.05,
    )
    logger.info(f"Output: {len(realtime_output)} samples @ 40kHz")

    # Analyze
    def count_discontinuities(audio: np.ndarray, threshold: float = 0.2):
        diff = np.abs(np.diff(audio))
        jump_indices = np.where(diff > threshold)[0]
        jump_values = diff[jump_indices]
        return jump_indices, jump_values

    rt_jump_indices, rt_jump_values = count_discontinuities(realtime_output)
    logger.info(f"\nReal-time SOLA discontinuities (threshold=0.2): {len(rt_jump_indices)}")

    # Show details of discontinuities
    if len(rt_jump_indices) > 0:
        logger.info(f"Discontinuity locations (sample indices):")
        for idx, (pos, val) in enumerate(zip(rt_jump_indices, rt_jump_values)):
            time_sec = pos / 40000
            logger.info(f"  #{idx+1}: sample={pos}, time={time_sec:.3f}s, jump={val:.4f}")

    if true_batch is not None:
        true_batch_jump_indices, _ = count_discontinuities(true_batch)
        logger.info(f"True batch discontinuities: {len(true_batch_jump_indices)}")
        logger.info(f"Difference: {len(rt_jump_indices) - len(true_batch_jump_indices):+d}")

        # Correlation
        min_len = min(len(realtime_output), len(true_batch))
        corr = np.corrcoef(realtime_output[:min_len], true_batch[:min_len])[0, 1]
        logger.info(f"Correlation: {corr:.6f}")

    # Save
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    wavfile.write(
        output_dir / "realtime_sola_output.wav",
        40000,
        (realtime_output * 32767).astype(np.int16),
    )
    logger.info(f"\nSaved to {output_dir}/realtime_sola_output.wav")
    logger.info("**Please listen to this file!**")


if __name__ == "__main__":
    main()
