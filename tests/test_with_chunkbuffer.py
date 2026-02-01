"""Test using ChunkBuffer directly (exactly like RealtimeVoiceChanger)."""

import logging
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from rcwx.audio.buffer import ChunkBuffer
from rcwx.audio.crossfade import SOLAState, apply_sola_crossfade
from rcwx.audio.resample import resample
from rcwx.config import RCWXConfig
from rcwx.pipeline.inference import RVCPipeline

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout
)
logger = logging.getLogger(__name__)


def process_with_chunkbuffer(
    pipeline: RVCPipeline,
    audio: np.ndarray,
    pitch_shift: int = 0,
    chunk_sec: float = 0.35,
    context_sec: float = 0.05,
    crossfade_sec: float = 0.05,
    use_sola: bool = True,
    mic_sample_rate: int = 48000,
    output_sample_rate: int = 40000,
) -> np.ndarray:
    """
    Process using ChunkBuffer - EXACTLY like RealtimeVoiceChanger.

    This ensures we replicate the exact same chunking logic.
    """
    pipeline.clear_cache()

    # Create ChunkBuffer (same as realtime.py)
    chunk_buffer = ChunkBuffer(
        chunk_samples=int(mic_sample_rate * chunk_sec),
        crossfade_samples=int(output_sample_rate * crossfade_sec),
        context_samples=int(mic_sample_rate * context_sec),
        lookahead_samples=0,
    )

    # Create SOLA state if enabled
    if use_sola:
        sola_state = SOLAState.create(
            int(output_sample_rate * crossfade_sec),
            output_sample_rate,
        )
    else:
        sola_state = None

    logger.info(f"ChunkBuffer: chunk={chunk_buffer.chunk_samples}, context={chunk_buffer.context_samples}")
    logger.info(f"SOLA: {use_sola}, crossfade_samples={int(output_sample_rate * crossfade_sec)}")

    # Feed audio in 20ms blocks (like real audio callbacks)
    block_size = int(mic_sample_rate * 0.02)
    pos = 0
    while pos < len(audio):
        block = audio[pos : pos + block_size]
        if len(block) < block_size:
            block = np.pad(block, (0, block_size - len(block)))
        chunk_buffer.add_input(block)
        pos += block_size

    # Process all chunks
    outputs = []
    chunk_idx = 0

    while chunk_buffer.has_chunk():
        # Get chunk (includes context for chunk 2+)
        chunk_48k = chunk_buffer.get_chunk()

        logger.info(f"Chunk {chunk_idx}: input_len={len(chunk_48k)}@48kHz")

        # Resample to 16kHz
        chunk_16k = resample(chunk_48k, mic_sample_rate, 16000)

        # Infer
        chunk_output = pipeline.infer(
            chunk_16k,
            input_sr=16000,
            pitch_shift=pitch_shift,
            f0_method="rmvpe",
            index_rate=0.0,
            voice_gate_mode="off",
            use_feature_cache=True,
        )

        # Resample to output rate
        if pipeline.sample_rate != output_sample_rate:
            chunk_output = resample(chunk_output, pipeline.sample_rate, output_sample_rate)

        # Apply SOLA or manual trim
        if use_sola and sola_state is not None:
            context_samples_output = 0
            if chunk_idx > 0 and context_sec > 0:
                context_samples_output = int(output_sample_rate * context_sec)

            cf_result = apply_sola_crossfade(
                chunk_output,
                sola_state,
                wokada_mode=True,
                context_samples=context_samples_output,
            )
            chunk_output = cf_result.audio
            logger.info(f"  After SOLA: output_len={len(chunk_output)}, offset={cf_result.sola_offset}")
        else:
            # Manual trim
            if chunk_idx > 0 and context_sec > 0:
                context_samples_output = int(output_sample_rate * context_sec)
                if len(chunk_output) > context_samples_output:
                    chunk_output = chunk_output[context_samples_output:]
                    logger.info(f"  After manual trim: output_len={len(chunk_output)}")

        outputs.append(chunk_output)
        chunk_idx += 1

    return np.concatenate(outputs)


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

    # Trim to exact multiple
    chunk_sec = 0.35
    chunk_samples = int(48000 * chunk_sec)
    num_chunks = 150
    audio = audio[: num_chunks * chunk_samples]

    logger.info(f"Input: {len(audio)} samples @ 48kHz ({len(audio)/48000:.2f}s)")
    logger.info(f"Expected chunks: {num_chunks}")

    # Load true batch reference
    true_batch_file = Path("test_output/true_batch_output.wav")
    _, true_batch = wavfile.read(true_batch_file)
    if true_batch.dtype == np.int16:
        true_batch = true_batch.astype(np.float32) / 32768.0
    logger.info(f"True batch: {len(true_batch)} samples @ 40kHz")

    # Process with ChunkBuffer + SOLA
    logger.info("\n--- Processing with ChunkBuffer (SOLA ON) ---")
    chunkbuf_output = process_with_chunkbuffer(
        pipeline,
        audio,
        pitch_shift=0,
        chunk_sec=0.35,
        context_sec=0.05,
        crossfade_sec=0.05,
        use_sola=True,
    )
    logger.info(f"Output: {len(chunkbuf_output)} samples @ 40kHz")

    # Analyze
    def count_discontinuities(audio: np.ndarray, threshold: float = 0.2) -> int:
        diff = np.abs(np.diff(audio))
        return len(np.where(diff > threshold)[0])

    true_batch_jumps = count_discontinuities(true_batch)
    chunkbuf_jumps = count_discontinuities(chunkbuf_output)

    logger.info(f"\nDiscontinuities (threshold=0.2):")
    logger.info(f"  True batch: {true_batch_jumps}")
    logger.info(f"  ChunkBuffer: {chunkbuf_jumps} ({chunkbuf_jumps - true_batch_jumps:+d})")

    # Correlation
    min_len = min(len(chunkbuf_output), len(true_batch))
    corr = np.corrcoef(chunkbuf_output[:min_len], true_batch[:min_len])[0, 1]
    logger.info(f"  Correlation: {corr:.6f}")

    # Save
    output_dir = Path("test_output")
    wavfile.write(
        output_dir / "chunkbuffer_output.wav",
        40000,
        (chunkbuf_output * 32767).astype(np.int16),
    )
    logger.info(f"\nSaved to {output_dir}/chunkbuffer_output.wav")
    logger.info("**Please listen to this file!**")


if __name__ == "__main__":
    main()
