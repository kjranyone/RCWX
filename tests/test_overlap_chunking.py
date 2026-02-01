"""Test overlapping chunk processing (RVC WebUI style)."""

import logging
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from rcwx.audio.resample import resample
from rcwx.config import RCWXConfig
from rcwx.pipeline.inference import RVCPipeline

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout
)
logger = logging.getLogger(__name__)


def process_with_overlap(
    pipeline: RVCPipeline,
    audio: np.ndarray,
    pitch_shift: int = 0,
    chunk_sec: float = 0.35,
    overlap_sec: float = 0.05,
    output_sample_rate: int = 40000,
) -> np.ndarray:
    """
    Process audio with overlapping chunks and linear crossfade.

    This is simpler than SOLA but more reliable:
    - Each chunk overlaps with the previous one
    - Crossfade in the overlap region
    - No complex offset search
    """
    pipeline.clear_cache()

    # Parameters @ 48kHz (input rate)
    chunk_samples = int(48000 * chunk_sec)
    overlap_samples = int(48000 * overlap_sec)
    hop_samples = chunk_samples - overlap_samples  # How much we advance each time

    # Crossfade window (sin^2 - same as RVC)
    overlap_samples_output = int(output_sample_rate * overlap_sec)
    t = np.linspace(0.0, 1.0, overlap_samples_output, dtype=np.float32)
    fade_in = np.sin(0.5 * np.pi * t) ** 2
    fade_out = 1.0 - fade_in

    outputs = []
    prev_tail = None  # Tail of previous chunk (for crossfade)

    pos = 0
    chunk_idx = 0

    while pos < len(audio):
        # Extract chunk
        chunk_end = min(pos + chunk_samples, len(audio))
        chunk_48k = audio[pos:chunk_end]

        # Pad if last chunk is short
        if len(chunk_48k) < chunk_samples:
            chunk_48k = np.pad(chunk_48k, (0, chunk_samples - len(chunk_48k)))

        # Resample to 16kHz
        chunk_16k = resample(chunk_48k, 48000, 16000)

        # Process
        chunk_output = pipeline.infer(
            chunk_16k,
            input_sr=16000,
            pitch_shift=pitch_shift,
            f0_method="rmvpe",
            index_rate=0.0,
            voice_gate_mode="off",
            use_feature_cache=False,  # Disable - it causes more harm than good!
        )

        # Resample to output rate
        if pipeline.sample_rate != output_sample_rate:
            chunk_output = resample(chunk_output, pipeline.sample_rate, output_sample_rate)

        logger.info(
            f"Chunk {chunk_idx}: pos={pos}@48k, output={len(chunk_output)}@{output_sample_rate}Hz"
        )

        # Crossfade with previous chunk
        if prev_tail is not None and len(chunk_output) >= overlap_samples_output:
            # Crossfade: blend prev_tail with beginning of current chunk
            crossfade_len = min(len(prev_tail), overlap_samples_output, len(chunk_output))

            blended = (
                prev_tail[:crossfade_len] * fade_out[:crossfade_len]
                + chunk_output[:crossfade_len] * fade_in[:crossfade_len]
            )

            # Output: blended part + remainder of current chunk
            outputs.append(blended)
            outputs.append(chunk_output[crossfade_len:])
        else:
            # First chunk: output as-is
            outputs.append(chunk_output)

        # Save tail for next crossfade (last overlap_samples)
        if len(chunk_output) >= overlap_samples_output:
            prev_tail = chunk_output[-overlap_samples_output:].copy()
        else:
            prev_tail = chunk_output.copy()

        # Advance by hop_samples (not chunk_samples!)
        pos += hop_samples
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

    # Trim to exact multiple of chunk for fair comparison
    chunk_sec = 0.35
    chunk_samples = int(48000 * chunk_sec)
    num_chunks = 150
    audio = audio[: num_chunks * chunk_samples]

    logger.info(f"Input: {len(audio)} samples @ 48kHz ({len(audio)/48000:.2f}s)")
    logger.info(f"Chunks: {num_chunks}, chunk_sec={chunk_sec}, overlap_sec=0.20 (sin^2 window)")

    # Load true batch reference
    true_batch_file = Path("test_output/true_batch_output.wav")
    if true_batch_file.exists():
        _, true_batch = wavfile.read(true_batch_file)
        if true_batch.dtype == np.int16:
            true_batch = true_batch.astype(np.float32) / 32768.0
        logger.info(f"True batch reference: {len(true_batch)} samples @ 40kHz")
    else:
        logger.warning("True batch reference not found, skipping comparison")
        true_batch = None

    # Process with overlap
    logger.info("\n--- Overlap Processing ---")
    overlap_output = process_with_overlap(
        pipeline, audio, pitch_shift=0, chunk_sec=0.35, overlap_sec=0.20
    )
    logger.info(f"Overlap output: {len(overlap_output)} samples @ 40kHz")

    # Analyze discontinuities
    def count_discontinuities(audio: np.ndarray, threshold: float = 0.2) -> int:
        diff = np.abs(np.diff(audio))
        jumps = np.where(diff > threshold)[0]
        return len(jumps)

    overlap_jumps = count_discontinuities(overlap_output)
    logger.info(f"\nOverlap discontinuities (threshold=0.2): {overlap_jumps}")

    if true_batch is not None:
        true_batch_jumps = count_discontinuities(true_batch)
        logger.info(f"True batch discontinuities: {true_batch_jumps}")
        logger.info(f"Difference: {overlap_jumps - true_batch_jumps:+d}")

        # Correlation
        min_len = min(len(overlap_output), len(true_batch))
        corr = np.corrcoef(overlap_output[:min_len], true_batch[:min_len])[0, 1]
        logger.info(f"Correlation with true batch: {corr:.6f}")

    # Save output
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    wavfile.write(
        output_dir / "overlap_output.wav",
        40000,
        (overlap_output * 32767).astype(np.int16),
    )
    logger.info(f"\nOutput saved to {output_dir}/overlap_output.wav")
    logger.info("Please listen to it and compare with true_batch_output.wav")


if __name__ == "__main__":
    main()
