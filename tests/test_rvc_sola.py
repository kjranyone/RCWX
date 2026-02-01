"""Test RVC WebUI style SOLA (overlap chunking + RVC SOLA)."""

import logging
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from rcwx.audio.crossfade import SOLAState, apply_sola_crossfade, flush_sola_buffer
from rcwx.audio.resample import resample
from rcwx.config import RCWXConfig
from rcwx.pipeline.inference import RVCPipeline

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout
)
logger = logging.getLogger(__name__)


def smooth_discontinuities(audio: np.ndarray, threshold: float = 0.2, window_size: int = 100) -> np.ndarray:
    """Apply local smoothing to discontinuities."""
    diff = np.abs(np.diff(audio))
    jump_indices = np.where(diff > threshold)[0]

    if len(jump_indices) == 0:
        return audio

    result = audio.copy()

    for idx in jump_indices:
        # Apply smoothing around the discontinuity
        start = max(0, idx - window_size // 2)
        end = min(len(result), idx + window_size // 2 + 1)

        # Simple moving average
        window = result[start:end]
        smoothed = np.convolve(window, np.ones(5) / 5, mode='same')
        result[start:end] = smoothed

    return result


def process_with_rvc_sola(
    pipeline: RVCPipeline,
    audio: np.ndarray,
    pitch_shift: int = 0,
    chunk_sec: float = 0.35,
    overlap_sec: float = 0.05,
    crossfade_sec: float = 0.05,
    output_sample_rate: int = 40000,
    use_level_normalization: bool = True,
    use_post_smoothing: bool = False,
) -> np.ndarray:
    """
    Process with RVC WebUI style SOLA + level normalization.

    Key differences from w-okada mode:
    1. Overlapping chunks (each chunk overlaps with previous)
    2. SOLA offset search in overlap region
    3. Output WITHOUT tail (tail appears in next chunk via crossfade)
    4. flush_sola_buffer() at the end to output final tail
    5. Level normalization to match chunk boundaries (if enabled)
    """
    pipeline.clear_cache()

    # Parameters @ 48kHz
    chunk_samples = int(48000 * chunk_sec)
    overlap_samples = int(48000 * overlap_sec)
    hop_samples = chunk_samples - overlap_samples

    # SOLA state @ output rate
    sola_state = SOLAState.create(
        int(output_sample_rate * crossfade_sec),
        output_sample_rate,
    )

    logger.info(
        f"RVC SOLA: chunk={chunk_sec}s, overlap={overlap_sec}s, "
        f"crossfade={crossfade_sec}s, hop={hop_samples}@48kHz"
    )
    logger.info(
        f"SOLA: buffer_frame={sola_state.sola_buffer_frame}, "
        f"search_frame={sola_state.sola_search_frame}"
    )

    outputs = []
    pos = 0
    chunk_idx = 0

    while pos < len(audio):
        # Extract overlapping chunk
        chunk_end = min(pos + chunk_samples, len(audio))
        chunk_48k = audio[pos:chunk_end]

        # Pad if last chunk is short
        if len(chunk_48k) < chunk_samples:
            chunk_48k = np.pad(chunk_48k, (0, chunk_samples - len(chunk_48k)))

        # Resample to 16kHz
        chunk_16k = resample(chunk_48k, 48000, 16000)

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

        logger.info(f"Chunk {chunk_idx}: pos={pos}@48k, output={len(chunk_output)}@{output_sample_rate}Hz")

        # Level normalization: match current chunk's crossfade region to previous buffer
        if use_level_normalization and sola_state.sola_buffer is not None and len(chunk_output) >= sola_state.sola_buffer_frame:
            # Compute RMS of previous buffer and current crossfade region
            prev_rms = np.sqrt(np.mean(sola_state.sola_buffer ** 2) + 1e-8)
            current_region = chunk_output[:sola_state.sola_buffer_frame]
            current_rms = np.sqrt(np.mean(current_region ** 2) + 1e-8)

            # Compute gain to match levels
            if current_rms > 1e-6:  # Avoid division by zero
                gain = prev_rms / current_rms
                # Limit gain to reasonable range (0.5 to 2.0)
                gain = np.clip(gain, 0.5, 2.0)

                # Apply gain to entire chunk
                chunk_output = chunk_output * gain
                logger.info(f"  Level normalization: gain={gain:.3f} (prev_rms={prev_rms:.4f}, curr_rms={current_rms:.4f})")

        # Apply RVC WebUI style SOLA
        cf_result = apply_sola_crossfade(
            chunk_output,
            sola_state,
            wokada_mode=False,  # RVC WebUI mode
            context_samples=0,
        )

        # Log SOLA details including correlation if available
        sola_info = f"  After SOLA: output={len(cf_result.audio)}, offset={cf_result.sola_offset}"
        if hasattr(cf_result, 'correlation'):
            sola_info += f", corr={cf_result.correlation:.4f}"
        logger.info(sola_info)

        outputs.append(cf_result.audio)

        # Advance by hop
        pos += hop_samples
        chunk_idx += 1

    # Flush final buffer
    final_buffer = flush_sola_buffer(sola_state)
    if len(final_buffer) > 0:
        logger.info(f"Flushed final buffer: {len(final_buffer)} samples")
        outputs.append(final_buffer)

    result = np.concatenate(outputs)

    # Apply post-processing smoothing if enabled
    if use_post_smoothing:
        logger.info("Applying post-processing smoothing to discontinuities...")
        result = smooth_discontinuities(result, threshold=0.2, window_size=200)

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

    # Trim to exact multiple
    chunk_sec = 0.35
    chunk_samples = int(48000 * chunk_sec)
    num_chunks = 150
    audio = audio[: num_chunks * chunk_samples]

    logger.info(f"Input: {len(audio)} samples @ 48kHz ({len(audio)/48000:.2f}s)")
    logger.info(f"Expected chunks: {num_chunks}")

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

    # Process with RVC SOLA
    logger.info("\n--- SOLA: Extended Feature Cache Blending ---")
    rvc_sola_output = process_with_rvc_sola(
        pipeline,
        audio,
        pitch_shift=0,
        chunk_sec=0.5,
        overlap_sec=0.22,
        crossfade_sec=0.22,
        use_level_normalization=False,
        use_post_smoothing=False,  # 後処理なしで試す
    )
    logger.info(f"Output: {len(rvc_sola_output)} samples @ 40kHz")

    # Analyze
    def count_discontinuities(audio: np.ndarray, threshold: float = 0.2):
        diff = np.abs(np.diff(audio))
        jump_indices = np.where(diff > threshold)[0]
        jump_values = diff[jump_indices]
        return jump_indices, jump_values

    rvc_sola_jump_indices, rvc_sola_jump_values = count_discontinuities(rvc_sola_output)
    logger.info(f"\nRVC SOLA discontinuities (threshold=0.2): {len(rvc_sola_jump_indices)}")

    # Show details of discontinuities
    if len(rvc_sola_jump_indices) > 0:
        logger.info(f"Discontinuity locations (sample indices):")
        for idx, (pos, val) in enumerate(zip(rvc_sola_jump_indices, rvc_sola_jump_values)):
            time_sec = pos / 40000
            logger.info(f"  #{idx+1}: sample={pos}, time={time_sec:.3f}s, jump={val:.4f}")

    if true_batch is not None:
        true_batch_jump_indices, _ = count_discontinuities(true_batch)
        logger.info(f"True batch discontinuities: {len(true_batch_jump_indices)}")
        logger.info(f"Difference: {len(rvc_sola_jump_indices) - len(true_batch_jump_indices):+d}")

        # Correlation
        min_len = min(len(rvc_sola_output), len(true_batch))
        corr = np.corrcoef(rvc_sola_output[:min_len], true_batch[:min_len])[0, 1]
        logger.info(f"Correlation: {corr:.6f}")

    # Save
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    wavfile.write(
        output_dir / "rvc_sola_output.wav",
        40000,
        (rvc_sola_output * 32767).astype(np.int16),
    )
    logger.info(f"\nSaved to {output_dir}/rvc_sola_output.wav")
    logger.info("**Please listen to this file!**")


if __name__ == "__main__":
    main()
