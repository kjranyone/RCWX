"""Test to debug why batch and streaming have different results."""

import logging
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).parent))

from rcwx.audio.resample import resample
from rcwx.device import get_device
from rcwx.pipeline.inference import RVCPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Load pipeline
    device_str = get_device("xpu")

    pipeline = RVCPipeline(
        model_path="sample_data/hogaraka/hogarakav2.pth",
        device=device_str,
        dtype="float16",
        use_compile=False,
    )
    pipeline.load()

    # Load test audio (5 seconds)
    sr, audio = wavfile.read("sample_data/seki.wav")
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    if len(audio.shape) > 1:
        audio = audio[:, 0]

    test_samples = int(5 * 48000)
    audio = audio[:test_samples].astype(np.float32)

    if sr != 48000:
        audio = resample(audio, sr, 48000)

    logger.info(f"Input: {len(audio)} samples @ 48kHz = {len(audio) / 48000:.2f}s")
    logger.info(f"Input RMS: {np.sqrt(np.mean(audio**2)):.6f}")
    logger.info(f"Input min/max: {audio.min():.4f} / {audio.max():.4f}")

    # Run batch inference
    logger.info("\nRunning batch inference...")
    batch_output = pipeline.infer(
        audio,
        input_sr=16000,
        pitch_shift=0,
        f0_method="fcpe",
        index_rate=0.0,
        voice_gate_mode="off",
        use_feature_cache=False,
    )

    logger.info(
        f"Batch output: {len(batch_output)} samples @ {pipeline.sample_rate}Hz = {len(batch_output) / pipeline.sample_rate:.2f}s"
    )
    logger.info(f"Batch RMS: {np.sqrt(np.mean(batch_output**2)):.6f}")
    logger.info(f"Batch min/max: {batch_output.min():.4f} / {batch_output.max():.4f}")

    # Check if there's phase inversion or other issues
    # Convert to same sample rate for comparison
    if pipeline.sample_rate != 48000:
        batch_48k = resample(batch_output, pipeline.sample_rate, 48000)
    else:
        batch_48k = batch_output

    # Align lengths
    min_len = min(len(audio), len(batch_48k))
    audio_trim = audio[:min_len]
    batch_trim = batch_48k[:min_len]

    correlation = np.corrcoef(audio_trim, batch_trim)[0, 1]
    energy_ratio = np.sum(batch_trim**2) / np.sum(audio_trim**2)

    logger.info(f"\nComparison (input vs batch):")
    logger.info(f"  Correlation: {correlation:.6f}")
    logger.info(f"  Energy ratio (batch/input): {energy_ratio:.6f}")


if __name__ == "__main__":
    main()
