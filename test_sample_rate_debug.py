"""Debug to trace exact sample rates through the pipeline."""

import logging
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).parent))

from rcwx.audio.resample import resample
from rcwx.device import get_device
from rcwx.pipeline.inference import RVCPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
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

    # Simple 1-second 440Hz tone @ 48kHz
    sr = 48000
    t = np.arange(sr) / sr
    audio_48k = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    logger.info(f"=== Test 1: Direct 48kHz input ===")
    logger.info(f"Input: {len(audio_48k)} samples @ 48kHz = {len(audio_48k)/sr:.2f}s")

    # Test with input_sr=48000
    output1 = pipeline.infer(
        audio_48k,
        input_sr=48000,  # Pass 48kHz directly
        pitch_shift=0,
        f0_method="fcpe",
        index_rate=0.0,
        voice_gate_mode="off",
        use_feature_cache=False,
    )
    logger.info(f"Output: {len(output1)} samples @ {pipeline.sample_rate}Hz = {len(output1)/pipeline.sample_rate:.2f}s")

    # Resample to 16kHz first
    audio_16k = resample(audio_48k, 48000, 16000)
    logger.info(f"\n=== Test 2: Resampled 16kHz input ===")
    logger.info(f"Resampled: {len(audio_16k)} samples @ 16kHz = {len(audio_16k)/16000:.2f}s")

    # Test with input_sr=16000
    output2 = pipeline.infer(
        audio_16k,
        input_sr=16000,
        pitch_shift=0,
        f0_method="fcpe",
        index_rate=0.0,
        voice_gate_mode="off",
        use_feature_cache=False,
    )
    logger.info(f"Output: {len(output2)} samples @ {pipeline.sample_rate}Hz = {len(output2)/pipeline.sample_rate:.2f}s")

    # Compare
    if len(output1) == len(output2):
        correlation = np.corrcoef(output1, output2)[0, 1]
        logger.info(f"\nComparison:")
        logger.info(f"  Lengths match: {len(output1)} == len(output2)}")
        logger.info(f"  Correlation: {correlation:.6f}")
        logger.info(f"  Correlation check: {'PASS' if abs(correlation) > 0.9 else 'FAIL (correlation too low)'}")

if __name__ == "__main__":
    main()
