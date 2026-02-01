"""Simple test to check pipeline output length."""

import torch
import numpy as np
from scipy.io import wavfile

from rcwx.audio.resample import resample
from rcwx.device import get_device
from rcwx.pipeline.inference import RVCPipeline


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

    # Create simple test input: 5 seconds of 440Hz tone
    sr = 48000
    t = np.arange(5 * sr) / sr
    audio_48k = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    print(f"Input: {len(audio_48k)} samples @ 48kHz = {len(audio_48k) / sr:.2f}s")

    # Resample to 16kHz (as pipeline does)
    audio_16k = resample(audio_48k, 48000, 16000)
    print(f"Resampled to 16kHz: {len(audio_16k)} samples = {len(audio_16k) / 16000:.2f}s")

    # Expected output length at 40kHz
    expected_40k = len(audio_16k) * 40000 / 16000
    print(f"Expected output @ 40kHz: {expected_40k:.0f} samples = {expected_40k / 40000:.2f}s")

    # Run inference
    output = pipeline.infer(
        audio_16k,
        input_sr=16000,
        pitch_shift=0,
        f0_method="fcpe",
        index_rate=0.0,
        voice_gate_mode="off",
        use_feature_cache=False,
    )

    print(f"Actual output: {len(output)} samples = {len(output) / 40000:.2f}s @ 40kHz")
    print(f"Ratio: {len(output) / expected_40k:.2f}x")
    print(f"Min/Max: {output.min():.4f} / {output.max():.4f}")
    print(f"RMS: {np.sqrt(np.mean(output**2)):.6f}")


if __name__ == "__main__":
    main()
