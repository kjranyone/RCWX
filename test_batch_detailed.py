"""Detailed debug of batch processing to find the bug."""

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

    # Use 1-second 440Hz tone
    sr = 48000
    t = np.arange(sr) / sr
    audio_48k = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    print(f"Input: {len(audio_48k)} samples @ 48kHz = {len(audio_48k) / sr:.2f}s")
    print(f"Input RMS: {np.sqrt(np.mean(audio_48k**2)):.6f}")

    # Test 1: 48kHz input with input_sr=48000
    print("\n=== Test 1: input_sr=48000 ===")
    output1 = pipeline.infer(
        audio_48k,
        input_sr=48000,
        pitch_shift=0,
        f0_method="fcpe",
        index_rate=0.0,
        voice_gate_mode="off",
        use_feature_cache=False,
    )
    print(
        f"Output: {len(output1)} samples @ {pipeline.sample_rate}Hz = {len(output1) / pipeline.sample_rate:.2f}s"
    )
    print(f"Output RMS: {np.sqrt(np.mean(output1**2)):.6f}")
    print(f"Expected ratio: 48000/16000 * {pipeline.sample_rate}/48000 = 3.0x")

    # Test 2: 16kHz input with input_sr=16000
    print("\n=== Test 2: input_sr=16000 ===")
    audio_16k = resample(audio_48k, 48000, 16000)
    output2 = pipeline.infer(
        audio_16k,
        input_sr=16000,
        pitch_shift=0,
        f0_method="fcpe",
        index_rate=0.0,
        voice_gate_mode="off",
        use_feature_cache=False,
    )
    print(
        f"Output: {len(output2)} samples @ {pipeline.sample_rate}Hz = {len(output2) / pipeline.sample_rate:.2f}s"
    )
    print(f"Output RMS: {np.sqrt(np.mean(output2**2)):.6f}")
    print(f"Expected ratio: 1.0x")

    # Compare
    print("\n=== Comparison ===")
    print(f"Output1 length: {len(output1)}")
    print(f"Output2 length: {len(output2)}")
    print(f"Length ratio (1/2): {len(output1) / len(output2):.3f}x")

    if len(output1) == len(output2):
        min_len = min(len(output1), len(output2))
        correlation = np.corrcoef(output1[:min_len], output2[:min_len])[0, 1]
        print(f"Correlation (1 vs 2): {correlation:.6f}")
        print(
            f"Correlation check: {'PASS' if abs(correlation) > 0.9 else 'FAIL (correlation too low)'}"
        )


if __name__ == "__main__":
    main()
