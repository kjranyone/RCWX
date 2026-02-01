"""Minimal test to check pipeline behavior."""

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

    # Simple 1-second tone
    audio_48k = (0.3 * np.sin(2 * np.pi * 440 * np.arange(48000) / 48000)).astype(np.float32)

    print(f"Input: {len(audio_48k)} samples @ 48kHz = {len(audio_48k) / 48000:.2f}s")

    # Test 1: input_sr=48000
    print("\nTest 1: input_sr=48000")
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

    # Test 2: input_sr=16000
    print("\nTest 2: input_sr=16000")
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


if __name__ == "__main__":
    main()
