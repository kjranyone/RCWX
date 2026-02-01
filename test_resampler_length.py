"""Test StatefulResampler output length consistency."""

import numpy as np
from rcwx.audio.resample import StatefulResampler

def test_resampler_output_length():
    """Test that StatefulResampler produces consistent output lengths."""

    # Test case 1: 48kHz -> 16kHz (input chunk = 4800 samples @ 48kHz)
    print("=" * 60)
    print("Test 1: 48kHz -> 16kHz (4800 input samples)")
    print("=" * 60)

    resampler1 = StatefulResampler(48000, 16000)
    expected_output = int(4800 * 16000 / 48000)  # 1600

    for i in range(10):
        chunk = np.random.randn(4800).astype(np.float32)
        output = resampler1.resample_chunk(chunk)
        diff = len(output) - expected_output
        print(f"Chunk {i+1}: input={len(chunk)}, output={len(output)}, expected={expected_output}, diff={diff}")
        if diff != 0:
            print(f"  ERROR: Output length mismatch!")

    # Test case 2: 40kHz -> 48kHz (matching actual inference)
    print("\n" + "=" * 60)
    print("Test 2: 40kHz -> 48kHz (20000 input samples)")
    print("=" * 60)

    resampler2 = StatefulResampler(40000, 48000)
    expected_output = int(20000 * 48000 / 40000)  # 24000

    for i in range(10):
        chunk = np.random.randn(20000).astype(np.float32)
        output = resampler2.resample_chunk(chunk)
        diff = len(output) - expected_output
        print(f"Chunk {i+1}: input={len(chunk)}, output={len(output)}, expected={expected_output}, diff={diff}")
        if diff != 0:
            print(f"  ERROR: Output length mismatch!")

    # Test case 3: Realistic inference scenario
    # Input: 4800 @ 48kHz -> 1600 @ 16kHz -> infer -> 20000 @ 40kHz -> 24000 @ 48kHz
    print("\n" + "=" * 60)
    print("Test 3: Full pipeline simulation (10 chunks)")
    print("=" * 60)

    input_resampler = StatefulResampler(48000, 16000)
    output_resampler = StatefulResampler(40000, 48000)

    mic_chunk_samples = 4800  # 100ms @ 48kHz
    processing_chunk_samples = 1600  # 100ms @ 16kHz
    model_output_samples = 20000  # ~125ms @ 40kHz (variable due to RVC)

    total_output = 0

    for i in range(10):
        # Simulate input processing
        mic_chunk = np.random.randn(mic_chunk_samples).astype(np.float32)
        processing_chunk = input_resampler.resample_chunk(mic_chunk)

        # Simulate inference (output may vary slightly)
        # In real code, this varies: out=17640 to out=18515 from logs
        model_output = np.random.randn(model_output_samples).astype(np.float32)

        # Resample back to 48kHz
        output_chunk = output_resampler.resample_chunk(model_output)
        total_output += len(output_chunk)

        expected_processing = int(mic_chunk_samples * 16000 / 48000)
        expected_output_chunk = int(model_output_samples * 48000 / 40000)

        print(f"Chunk {i+1}:")
        print(f"  Input: {len(mic_chunk)} -> {len(processing_chunk)} (expected {expected_processing}, diff={len(processing_chunk) - expected_processing})")
        print(f"  Output: {len(model_output)} -> {len(output_chunk)} (expected {expected_output_chunk}, diff={len(output_chunk) - expected_output_chunk})")

    print(f"\nTotal output samples: {total_output}")
    print(f"Expected per chunk: {int(model_output_samples * 48000 / 40000)}")
    print(f"Expected total (10 chunks): {10 * int(model_output_samples * 48000 / 40000)}")

if __name__ == "__main__":
    test_resampler_output_length()
