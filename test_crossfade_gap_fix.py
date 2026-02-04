"""Unit test for crossfade gap fix - no model required."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from rcwx.audio.crossfade import SOLAState, apply_sola_crossfade


def test_wokada_gap_fix():
    """Test that w-okada mode doesn't create gaps between chunks."""
    print("=" * 80)
    print("CROSSFADE GAP FIX TEST (Unit Test - No Model)")
    print("=" * 80)

    # Simulate output parameters
    output_sr = 48000
    chunk_sec = 0.15
    context_sec = 0.10
    crossfade_sec = 0.05

    chunk_samples = int(output_sr * chunk_sec)  # 7200 samples (150ms)
    context_samples = int(output_sr * context_sec)  # 4800 samples (100ms)
    crossfade_samples = int(output_sr * crossfade_sec)  # 2400 samples (50ms)

    print(f"\nConfiguration:")
    print(f"  Sample rate: {output_sr}Hz")
    print(f"  Chunk size: {chunk_sec}s = {chunk_samples} samples")
    print(f"  Context size: {context_sec}s = {context_samples} samples")
    print(f"  Crossfade size: {crossfade_sec}s = {crossfade_samples} samples")

    # Create SOLA state
    state = SOLAState.create(crossfade_samples, sample_rate=output_sr)

    # Simulate 5 chunks of inference output
    # Each chunk has: [context | main]
    total_samples_per_chunk = context_samples + chunk_samples  # 12000 samples (250ms)

    print(f"\n" + "=" * 80)
    print("PROCESSING CHUNKS")
    print("=" * 80)

    output_lengths = []
    total_output = 0

    for i in range(5):
        # Generate fake audio: [context | main]
        chunk_audio = np.random.randn(total_samples_per_chunk).astype(np.float32) * 0.1

        # Apply SOLA crossfade (w-okada mode)
        result = apply_sola_crossfade(
            chunk_audio,
            state,
            wokada_mode=True,
            context_samples=context_samples if i > 0 else 0,  # First chunk has no context to trim
        )

        output_length = len(result.audio)
        output_lengths.append(output_length)
        total_output += output_length

        expected_length = chunk_samples  # Should be chunk_samples after context trim
        diff = output_length - expected_length

        status = "OK" if abs(diff) <= 10 else "!!"  # Allow 10 sample tolerance
        print(f"  {status} Chunk {i}: output={output_length} samples "
              f"({output_length/output_sr*1000:.1f}ms), "
              f"expected={expected_length} ({expected_length/output_sr*1000:.1f}ms), "
              f"diff={diff:+d}")

    print(f"\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    # Expected total output
    # Chunk 0: full chunk (no context to trim) = 7200 + 4800 = 12000 samples
    # Chunk 1-4: chunk_samples after context trim = 7200 samples each
    expected_chunk0 = total_samples_per_chunk  # First chunk keeps context
    expected_others = chunk_samples * 4

    # After our fix: expect full output without tail removal
    expected_total_after_fix = expected_chunk0 + expected_others

    print(f"\nTotal output: {total_output} samples ({total_output/output_sr*1000:.0f}ms)")
    print(f"Expected (with fix): {expected_total_after_fix} samples ({expected_total_after_fix/output_sr*1000:.0f}ms)")

    gap_samples = expected_total_after_fix - total_output
    gap_ms = gap_samples / output_sr * 1000

    print(f"\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    if abs(gap_samples) <= 50:  # Allow small tolerance
        print(f"[SUCCESS] SUCCESS: No significant gaps!")
        print(f"   Gap: {abs(gap_samples)} samples ({abs(gap_ms):.1f}ms) - within tolerance")
        print(f"   Chunks are properly connected without gaps.")
        return True
    else:
        print(f"[FAILED] ISSUE: Gaps detected!")
        print(f"   Gap: {abs(gap_samples)} samples ({abs(gap_ms):.1f}ms)")
        print(f"   Expected {expected_total_after_fix}, got {total_output}")
        return False


if __name__ == "__main__":
    success = test_wokada_gap_fix()
    sys.exit(0 if success else 1)
