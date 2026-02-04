"""Analyze discontinuity patterns in detail."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from scipy.io import wavfile


def load_audio(path: str) -> tuple[int, np.ndarray]:
    """Load audio file."""
    sr, audio = wavfile.read(path)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    else:
        audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio[:, 0]
    return sr, audio


def analyze_discontinuities(audio: np.ndarray, sr: int, threshold: float = 0.10):
    """Analyze discontinuities in detail."""
    if len(audio) < 2:
        return

    # Compute absolute difference
    diff = np.abs(np.diff(audio))

    # Find discontinuities
    disc_indices = np.where(diff > threshold)[0]

    print(f"  Total discontinuities: {len(disc_indices)}")
    print(f"  Threshold: {threshold}")
    print(f"  Audio length: {len(audio)} samples ({len(audio)/sr:.2f}s)")

    if len(disc_indices) == 0:
        print("  No discontinuities found!")
        return

    # Statistics
    disc_values = diff[disc_indices]
    print(f"\n  Discontinuity magnitude statistics:")
    print(f"    Mean:   {np.mean(disc_values):.6f}")
    print(f"    Median: {np.median(disc_values):.6f}")
    print(f"    Max:    {np.max(disc_values):.6f}")
    print(f"    Min:    {np.min(disc_values):.6f}")
    print(f"    Std:    {np.std(disc_values):.6f}")

    # Temporal distribution
    disc_times = disc_indices / sr
    if len(disc_times) > 1:
        intervals = np.diff(disc_times)
        print(f"\n  Time interval between discontinuities:")
        print(f"    Mean:   {np.mean(intervals)*1000:.2f}ms")
        print(f"    Median: {np.median(intervals)*1000:.2f}ms")
        print(f"    Max:    {np.max(intervals)*1000:.2f}ms")
        print(f"    Min:    {np.min(intervals)*1000:.2f}ms")

    # Histogram of magnitudes
    print(f"\n  Magnitude distribution:")
    bins = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]
    for i in range(len(bins)):
        if i == 0:
            count = np.sum(disc_values < bins[i])
            print(f"    < {bins[i]:.2f}:  {count}")
        else:
            count = np.sum((disc_values >= bins[i-1]) & (disc_values < bins[i]))
            print(f"    {bins[i-1]:.2f}-{bins[i]:.2f}: {count}")
    count = np.sum(disc_values >= bins[-1])
    print(f"    >= {bins[-1]:.2f}: {count}")

    # Show first 10 discontinuities with context
    print(f"\n  First 10 discontinuities with context:")
    for i, idx in enumerate(disc_indices[:10]):
        time_s = idx / sr
        value = diff[idx]
        # Show surrounding samples
        start = max(0, idx - 2)
        end = min(len(audio), idx + 3)
        context = audio[start:end+1]
        print(f"    {i+1}. Sample {idx} ({time_s:.3f}s): diff={value:.6f}")
        print(f"       Context: {context}")


def main():
    print("="*80)
    print("DISCONTINUITY ANALYSIS")
    print("="*80)

    files = [
        ("No SOLA, No Cache", "test_output/test_no_sola,_no_cache.wav"),
        ("SOLA only", "test_output/test_sola_only.wav"),
        ("Cache only", "test_output/test_cache_only.wav"),
        ("SOLA + Cache (current)", "test_output/test_sola_and_cache_(current).wav"),
    ]

    for name, filepath in files:
        print(f"\n{'='*80}")
        print(f"{name}")
        print(f"{'='*80}")

        if not Path(filepath).exists():
            print(f"  File not found: {filepath}")
            continue

        sr, audio = load_audio(filepath)
        analyze_discontinuities(audio, sr, threshold=0.10)

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
