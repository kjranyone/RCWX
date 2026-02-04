"""Detailed analysis of discontinuities near chunk boundaries."""

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


def analyze_boundary_pattern(audio: np.ndarray, sr: int, chunk_sec: float = 0.15, threshold: float = 0.10):
    """Analyze discontinuity patterns near chunk boundaries."""
    if len(audio) < 2:
        return

    # Compute absolute difference
    diff = np.abs(np.diff(audio))

    # Find discontinuities
    disc_indices = np.where(diff > threshold)[0]

    print(f"Total discontinuities (>= {threshold}): {len(disc_indices)}")
    print(f"Audio length: {len(audio)} samples ({len(audio)/sr:.2f}s)")

    if len(disc_indices) == 0:
        return

    # Calculate chunk boundaries
    # For wokada mode with context_sec=0.10:
    # - First chunk output: chunk_sec * sr = 7200 samples
    # - Subsequent chunks: (chunk_sec - context_sec) * sr = 2400 samples
    chunk_samples = int(chunk_sec * sr)
    context_samples = int(0.10 * sr)
    output_per_chunk = chunk_samples - context_samples

    chunk_boundaries = []
    pos = chunk_samples  # First chunk
    chunk_boundaries.append(pos)
    while pos < len(audio):
        pos += output_per_chunk
        chunk_boundaries.append(pos)

    # Analyze discontinuities relative to chunk boundaries
    boundary_tolerance = int(0.01 * sr)  # 10ms = 480 samples

    near_boundary = []
    far_from_boundary = []

    for disc_idx in disc_indices:
        # Find nearest boundary
        distances = [abs(disc_idx - b) for b in chunk_boundaries]
        min_distance = min(distances)
        nearest_boundary = chunk_boundaries[distances.index(min_distance)]

        if min_distance < boundary_tolerance:
            near_boundary.append({
                'idx': disc_idx,
                'boundary': nearest_boundary,
                'distance': disc_idx - nearest_boundary,
                'magnitude': diff[disc_idx],
            })
        else:
            far_from_boundary.append({
                'idx': disc_idx,
                'boundary': nearest_boundary,
                'distance': disc_idx - nearest_boundary,
                'magnitude': diff[disc_idx],
            })

    print(f"\nNear chunk boundaries (±{boundary_tolerance/sr*1000:.1f}ms): {len(near_boundary)} ({len(near_boundary)/len(disc_indices)*100:.1f}%)")
    print(f"Far from boundaries: {len(far_from_boundary)} ({len(far_from_boundary)/len(disc_indices)*100:.1f}%)")

    # Analyze distance distribution for near-boundary discontinuities
    if near_boundary:
        distances = [d['distance'] for d in near_boundary]
        print(f"\nNear-boundary distance statistics (samples):")
        print(f"  Mean:   {np.mean(distances):+.1f}")
        print(f"  Median: {np.median(distances):+.1f}")
        print(f"  Min:    {np.min(distances):+d}")
        print(f"  Max:    {np.max(distances):+d}")

        # Group by distance range
        before = sum(1 for d in distances if d < 0)
        after = sum(1 for d in distances if d >= 0)
        print(f"\n  Before boundary: {before} ({before/len(distances)*100:.1f}%)")
        print(f"  After boundary:  {after} ({after/len(distances)*100:.1f}%)")

        # Show first 10
        print(f"\nFirst 10 near-boundary discontinuities:")
        for i, d in enumerate(near_boundary[:10]):
            time_s = d['idx'] / sr
            dist_ms = d['distance'] / sr * 1000
            boundary_time = d['boundary'] / sr
            print(f"  {i+1}. Sample {d['idx']} ({time_s:.3f}s)")
            print(f"     Boundary: {d['boundary']} ({boundary_time:.3f}s), Distance: {d['distance']:+d} ({dist_ms:+.2f}ms)")
            print(f"     Magnitude: {d['magnitude']:.4f}")

    # Analyze far-from-boundary discontinuities
    if far_from_boundary:
        print(f"\nFar-from-boundary discontinuities:")
        print(f"  First 5:")
        for i, d in enumerate(far_from_boundary[:5]):
            time_s = d['idx'] / sr
            dist_ms = d['distance'] / sr * 1000
            boundary_time = d['boundary'] / sr
            print(f"  {i+1}. Sample {d['idx']} ({time_s:.3f}s)")
            print(f"     Nearest boundary: {d['boundary']} ({boundary_time:.3f}s), Distance: {d['distance']:+d} ({dist_ms:+.2f}ms)")
            print(f"     Magnitude: {d['magnitude']:.4f}")

    # Distance distribution histogram (text-based)
    if near_boundary:
        print(f"\nDistance distribution histogram:")
        distances_ms = [d['distance']/sr*1000 for d in near_boundary]
        # Create bins
        bins = np.linspace(-10, 10, 21)  # -10ms to +10ms, 1ms bins
        hist, _ = np.histogram(distances_ms, bins=bins)

        print(f"  Distance (ms)  | Count")
        print(f"  " + "-"*30)
        for i, count in enumerate(hist):
            if count > 0:
                bin_center = (bins[i] + bins[i+1]) / 2
                bar = '#' * int(count * 40 / max(hist))
                print(f"  {bin_center:+6.1f}        | {count:3d} {bar}")


def main():
    print("="*80)
    print("CHUNK BOUNDARY DISCONTINUITY ANALYSIS")
    print("="*80)

    filepath = "test_output/test_sola_cache_no_smoothing.wav"

    if not Path(filepath).exists():
        print(f"File not found: {filepath}")
        return

    sr, audio = load_audio(filepath)
    analyze_boundary_pattern(audio, sr, chunk_sec=0.15, threshold=0.10)

    print("\n" + "="*80)
    print("HYPOTHESIS:")
    print("="*80)
    print("If discontinuities cluster just AFTER boundaries:")
    print("  → Context trimming or SOLA crossfade issue")
    print("If discontinuities cluster just BEFORE boundaries:")
    print("  → Output buffer concatenation issue")
    print("If discontinuities are evenly distributed:")
    print("  → Resampling or model output artifacts")
    print("="*80)


if __name__ == "__main__":
    main()
