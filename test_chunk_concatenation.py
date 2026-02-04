"""Analyze discontinuities at chunk concatenation points."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from scipy.io import wavfile


def main():
    print("="*80)
    print("CHUNK CONCATENATION ANALYSIS")
    print("="*80)

    # Load the output audio
    filepath = "test_output/test_sola_cache_no_smoothing.wav"
    sr, audio = wavfile.read(filepath)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    else:
        audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio[:, 0]

    print(f"Audio: {len(audio)} samples ({len(audio)/sr:.2f}s)\n")

    # Calculate chunk boundaries
    chunk_sec = 0.15
    context_sec = 0.10
    chunk_samples = int(chunk_sec * sr)  # 7200
    context_samples = int(context_sec * sr)  # 4800
    output_per_chunk = chunk_samples - context_samples  # 2400

    boundaries = []
    pos = chunk_samples  # First chunk output: 7200
    boundaries.append(pos)
    while pos < len(audio):
        pos += output_per_chunk  # Each subsequent chunk adds 2400
        boundaries.append(pos)

    print(f"Chunk boundaries: {len(boundaries)}")
    print(f"First 10: {boundaries[:10]}\n")

    # Analyze each boundary
    diff = np.abs(np.diff(audio))
    boundary_tolerance = 500  # ~10ms

    print("Boundary discontinuity analysis:")
    print(f"{'Boundary':<10} {'Sample':<10} {'Region':<20} {'Max Disc':<12} {'Position':<15}")
    print("-"*80)

    for i, boundary in enumerate(boundaries[:20]):
        # Check region around boundary: 10ms before to 5ms after
        start = max(0, boundary - boundary_tolerance)
        end = min(len(diff), boundary + 250)

        if end > start:
            region_diff = diff[start:end]
            max_disc = np.max(region_diff)
            max_pos_in_region = np.argmax(region_diff)
            max_pos = start + max_pos_in_region
            distance = max_pos - boundary

            region_str = f"{start}-{end}"
            pos_str = f"{distance:+d} ({distance/sr*1000:+.2f}ms)"

            print(f"{i:<10} {boundary:<10} {region_str:<20} {max_disc:<12.4f} {pos_str:<15}")

            if max_disc > 0.10:
                # Show detailed context
                context_start = max(0, max_pos - 5)
                context_end = min(len(audio), max_pos + 6)
                context = audio[context_start:context_end]
                print(f"  → Context: {context}")

    # Find all discontinuities > 0.10
    disc_indices = np.where(diff > 0.10)[0]
    print(f"\n{'='*80}")
    print(f"Total discontinuities (>0.10): {len(disc_indices)}")

    # Count how many are near boundaries
    near_boundary = 0
    for disc_idx in disc_indices:
        for boundary in boundaries:
            if abs(disc_idx - boundary) < boundary_tolerance:
                near_boundary += 1
                break

    print(f"Near boundaries (±{boundary_tolerance/sr*1000:.1f}ms): {near_boundary} ({near_boundary/len(disc_indices)*100:.1f}%)")

    # Analyze the distance distribution for near-boundary discontinuities
    distances = []
    for disc_idx in disc_indices:
        min_distance = min(abs(disc_idx - b) for b in boundaries)
        if min_distance < boundary_tolerance:
            # Find the nearest boundary
            nearest_boundary = min(boundaries, key=lambda b: abs(b - disc_idx))
            distance = disc_idx - nearest_boundary
            distances.append(distance)

    if distances:
        print(f"\nDistance distribution (near-boundary):")
        print(f"  Mean:   {np.mean(distances):+.1f} samples ({np.mean(distances)/sr*1000:+.2f}ms)")
        print(f"  Median: {np.median(distances):+.1f} samples ({np.median(distances)/sr*1000:+.2f}ms)")
        print(f"  Min:    {np.min(distances):+d} samples ({np.min(distances)/sr*1000:+.2f}ms)")
        print(f"  Max:    {np.max(distances):+d} samples ({np.max(distances)/sr*1000:+.2f}ms)")

        # Histogram
        before = sum(1 for d in distances if d < 0)
        after = sum(1 for d in distances if d >= 0)
        print(f"\n  Before boundary: {before} ({before/len(distances)*100:.1f}%)")
        print(f"  After boundary:  {after} ({after/len(distances)*100:.1f}%)")

    print("\n" + "="*80)
    print("CONCLUSION:")
    print("="*80)
    if near_boundary > 0 and distances:
        avg_distance_ms = np.mean(distances) / sr * 1000
        if avg_distance_ms < -2:
            print(f"Discontinuities occur {abs(avg_distance_ms):.1f}ms BEFORE boundaries")
            print("→ Problem: Each chunk's END has discontinuities")
            print("→ Likely cause: Resampling artifacts at chunk end")
            print("→ Or: SOLA buffer saved from discontinuous region")
        elif avg_distance_ms > 2:
            print(f"Discontinuities occur {avg_distance_ms:.1f}ms AFTER boundaries")
            print("→ Problem: Chunk concatenation or SOLA crossfade")
            print("→ Likely cause: SOLA offset search not finding optimal position")
        else:
            print("Discontinuities evenly distributed around boundaries")
            print("→ Problem: Complex interaction between multiple factors")
    print("="*80)


if __name__ == "__main__":
    main()
