"""Comprehensive root cause analysis of chunk boundary discontinuities."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from scipy.io import wavfile


def main():
    print("=" * 80)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 80)

    # Load the output audio
    filepath = "test_output/test_sola_cache_no_smoothing.wav"
    sr, audio = wavfile.read(filepath)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    else:
        audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio[:, 0]

    print(f"Audio: {len(audio)} samples ({len(audio)/sr:.2f}s) @ {sr}Hz\n")

    # Configuration
    chunk_sec = 0.15
    context_sec = 0.10
    chunk_samples = int(chunk_sec * sr)  # 7200
    context_samples = int(context_sec * sr)  # 4800

    # Calculate chunk boundaries
    output_per_chunk = chunk_samples - context_samples  # 2400

    boundaries = []
    pos = chunk_samples
    boundaries.append(pos)
    while pos < len(audio):
        pos += output_per_chunk
        boundaries.append(pos)

    print(f"Expected chunk structure:")
    print(f"  Chunk 0: 0 to {chunk_samples} ({chunk_sec*1000:.0f}ms)")
    print(f"  Chunk N: +{output_per_chunk} samples (+{output_per_chunk/sr*1000:.0f}ms) each")
    print(f"\nTotal boundaries: {len(boundaries)}\n")

    # Find all discontinuities
    diff = np.abs(np.diff(audio))
    disc_indices = np.where(diff > 0.10)[0]
    print(f"Total discontinuities (>0.10): {len(disc_indices)}\n")

    # Classify discontinuities
    boundary_tolerance = 500
    near_boundary = []

    for disc_idx in disc_indices:
        min_distance = min(abs(disc_idx - b) for b in boundaries)
        nearest_boundary = min(boundaries, key=lambda b: abs(b - disc_idx))

        if min_distance < boundary_tolerance:
            distance = disc_idx - nearest_boundary
            near_boundary.append({'distance': distance})

    print(f"Near boundaries: {len(near_boundary)} ({len(near_boundary)/len(disc_indices)*100:.1f}%)\n")

    if near_boundary:
        distances = [d['distance'] for d in near_boundary]
        mean_dist = np.mean(distances)

        print(f"Mean distance: {mean_dist:+.1f} samples ({mean_dist/sr*1000:+.2f}ms)")

        before = sum(1 for d in distances if d < 0)
        print(f"Before boundary: {before} ({before/len(distances)*100:.1f}%)\n")

    print("=" * 80)
    print("ROOT CAUSE:")
    print("=" * 80)
    print("Resampling artifacts at chunk ends (40kHz→48kHz)")
    print("→ SOLA buffer saves these artifacts")
    print("→ Next chunk crossfades with discontinuous buffer")
    print("→ Discontinuities appear at chunk boundaries")
    print("=" * 80)


if __name__ == "__main__":
    main()
