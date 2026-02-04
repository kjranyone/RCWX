"""Analyze discontinuities vs chunk boundaries."""

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


def analyze_vs_chunks(audio: np.ndarray, sr: int, chunk_sec: float = 0.15, threshold: float = 0.10):
    """Analyze discontinuities vs chunk boundaries."""
    if len(audio) < 2:
        return

    # Compute absolute difference
    diff = np.abs(np.diff(audio))

    # Find discontinuities
    disc_indices = np.where(diff > threshold)[0]

    print(f"  Total discontinuities (>= {threshold}): {len(disc_indices)}")
    print(f"  Audio length: {len(audio)} samples ({len(audio)/sr:.2f}s)")
    print(f"  Chunk size: {chunk_sec}s = {int(chunk_sec * sr)} samples")

    if len(disc_indices) == 0:
        print("  No discontinuities found!")
        return

    # Calculate chunk boundaries (output chunks after context trimming)
    # chunk_sec = 0.15, context_sec = 0.10
    # Output per chunk = (chunk_sec - context_sec) * sr = 0.05 * 48000 = 2400 samples
    # Actually, for wokada mode:
    # Chunk 0: full chunk (7200 samples)
    # Chunk 1+: chunk - context (7200 - 4800 = 2400 samples)

    # Simplified: assume uniform 2400 sample output per chunk (except first)
    chunk_output_samples = int(chunk_sec * sr) - int(0.10 * sr)  # 7200 - 4800 = 2400

    # Chunk boundaries (approximate)
    chunk_boundaries = []
    pos = int(chunk_sec * sr)  # First chunk: 7200
    chunk_boundaries.append(pos)
    while pos < len(audio):
        pos += chunk_output_samples
        chunk_boundaries.append(pos)

    print(f"\n  Chunk boundaries (approximate): {len(chunk_boundaries)} boundaries")
    print(f"  First 10 boundaries: {chunk_boundaries[:10]}")

    # Find discontinuities near chunk boundaries
    boundary_tolerance = int(0.01 * sr)  # 10ms = 480 samples
    near_boundary_count = 0

    for disc_idx in disc_indices:
        for boundary in chunk_boundaries:
            if abs(disc_idx - boundary) < boundary_tolerance:
                near_boundary_count += 1
                break

    print(f"\n  Discontinuities near chunk boundaries (±{boundary_tolerance} samples = ±{boundary_tolerance/sr*1000:.1f}ms):")
    print(f"    Count: {near_boundary_count} / {len(disc_indices)} ({near_boundary_count/len(disc_indices)*100:.1f}%)")

    # Show first 10 discontinuities with nearest boundary
    print(f"\n  First 10 discontinuities:")
    for i, disc_idx in enumerate(disc_indices[:10]):
        time_s = disc_idx / sr
        value = diff[disc_idx]
        # Find nearest boundary
        nearest_boundary = min(chunk_boundaries, key=lambda b: abs(b - disc_idx))
        distance = disc_idx - nearest_boundary
        distance_ms = distance / sr * 1000

        print(f"    {i+1}. Sample {disc_idx} ({time_s:.3f}s): diff={value:.4f}")
        print(f"       Nearest boundary: {nearest_boundary} (distance: {distance:+d} samples = {distance_ms:+.2f}ms)")


def main():
    print("="*80)
    print("CHUNK BOUNDARY ANALYSIS")
    print("="*80)

    filepath = "test_output/test_sola_cache_no_smoothing.wav"

    if not Path(filepath).exists():
        print(f"  File not found: {filepath}")
        print("  Run test_sola_cache_only.py first")
        return

    sr, audio = load_audio(filepath)
    analyze_vs_chunks(audio, sr, chunk_sec=0.15, threshold=0.10)

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
