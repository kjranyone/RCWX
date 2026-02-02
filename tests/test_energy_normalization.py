"""Test energy normalization effect on envelope correlation."""

import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).parent.parent))


def compute_envelope_correlation(audio1: np.ndarray, audio2: np.ndarray, window_ms: float = 20.0, sr: int = 48000) -> float:
    """Compute phase-invariant envelope correlation."""
    min_len = min(len(audio1), len(audio2))
    audio1 = audio1[:min_len]
    audio2 = audio2[:min_len]

    window_samples = int(sr * window_ms / 1000)
    n_windows = min_len // window_samples

    env1 = np.zeros(n_windows)
    env2 = np.zeros(n_windows)

    for i in range(n_windows):
        start = i * window_samples
        end = start + window_samples
        env1[i] = np.sqrt(np.mean(audio1[start:end] ** 2))
        env2[i] = np.sqrt(np.mean(audio2[start:end] ** 2))

    # Pearson correlation
    env1_centered = env1 - np.mean(env1)
    env2_centered = env2 - np.mean(env2)

    numerator = np.sum(env1_centered * env2_centered)
    denominator = np.sqrt(np.sum(env1_centered ** 2) * np.sum(env2_centered ** 2))

    if denominator < 1e-8:
        return 0.0

    return float(numerator / denominator)


def main():
    test_output_dir = Path(__file__).parent.parent / "test_output" / "mode_comparison"

    batch_wav = test_output_dir / "true_batch.wav"
    wokada_wav = test_output_dir / "wokada_output.wav"

    if not batch_wav.exists() or not wokada_wav.exists():
        print("Test outputs not found. Run test_chunking_modes_comparison.py first.")
        return

    sr1, batch = wavfile.read(batch_wav)
    sr2, wokada = wavfile.read(wokada_wav)

    # Normalize to float
    if batch.dtype == np.int16:
        batch = batch.astype(np.float32) / 32768.0
    if wokada.dtype == np.int16:
        wokada = wokada.astype(np.float32) / 32768.0

    # Align lengths
    min_len = min(len(batch), len(wokada))
    batch = batch[:min_len]
    wokada = wokada[:min_len]

    # Current metrics
    batch_rms = np.sqrt(np.mean(batch ** 2))
    wokada_rms = np.sqrt(np.mean(wokada ** 2))
    energy_ratio = wokada_rms / batch_rms

    print("=" * 60)
    print("Energy Normalization Analysis")
    print("=" * 60)
    print(f"Batch RMS:  {batch_rms:.4f}")
    print(f"Wokada RMS: {wokada_rms:.4f}")
    print(f"Energy ratio: {energy_ratio:.4f}")

    # Before normalization
    corr_before = compute_envelope_correlation(batch, wokada, sr=sr1)
    print(f"\nEnvelope correlation (before normalization): {corr_before:.4f}")

    # After normalization (scale wokada to match batch energy)
    wokada_normalized = wokada / energy_ratio
    corr_after = compute_envelope_correlation(batch, wokada_normalized, sr=sr1)
    print(f"Envelope correlation (after normalization):  {corr_after:.4f}")
    print(f"Improvement: {(corr_after - corr_before) * 100:.2f}%")

    # Per-segment analysis
    print("\n--- Per-segment Analysis ---")
    segment_sec = 0.5
    segment_samples = int(sr1 * segment_sec)
    n_segments = min_len // segment_samples

    improvements = []
    for i in range(min(10, n_segments)):  # First 10 segments
        start = i * segment_samples
        end = start + segment_samples

        b_seg = batch[start:end]
        w_seg = wokada[start:end]

        # Local energy ratio
        b_rms = np.sqrt(np.mean(b_seg ** 2))
        w_rms = np.sqrt(np.mean(w_seg ** 2))
        local_ratio = w_rms / b_rms if b_rms > 0.001 else 1.0

        # Correlation before/after
        corr_b = compute_envelope_correlation(b_seg, w_seg, sr=sr1)
        w_norm = w_seg / local_ratio
        corr_a = compute_envelope_correlation(b_seg, w_norm, sr=sr1)

        improvements.append(corr_a - corr_b)
        print(f"  Seg {i}: ratio={local_ratio:.3f}, before={corr_b:.4f}, after={corr_a:.4f}")

    print(f"\nAverage improvement: {np.mean(improvements) * 100:.2f}%")

    # Different normalization strategies
    print("\n--- Normalization Strategies ---")

    # 1. Global scaling
    w_global = wokada / energy_ratio
    c1 = compute_envelope_correlation(batch, w_global, sr=sr1)
    print(f"1. Global scaling (1/{energy_ratio:.3f}): {c1:.4f}")

    # 2. Per-segment adaptive
    window_sec = 0.1
    window_samples = int(sr1 * window_sec)
    w_adaptive = wokada.copy()
    for i in range(0, len(wokada) - window_samples, window_samples):
        b_rms = np.sqrt(np.mean(batch[i:i+window_samples] ** 2))
        w_rms = np.sqrt(np.mean(wokada[i:i+window_samples] ** 2))
        if w_rms > 0.001:
            w_adaptive[i:i+window_samples] *= (b_rms / w_rms)
    c2 = compute_envelope_correlation(batch, w_adaptive, sr=sr1)
    print(f"2. Per-segment adaptive (100ms): {c2:.4f}")

    # 3. Running average
    smooth_window = int(sr1 * 0.5)  # 500ms window
    w_smooth = wokada.copy()
    for i in range(0, len(wokada), window_samples):
        start = max(0, i - smooth_window // 2)
        end = min(len(wokada), i + smooth_window // 2)
        b_rms = np.sqrt(np.mean(batch[start:end] ** 2))
        w_rms = np.sqrt(np.mean(wokada[start:end] ** 2))
        if w_rms > 0.001:
            w_smooth[i:i+window_samples] *= (b_rms / w_rms)
    c3 = compute_envelope_correlation(batch, w_smooth, sr=sr1)
    print(f"3. Smoothed adaptive (500ms window): {c3:.4f}")

    print("\n" + "=" * 60)
    print("Conclusion:")
    if c1 > corr_before:
        print(f"Global energy normalization improves correlation by {(c1 - corr_before) * 100:.2f}%")
    else:
        print("Energy normalization does not significantly improve correlation")
    print("=" * 60)


if __name__ == "__main__":
    main()
