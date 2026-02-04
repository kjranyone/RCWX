"""Quick volume comparison without model processing."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from scipy.io import wavfile


def analyze_volume_range(audio: np.ndarray, start_pct: float, end_pct: float, sr: int):
    """Analyze min/max/RMS in percentage range."""
    total_len = len(audio)
    start_idx = int(total_len * start_pct / 100)
    end_idx = int(total_len * end_pct / 100)

    region = audio[start_idx:end_idx]

    return {
        'min': float(np.min(region)),
        'max': float(np.max(region)),
        'rms': float(np.sqrt(np.mean(region**2))),
        'samples': len(region),
        'duration_ms': len(region) / sr * 1000,
    }


def load_audio(path: str):
    """Load audio file."""
    sr, audio = wavfile.read(path)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio, sr


def main():
    print("=" * 80)
    print("QUICK VOLUME COMPARISON (30-70% range)")
    print("=" * 80)

    # Load original
    print("\nLoading original audio...")
    audio_orig, sr_orig = load_audio("debug_audio/01_input_raw.wav")
    print(f"Original: {len(audio_orig)/sr_orig:.2f}s @ {sr_orig}Hz")

    # Analyze original
    orig_stats = analyze_volume_range(audio_orig, 30, 70, sr_orig)

    print(f"\nOriginal (30-70% = {orig_stats['duration_ms']:.0f}ms):")
    print(f"  Min: {orig_stats['min']:+.6f}")
    print(f"  Max: {orig_stats['max']:+.6f}")
    print(f"  RMS: {orig_stats['rms']:.6f}")
    print(f"  Peak-to-Peak: {orig_stats['max'] - orig_stats['min']:.6f}")

    # Check if processed file exists
    processed_files = [
        "test_output/volume_compare_output.wav",
        "test_output/rms_test_output.wav",
        "test_output/quick_rms_test_output.wav",
        "test_output/continuous_tone_output.wav",
    ]

    processed_path = None
    for p in processed_files:
        if Path(p).exists():
            processed_path = p
            break

    if processed_path is None:
        print("\n[INFO] No processed file found yet.")
        print("Run the volume comparison test first to generate processed audio.")
        print("\nShowing original audio statistics only:")
        print("=" * 80)
        return

    # Load processed
    print(f"\nLoading processed audio: {processed_path}")
    audio_proc, sr_proc = load_audio(processed_path)
    print(f"Processed: {len(audio_proc)/sr_proc:.2f}s @ {sr_proc}Hz")

    # Analyze processed
    proc_stats = analyze_volume_range(audio_proc, 30, 70, sr_proc)

    print(f"\nProcessed (30-70% = {proc_stats['duration_ms']:.0f}ms):")
    print(f"  Min: {proc_stats['min']:+.6f}")
    print(f"  Max: {proc_stats['max']:+.6f}")
    print(f"  RMS: {proc_stats['rms']:.6f}")
    print(f"  Peak-to-Peak: {proc_stats['max'] - proc_stats['min']:.6f}")

    # Compare
    print(f"\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    min_ratio = proc_stats['min'] / orig_stats['min'] if orig_stats['min'] != 0 else 0
    max_ratio = proc_stats['max'] / orig_stats['max'] if orig_stats['max'] != 0 else 0
    rms_ratio = proc_stats['rms'] / orig_stats['rms'] if orig_stats['rms'] > 0 else 0
    rms_db = 20 * np.log10(rms_ratio) if rms_ratio > 0 else -np.inf

    print(f"\nMin: {proc_stats['min']:+.6f} vs {orig_stats['min']:+.6f} = {min_ratio:.2%}")
    print(f"Max: {proc_stats['max']:+.6f} vs {orig_stats['max']:+.6f} = {max_ratio:.2%}")
    print(f"RMS: {proc_stats['rms']:.6f} vs {orig_stats['rms']:.6f} = {rms_ratio:.2%} ({rms_db:+.1f}dB)")

    # Verdict
    print(f"\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    if rms_ratio >= 0.95:
        verdict = "[SUCCESS] Volume preserved excellently (>=95%)"
    elif rms_ratio >= 0.85:
        verdict = "[OK] Minor volume reduction (85-95%)"
    elif rms_ratio >= 0.70:
        verdict = "[WARNING] Noticeable volume reduction (70-85%)"
    else:
        verdict = "[ISSUE] Significant volume loss (<70%)"

    print(f"\n{verdict}")
    print(f"RMS ratio: {rms_ratio:.1%}")
    print(f"RMS change: {rms_db:+.1f}dB")
    print("=" * 80)


if __name__ == "__main__":
    main()
