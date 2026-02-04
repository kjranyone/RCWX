"""Volume comparison using batch processing (faster)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from scipy.io import wavfile
from rcwx.pipeline.inference import RVCPipeline


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


def main():
    print("=" * 80)
    print("BATCH VOLUME COMPARISON (30-70% range)")
    print("=" * 80)

    # Load original audio
    print("\nLoading original audio...")
    sr_orig, audio_orig = wavfile.read("debug_audio/01_input_raw.wav")
    if audio_orig.dtype == np.int16:
        audio_orig = audio_orig.astype(np.float32) / 32768.0
    if audio_orig.ndim > 1:
        audio_orig = audio_orig[:, 0]

    print(f"Original: {len(audio_orig)/sr_orig:.2f}s @ {sr_orig}Hz")

    # Limit to 2 seconds for faster processing
    max_samples = int(sr_orig * 2.0)
    if len(audio_orig) > max_samples:
        audio_orig = audio_orig[:max_samples]
        print(f"Using first 2.0s for faster processing")

    # Analyze original
    orig_stats = analyze_volume_range(audio_orig, 30, 70, sr_orig)

    print(f"\nOriginal (30-70% = {orig_stats['duration_ms']:.0f}ms):")
    print(f"  Min: {orig_stats['min']:+.6f}")
    print(f"  Max: {orig_stats['max']:+.6f}")
    print(f"  RMS: {orig_stats['rms']:.6f}")

    # Process through RVC (batch mode)
    print("\nLoading model...")
    pipeline = RVCPipeline("sample_data/hogaraka/hogarakav2.pth", use_compile=False)
    pipeline.load()

    print("Processing audio (batch mode)...")

    # Batch inference - much faster than realtime processing
    audio_processed = pipeline.infer(
        audio_orig,
        input_sr=sr_orig,
        f0_method="fcpe",
        pitch_shift=0,
        index_rate=0.0,
        filter_radius=0,
        volume_envelope=0,
        protect=0.33,
    )

    output_sr = 40000  # RVC default output rate
    print(f"Processed: {len(audio_processed)/output_sr:.2f}s @ {output_sr}Hz")

    # Analyze processed
    proc_stats = analyze_volume_range(audio_processed, 30, 70, output_sr)

    print(f"\nProcessed (30-70% = {proc_stats['duration_ms']:.0f}ms):")
    print(f"  Min: {proc_stats['min']:+.6f}")
    print(f"  Max: {proc_stats['max']:+.6f}")
    print(f"  RMS: {proc_stats['rms']:.6f}")

    # Compare
    print(f"\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    min_ratio = proc_stats['min'] / orig_stats['min'] if abs(orig_stats['min']) > 0.001 else 0
    max_ratio = proc_stats['max'] / orig_stats['max'] if orig_stats['max'] > 0.001 else 0
    rms_ratio = proc_stats['rms'] / orig_stats['rms'] if orig_stats['rms'] > 0 else 0
    rms_db = 20 * np.log10(rms_ratio) if rms_ratio > 0 else -np.inf

    print(f"\nMin: {proc_stats['min']:+.6f} vs {orig_stats['min']:+.6f}")
    print(f"Max: {proc_stats['max']:+.6f} vs {orig_stats['max']:+.6f} = {max_ratio:.1%}")
    print(f"RMS: {proc_stats['rms']:.6f} vs {orig_stats['rms']:.6f} = {rms_ratio:.1%} ({rms_db:+.1f}dB)")

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

    # Save
    output_path = "test_output/batch_volume_output.wav"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output_int16 = (audio_processed * 32767).astype(np.int16)
    wavfile.write(output_path, output_sr, output_int16)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
