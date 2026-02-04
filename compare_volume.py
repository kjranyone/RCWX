"""Compare volume (min/max) between original and processed audio."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from scipy.io import wavfile
from rcwx.pipeline.inference import RVCPipeline
from rcwx.pipeline.realtime import RealtimeVoiceChanger, RealtimeConfig
from rcwx.audio.resample import resample


def analyze_volume_range(audio: np.ndarray, start_pct: float, end_pct: float):
    """Analyze min/max in percentage range."""
    total_len = len(audio)
    start_idx = int(total_len * start_pct / 100)
    end_idx = int(total_len * end_pct / 100)

    region = audio[start_idx:end_idx]
    return {
        'min': float(np.min(region)),
        'max': float(np.max(region)),
        'rms': float(np.sqrt(np.mean(region**2))),
        'samples': len(region),
    }


def main():
    print("=" * 80)
    print("VOLUME COMPARISON TEST")
    print("=" * 80)

    # Load original audio
    print("\nLoading original audio...")
    sr_orig, audio_orig = wavfile.read("debug_audio/01_input_raw.wav")
    if audio_orig.dtype == np.int16:
        audio_orig = audio_orig.astype(np.float32) / 32768.0
    if audio_orig.ndim > 1:
        audio_orig = audio_orig[:, 0]

    print(f"Original: {len(audio_orig)/sr_orig:.2f}s @ {sr_orig}Hz")

    # Resample to 48kHz for comparison
    if sr_orig != 48000:
        audio_orig_48k = resample(audio_orig, sr_orig, 48000)
    else:
        audio_orig_48k = audio_orig

    # Limit to 3s for quick processing
    max_samples = int(48000 * 3.0)
    if len(audio_orig_48k) > max_samples:
        audio_orig_48k = audio_orig_48k[:max_samples]

    print(f"Using first {len(audio_orig_48k)/48000:.2f}s for comparison")

    # Process through RVC
    print("\nLoading model...")
    pipeline = RVCPipeline("sample_data/hogaraka/hogarakav2.pth", use_compile=False)
    pipeline.load()

    config = RealtimeConfig(
        mic_sample_rate=48000,
        input_sample_rate=16000,
        output_sample_rate=48000,
        chunk_sec=0.15,
        f0_method="fcpe",
        chunking_mode="wokada",
        context_sec=0.10,
        crossfade_sec=0.05,
        use_sola=True,
        index_rate=0.0,
        voice_gate_mode="expand",
        use_feature_cache=True,
        prebuffer_chunks=1,
        buffer_margin=0.3,
    )

    print("\nProcessing audio...")
    changer = RealtimeVoiceChanger(pipeline, config=config)
    pipeline.clear_cache()
    changer._recalculate_buffers()
    changer._running = True

    expected_chunks = int(len(audio_orig_48k) / 48000 / config.chunk_sec) + 10
    changer.output_buffer.set_max_latency(expected_chunks * int(48000 * config.chunk_sec) * 3)

    # Feed audio
    input_block = int(48000 * 0.02)
    pos = 0

    while pos < len(audio_orig_48k):
        block = audio_orig_48k[pos:pos + input_block]
        if len(block) < input_block:
            block = np.pad(block, (0, input_block - len(block)))
        pos += input_block
        changer.process_input_chunk(block)
        changer.process_next_chunk()
        changer.get_output_chunk(0)

    # Final processing
    while changer.process_next_chunk():
        changer.get_output_chunk(0)

    changer.flush_final_sola_buffer()

    # Collect output
    output_chunks = []
    while True:
        chunk = changer.get_output_chunk(int(48000 * 0.02))
        if len(chunk) == 0:
            break
        output_chunks.append(chunk)

    audio_processed = np.concatenate(output_chunks) if output_chunks else np.array([], dtype=np.float32)

    print(f"Processed: {len(audio_processed)/48000:.2f}s")

    # Compare volumes in 30-70% range
    print("\n" + "=" * 80)
    print("VOLUME COMPARISON (30-70% range)")
    print("=" * 80)

    orig_stats = analyze_volume_range(audio_orig_48k, 30, 70)
    proc_stats = analyze_volume_range(audio_processed, 30, 70)

    print(f"\nOriginal (30-70%):")
    print(f"  Min: {orig_stats['min']:+.6f}")
    print(f"  Max: {orig_stats['max']:+.6f}")
    print(f"  RMS: {orig_stats['rms']:.6f}")
    print(f"  Samples: {orig_stats['samples']}")

    print(f"\nProcessed (30-70%):")
    print(f"  Min: {proc_stats['min']:+.6f}")
    print(f"  Max: {proc_stats['max']:+.6f}")
    print(f"  RMS: {proc_stats['rms']:.6f}")
    print(f"  Samples: {proc_stats['samples']}")

    # Calculate differences
    print(f"\n" + "=" * 80)
    print("DIFFERENCE")
    print("=" * 80)

    min_diff = proc_stats['min'] - orig_stats['min']
    max_diff = proc_stats['max'] - orig_stats['max']
    rms_ratio = proc_stats['rms'] / orig_stats['rms'] if orig_stats['rms'] > 0 else 0
    rms_db = 20 * np.log10(rms_ratio) if rms_ratio > 0 else -np.inf

    print(f"\nMin: {min_diff:+.6f} ({proc_stats['min']/orig_stats['min']*100:.1f}% of original)")
    print(f"Max: {max_diff:+.6f} ({proc_stats['max']/orig_stats['max']*100:.1f}% of original)")
    print(f"RMS: {rms_ratio:.3f}x ({rms_db:+.1f}dB)")

    # Verdict
    print(f"\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    if rms_ratio >= 0.95:
        print("[SUCCESS] Volume preserved well (>95%)")
    elif rms_ratio >= 0.85:
        print("[OK] Minor volume reduction (85-95%)")
    elif rms_ratio >= 0.70:
        print("[WARNING] Noticeable volume reduction (70-85%)")
    else:
        print("[ISSUE] Significant volume loss (<70%)")

    print(f"\nRMS ratio: {rms_ratio:.1%}")
    print(f"RMS change: {rms_db:+.1f}dB")
    print("=" * 80)

    # Save for inspection
    output_path = "test_output/volume_compare_output.wav"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output_int16 = (audio_processed * 32767).astype(np.int16)
    wavfile.write(output_path, 48000, output_int16)
    print(f"\nProcessed audio saved: {output_path}")


if __name__ == "__main__":
    main()
