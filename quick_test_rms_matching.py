"""Quick test of RMS matching with continuous tone."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from scipy.io import wavfile
from rcwx.pipeline.inference import RVCPipeline
from rcwx.pipeline.realtime import RealtimeVoiceChanger, RealtimeConfig
from rcwx.audio.resample import resample


def load_audio(path: str, max_duration: float = 3.0) -> tuple[np.ndarray, int]:
    """Load audio (limited duration for quick testing)."""
    sr, audio = wavfile.read(path)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    else:
        audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio[:, 0]

    # Limit duration
    max_samples = int(sr * max_duration)
    if len(audio) > max_samples:
        audio = audio[:max_samples]

    return audio, sr


def analyze_boundaries(audio: np.ndarray, chunk_sec: float, sr: int):
    """Quick boundary energy analysis."""
    chunk_samples = int(chunk_sec * sr)
    num_chunks = len(audio) // chunk_samples

    energies = []
    for i in range(num_chunks):
        start = i * chunk_samples
        end = start + chunk_samples
        chunk = audio[start:end]
        rms = np.sqrt(np.mean(chunk**2))
        energies.append(rms)

    # Calculate drops
    drops = []
    for i in range(1, len(energies)):
        drop_percent = (energies[i-1] - energies[i]) / energies[i-1] * 100 if energies[i-1] > 0 else 0
        drops.append(drop_percent)

    return energies, drops


def main():
    print("=" * 80)
    print("QUICK RMS MATCHING TEST")
    print("=" * 80)

    # Load short audio (3s for quick test)
    audio, orig_sr = load_audio("debug_audio/01_input_raw.wav", max_duration=3.0)
    print(f"\nInput: {len(audio)/orig_sr:.2f}s @ {orig_sr}Hz")

    # Resample to 48kHz
    if orig_sr != 48000:
        audio = resample(audio, orig_sr, 48000)
        print(f"Resampled to 48kHz")

    # Load model
    print("\nLoading model...")
    pipeline = RVCPipeline("sample_data/hogaraka/hogarakav2.pth", use_compile=False)
    pipeline.load()

    # Test configuration
    rt_config = RealtimeConfig(
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
        use_feature_cache=True,  # Enable feature cache for boundary blending
        prebuffer_chunks=1,
        buffer_margin=0.3,
    )

    print(f"\nConfiguration:")
    print(f"  chunk_sec: {rt_config.chunk_sec}")
    print(f"  crossfade_sec: {rt_config.crossfade_sec}")
    print(f"  use_feature_cache: {rt_config.use_feature_cache}")
    print(f"  use_sola: {rt_config.use_sola}")

    # Process
    changer = RealtimeVoiceChanger(pipeline, config=rt_config)
    pipeline.clear_cache()
    changer._recalculate_buffers()
    changer._running = True

    expected_chunks = int(len(audio) / 48000 / 0.15) + 10
    changer.output_buffer.set_max_latency(expected_chunks * int(48000 * 0.15) * 3)

    # Feed audio
    print("\nProcessing...")
    input_block = int(48000 * 0.02)
    pos = 0

    while pos < len(audio):
        block = audio[pos:pos + input_block]
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

    output = np.concatenate(output_chunks) if output_chunks else np.array([], dtype=np.float32)

    print(f"Output: {len(output)} samples ({len(output)/48000:.2f}s)")

    # Analyze boundaries
    energies, drops = analyze_boundaries(output, chunk_sec=0.15, sr=48000)

    print(f"\n" + "=" * 80)
    print("BOUNDARY ENERGY ANALYSIS")
    print("=" * 80)

    print(f"\nChunks analyzed: {len(energies)}")
    print(f"\nEnergy per chunk (RMS):")
    for i, energy in enumerate(energies):
        print(f"  Chunk {i}: {energy:.6f}")

    print(f"\nEnergy changes at boundaries:")
    for i, drop in enumerate(drops):
        marker = "⚠" if abs(drop) > 5.0 else "✓"
        print(f"  {marker} Boundary {i+1}: {drop:+6.1f}%")

    # Statistics
    print(f"\nStatistics:")
    print(f"  Mean energy: {np.mean(energies):.6f}")
    print(f"  Std energy:  {np.std(energies):.6f}")
    print(f"  CV (std/mean): {np.std(energies)/np.mean(energies)*100:.1f}%")

    # Count significant drops
    significant = [d for d in drops if abs(d) > 5.0]
    print(f"\nSignificant changes (>5%): {len(significant)}/{len(drops)}")

    if len(significant) > 0:
        print(f"  Average drop: {np.mean(significant):.1f}%")
        print(f"  Max drop: {np.max(np.abs(significant)):.1f}%")

    # Discontinuities
    diff = np.abs(np.diff(output))
    disc_count = int(np.sum(diff > 0.10))
    print(f"\nDiscontinuities (>0.10): {disc_count}")

    # Save output
    output_path = "test_output/quick_rms_test_output.wav"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output_int16 = (output * 32767).astype(np.int16)
    wavfile.write(output_path, 48000, output_int16)
    print(f"\nAudio saved: {output_path}")

    # Verdict
    print(f"\n" + "=" * 80)
    print("VERDICT:")
    print("=" * 80)

    if len(significant) == 0:
        print("✅ SUCCESS: No significant energy drops at chunk boundaries!")
        print("   RMS matching is working correctly.")
    elif len(significant) <= len(drops) * 0.2:  # Less than 20%
        print("⚠ PARTIAL SUCCESS: Some minor energy variations remain.")
        print(f"   {len(significant)}/{len(drops)} boundaries show >5% change.")
    else:
        print("❌ ISSUE: Significant energy drops still present.")
        print(f"   {len(significant)}/{len(drops)} boundaries affected.")

    print("=" * 80)


if __name__ == "__main__":
    main()
