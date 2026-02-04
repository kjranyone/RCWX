"""Test with continuous tone to visualize chunk boundary issues."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from scipy.io import wavfile
from rcwx.pipeline.inference import RVCPipeline
from rcwx.pipeline.realtime import RealtimeVoiceChanger, RealtimeConfig
from rcwx.audio.resample import resample


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load audio and return with original sample rate."""
    sr, audio = wavfile.read(path)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    else:
        audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio, sr


def analyze_chunk_boundaries(audio: np.ndarray, chunk_sec: float = 0.15, sr: int = 48000):
    """Analyze energy at chunk boundaries."""
    chunk_samples = int(chunk_sec * sr)

    # Calculate RMS energy for each chunk
    num_chunks = len(audio) // chunk_samples
    energies = []
    positions = []

    for i in range(num_chunks):
        start = i * chunk_samples
        end = start + chunk_samples
        chunk = audio[start:end]

        rms = np.sqrt(np.mean(chunk**2))
        energies.append(rms)
        positions.append(start / sr)

    # Calculate boundary transitions
    boundary_drops = []
    for i in range(1, len(energies)):
        drop = energies[i-1] - energies[i]
        drop_percent = (drop / energies[i-1] * 100) if energies[i-1] > 0 else 0
        boundary_drops.append({
            'chunk': i,
            'time': positions[i],
            'prev_energy': energies[i-1],
            'curr_energy': energies[i],
            'drop': drop,
            'drop_percent': drop_percent
        })

    return energies, positions, boundary_drops


def main():
    print("=" * 80)
    print("CONTINUOUS TONE TEST - Chunk Boundary Analysis")
    print("=" * 80)

    # Load continuous tone
    audio, orig_sr = load_audio("debug_audio/01_input_raw.wav")
    print(f"Input: {len(audio)/orig_sr:.2f}s @ {orig_sr}Hz")

    # Resample to 48kHz if needed
    if orig_sr != 48000:
        audio = resample(audio, orig_sr, 48000)
        print(f"Resampled to 48kHz")

    # Process through RVC
    pipeline = RVCPipeline("sample_data/hogaraka/hogarakav2.pth", use_compile=False)
    pipeline.load()

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
        use_feature_cache=True,
        prebuffer_chunks=1,
        buffer_margin=0.3,
    )

    changer = RealtimeVoiceChanger(pipeline, config=rt_config)
    pipeline.clear_cache()
    changer._recalculate_buffers()
    changer._running = True

    expected_chunks = int(len(audio) / 48000 / 0.15) + 10
    changer.output_buffer.set_max_latency(expected_chunks * int(48000 * 0.15) * 3)

    # Process audio
    input_block = int(48000 * 0.02)
    pos = 0

    print(f"\nProcessing...")
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

    # Analyze chunk boundaries
    print(f"\n" + "=" * 80)
    print("CHUNK BOUNDARY ENERGY ANALYSIS")
    print("=" * 80)

    energies, positions, boundary_drops = analyze_chunk_boundaries(output, chunk_sec=0.15, sr=48000)

    print(f"\nTotal chunks: {len(energies)}")
    print(f"\nEnergy statistics:")
    print(f"  Mean: {np.mean(energies):.6f}")
    print(f"  Std:  {np.std(energies):.6f}")
    print(f"  Min:  {np.min(energies):.6f}")
    print(f"  Max:  {np.max(energies):.6f}")

    # Find significant drops
    significant_drops = [d for d in boundary_drops if abs(d['drop_percent']) > 5.0]

    print(f"\nSignificant energy changes (>5%):")
    for drop in significant_drops[:10]:
        print(f"  Chunk {drop['chunk']}: {drop['drop_percent']:+.1f}% "
              f"({drop['prev_energy']:.4f} -> {drop['curr_energy']:.4f})")

    if len(significant_drops) > 10:
        print(f"  ... and {len(significant_drops) - 10} more")

    # Print all boundary changes
    print(f"\nAll boundary energy changes:")
    for i, drop in enumerate(boundary_drops):
        marker = "⚠" if abs(drop['drop_percent']) > 5 else " "
        print(f"  {marker} Boundary {drop['chunk']}: {drop['drop_percent']:+6.1f}% "
              f"({drop['prev_energy']:.4f} -> {drop['curr_energy']:.4f})")

    # Save output audio
    audio_path = "test_output/continuous_tone_output.wav"
    output_int16 = (output * 32767).astype(np.int16)
    wavfile.write(audio_path, 48000, output_int16)
    print(f"Audio saved: {audio_path}")

    # Count discontinuities
    diff = np.abs(np.diff(output))
    disc_count = int(np.sum(diff > 0.10))
    print(f"\nDiscontinuities (>0.10): {disc_count}")

    print("\n" + "=" * 80)
    print("KEY FINDINGS:")
    print("=" * 80)
    if len(significant_drops) > 0:
        print(f"⚠ Found {len(significant_drops)} boundaries with >5% energy change")
        print("→ This indicates chunk boundary processing issues")
    else:
        print("✓ No significant energy drops at boundaries")

    if disc_count > 0:
        print(f"⚠ Found {disc_count} discontinuities in output")
    else:
        print("✓ No discontinuities detected")
    print("=" * 80)


if __name__ == "__main__":
    main()
