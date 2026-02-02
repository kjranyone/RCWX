"""Test chunk boundary continuity to detect click/pop noise.

Generate actual audio files to listen for puchi-puchi artifacts.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from scipy.io import wavfile
from rcwx.pipeline.inference import RVCPipeline
from rcwx.pipeline.realtime import RealtimeVoiceChanger, RealtimeConfig
from rcwx.audio.resample import resample, StatefulResampler


def load_audio(path: str, max_sec: float = 5.0) -> np.ndarray:
    """Load audio."""
    sr, audio = wavfile.read(path)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    else:
        audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio[:, 0]
    if len(audio) > int(sr * max_sec):
        audio = audio[:int(sr * max_sec)]
    if sr != 48000:
        audio = resample(audio, sr, 48000)
    return audio


def detect_discontinuities(audio: np.ndarray, threshold: float = 0.05) -> list:
    """Detect sudden jumps (discontinuities) in audio."""
    if len(audio) < 2:
        return []

    diff = np.abs(np.diff(audio))
    discontinuities = np.where(diff > threshold)[0]

    # Group nearby discontinuities
    groups = []
    if len(discontinuities) > 0:
        current_group = [discontinuities[0]]
        for idx in discontinuities[1:]:
            if idx - current_group[-1] < 100:  # Within 100 samples
                current_group.append(idx)
            else:
                groups.append(current_group)
                current_group = [idx]
        groups.append(current_group)

    return groups


def process_streaming(pipeline, audio, use_sola, use_cache, chunking_mode, description):
    """Process with specific settings and return output + stats."""
    print(f"\n{'='*80}")
    print(f"Testing: {description}")
    print(f"  use_sola: {use_sola}, use_cache: {use_cache}, mode: {chunking_mode}")
    print(f"{'='*80}")

    rt_config = RealtimeConfig(
        mic_sample_rate=48000,
        input_sample_rate=16000,
        output_sample_rate=48000,
        chunk_sec=0.15,
        f0_method="fcpe",
        chunking_mode=chunking_mode,
        context_sec=0.10,
        crossfade_sec=0.05,
        use_sola=use_sola,
        index_rate=0.0,
        voice_gate_mode="expand",
        use_feature_cache=use_cache,
        prebuffer_chunks=1,
        buffer_margin=0.3,
    )

    changer = RealtimeVoiceChanger(pipeline, config=rt_config)
    pipeline.clear_cache()
    changer._recalculate_buffers()
    changer._running = True

    expected_chunks = int(len(audio) / 48000 / 0.15) + 10
    changer.output_buffer.set_max_latency(expected_chunks * int(48000 * 0.15) * 3)

    input_block = int(48000 * 0.02)
    output_block = int(48000 * 0.02)
    outputs = []
    pos = 0

    while pos < len(audio):
        block = audio[pos:pos + input_block]
        if len(block) < input_block:
            block = np.pad(block, (0, input_block - len(block)))
        pos += input_block

        changer.process_input_chunk(block)
        while changer.process_next_chunk():
            pass
        changer.get_output_chunk(0)

    while changer.process_next_chunk():
        changer.get_output_chunk(0)
    changer.flush_final_sola_buffer()
    changer.get_output_chunk(0)

    while changer.output_buffer.available > 0:
        outputs.append(changer.get_output_chunk(output_block))

    output = np.concatenate(outputs) if outputs else np.array([], dtype=np.float32)

    # Analyze discontinuities
    discontinuities = detect_discontinuities(output, threshold=0.05)

    print(f"  Output: {len(output)} samples ({len(output)/48000:.2f}s)")
    print(f"  Discontinuities: {len(discontinuities)} groups")

    if len(discontinuities) > 0:
        print(f"  First 5 discontinuity positions (samples):")
        for i, group in enumerate(discontinuities[:5]):
            time_sec = group[0] / 48000
            print(f"    {i+1}. Sample {group[0]} ({time_sec:.3f}s) - {len(group)} samples")

    return output, len(discontinuities)


def main():
    print("="*80)
    print("CHUNK BOUNDARY CONTINUITY TEST")
    print("="*80)
    print("\nGenerating audio files to check for puchi-puchi (click/pop) noise")
    print("Listen to the generated WAV files for audible artifacts\n")

    # Load
    pipeline = RVCPipeline("sample_data/hogaraka/hogarakav2.pth", use_compile=False)
    pipeline.load()  # IMPORTANT: Load models before use
    audio = load_audio("sample_data/seki.wav", max_sec=5.0)
    print(f"Input: {len(audio)/48000:.2f}s @ 48kHz\n")

    # Output directory
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)

    tests = [
        (False, False, "wokada", "No SOLA, No Cache"),
        (True, False, "wokada", "SOLA only"),
        (False, True, "wokada", "Cache only"),
        (True, True, "wokada", "SOLA + Cache (current)"),
    ]

    results = []

    for use_sola, use_cache, mode, desc in tests:
        output, disc_count = process_streaming(
            pipeline, audio, use_sola, use_cache, mode, desc
        )

        # Save to file
        filename = f"test_{desc.replace(' ', '_').replace('+', 'and').lower()}.wav"
        filepath = output_dir / filename

        # Convert to int16 for WAV
        output_int16 = (output * 32767).astype(np.int16)
        wavfile.write(filepath, 48000, output_int16)

        results.append({
            'desc': desc,
            'discontinuities': disc_count,
            'file': filename,
        })

        print(f"  Saved: {filepath}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Configuration':<30} {'Discontinuities':>15} {'Status':>10}")
    print("-"*80)

    for r in results:
        status = "CLEAN" if r['discontinuities'] == 0 else "ARTIFACTS"
        print(f"{r['desc']:<30} {r['discontinuities']:>15} {status:>10}")

    print("="*80)

    print("\nNEXT STEPS:")
    print("  1. Listen to files in test_output/ directory")
    print("  2. Identify which has the puchi-puchi problem")
    print("  3. Compare discontinuity patterns")

    if all(r['discontinuities'] > 0 for r in results):
        print("\n  [!] ALL configurations have discontinuities")
        print("      Problem may be in:")
        print("      - Length adjustment resampling (inference.py)")
        print("      - StatefulResampler state transitions")
        print("      - Context trimming logic")
    else:
        clean = [r for r in results if r['discontinuities'] == 0]
        if clean:
            print(f"\n  [OK] Clean output: {', '.join(r['desc'] for r in clean)}")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
