"""Test SOLA + Cache configuration only (without smoothing)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from scipy.io import wavfile
from rcwx.pipeline.inference import RVCPipeline
from rcwx.pipeline.realtime import RealtimeVoiceChanger, RealtimeConfig
from rcwx.audio.resample import resample


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


def detect_discontinuities(audio: np.ndarray, threshold: float = 0.10) -> int:
    """Count discontinuities above threshold."""
    if len(audio) < 2:
        return 0
    diff = np.abs(np.diff(audio))
    return np.sum(diff > threshold)


def process_streaming(pipeline, audio):
    """Process with SOLA + Cache."""
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

    # Count discontinuities
    disc_count = detect_discontinuities(output, threshold=0.10)

    print(f"  Output: {len(output)} samples ({len(output)/48000:.2f}s)")
    print(f"  Discontinuities (>0.10): {disc_count}")

    return output, disc_count


def main():
    print("="*80)
    print("SOLA + CACHE TEST (Smoothing Disabled)")
    print("="*80)

    # Load
    pipeline = RVCPipeline("sample_data/hogaraka/hogarakav2.pth", use_compile=False)
    pipeline.load()  # IMPORTANT: Load models before use
    audio = load_audio("sample_data/seki.wav", max_sec=5.0)
    print(f"\nInput: {len(audio)/48000:.2f}s @ 48kHz\n")

    # Test
    output, disc_count = process_streaming(pipeline, audio)

    # Save to file
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    filename = "test_sola_cache_no_smoothing.wav"
    filepath = output_dir / filename

    # Convert to int16 for WAV
    output_int16 = (output * 32767).astype(np.int16)
    wavfile.write(filepath, 48000, output_int16)

    print(f"\n  Saved: {filepath}")

    print("\n" + "="*80)
    print("RESULT")
    print("="*80)
    print(f"  Discontinuities (>0.10): {disc_count}")
    print(f"  Previous (with smoothing): 178")

    if disc_count < 178:
        print(f"\n  [OK] Improved! Reduction: {178 - disc_count}")
        print("       Smoothing was causing additional discontinuities.")
    elif disc_count == 178:
        print("\n  [INFO] No change. Smoothing was not the issue.")
    else:
        print(f"\n  [NG] Worse! Increase: {disc_count - 178}")

    print("="*80)


if __name__ == "__main__":
    main()
