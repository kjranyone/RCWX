"""Test without final 40kHz->48kHz resampling to isolate resample artifacts."""

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


def count_discontinuities(audio: np.ndarray, threshold: float = 0.10) -> int:
    """Count audio discontinuities above threshold."""
    if len(audio) < 2:
        return 0
    diff = np.abs(np.diff(audio))
    return int(np.sum(diff > threshold))


def main():
    print("=" * 80)
    print("TEST: 40kHz OUTPUT (NO FINAL RESAMPLE)")
    print("=" * 80)

    pipeline = RVCPipeline("sample_data/hogaraka/hogarakav2.pth", use_compile=False)
    pipeline.load()
    audio = load_audio("sample_data/seki.wav", max_sec=5.0)
    print(f"Input: {len(audio)/48000:.2f}s @ 48kHz\n")

    # Test with 40kHz output (no final resample)
    rt_config = RealtimeConfig(
        mic_sample_rate=48000,
        input_sample_rate=16000,
        output_sample_rate=40000,  # Model native output (NO resample to 48kHz)
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
    changer.output_buffer.set_max_latency(expected_chunks * int(40000 * 0.15) * 3)

    input_block = int(48000 * 0.02)
    output_block = int(40000 * 0.02)

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
        chunk = changer.get_output_chunk(output_block)
        if len(chunk) == 0:
            break
        output_chunks.append(chunk)

    output = np.concatenate(output_chunks) if output_chunks else np.array([], dtype=np.float32)

    disc_count = count_discontinuities(output, threshold=0.10)

    print(f"  Output: {len(output)} samples ({len(output)/40000:.2f}s) @ 40kHz")
    print(f"  Discontinuities (>0.10): {disc_count}")
    print()

    # Save output
    output_path = "test_output/test_40khz_output.wav"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(output_path, 40000, output)
    print(f"  Saved: {output_path}")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("If discontinuity count is MUCH LOWER:")
    print("  → 40kHz→48kHz resampling is causing artifacts")
    print("If discontinuity count is SIMILAR:")
    print("  → Problem is in model output or SOLA processing")
    print("=" * 80)


if __name__ == "__main__":
    main()
