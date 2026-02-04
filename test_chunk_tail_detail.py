"""Detailed analysis of chunk tail regions where discontinuities occur."""

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


def analyze_tail_region(chunk: np.ndarray, chunk_idx: int, tail_ms: float = 15.0, sr: int = 48000):
    """Analyze tail region of a chunk for discontinuities."""
    tail_samples = int(tail_ms / 1000 * sr)
    if len(chunk) < tail_samples:
        return

    tail = chunk[-tail_samples:]
    diff = np.abs(np.diff(tail))

    # Find all discontinuities > 0.05
    disc_indices = np.where(diff > 0.05)[0]

    if len(disc_indices) > 0:
        print(f"\nChunk {chunk_idx}: length={len(chunk)} samples ({len(chunk)/sr*1000:.1f}ms)")
        print(f"  Tail region ({tail_ms}ms = {tail_samples} samples):")
        print(f"  Discontinuities (>0.05): {len(disc_indices)}")

        for i, idx in enumerate(disc_indices):
            if i >= 5:  # Show first 5
                print(f"  ... and {len(disc_indices) - 5} more")
                break

            # Position from end
            pos_from_end = len(tail) - idx - 1
            ms_from_end = pos_from_end / sr * 1000

            # Magnitude
            mag = diff[idx]

            # Context
            context_start = max(0, idx - 2)
            context_end = min(len(tail), idx + 3)
            context_vals = tail[context_start:context_end]

            print(f"  [{i+1}] @-{pos_from_end}samples (-{ms_from_end:.2f}ms): mag={mag:.4f}")
            print(f"      context: {context_vals}")


def main():
    print("=" * 80)
    print("CHUNK TAIL DETAILED ANALYSIS")
    print("=" * 80)

    pipeline = RVCPipeline("sample_data/hogaraka/hogarakav2.pth", use_compile=False)
    pipeline.load()
    audio = load_audio("sample_data/seki.wav", max_sec=2.0)  # Shorter for detailed analysis
    print(f"Input: {len(audio)/48000:.2f}s @ 48kHz\n")

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

    chunk_outputs = []
    pos = 0

    while pos < len(audio):
        block = audio[pos:pos + input_block]
        if len(block) < input_block:
            block = np.pad(block, (0, input_block - len(block)))
        pos += input_block

        changer.process_input_chunk(block)

        # Capture chunk outputs
        while changer.process_next_chunk():
            try:
                chunk_output = changer._output_queue.get_nowait()
                chunk_outputs.append(chunk_output)
            except:
                break

        changer.get_output_chunk(0)

    # Final processing
    while changer.process_next_chunk():
        try:
            chunk_output = changer._output_queue.get_nowait()
            chunk_outputs.append(chunk_output)
        except:
            break
        changer.get_output_chunk(0)

    changer.flush_final_sola_buffer()

    print(f"Total chunks processed: {len(chunk_outputs)}")

    # Analyze each chunk's tail
    for idx, chunk in enumerate(chunk_outputs[:15]):  # First 15 chunks
        analyze_tail_region(chunk, idx, tail_ms=15.0, sr=48000)

    print("\n" + "=" * 80)
    print("KEY INSIGHT:")
    print("=" * 80)
    print("If discontinuities are found ~2-5ms from chunk end:")
    print("  → Likely resampling artifacts (40kHz→48kHz)")
    print("  → Or SOLA buffer saving from discontinuous region")
    print("If discontinuities are found >10ms from chunk end:")
    print("  → Model output characteristics or processing artifacts")
    print("=" * 80)


if __name__ == "__main__":
    main()
