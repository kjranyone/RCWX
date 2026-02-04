"""Analyze what happens at chunk ends to cause discontinuities."""

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


def test_with_chunk_logging(pipeline, audio):
    """Process and log each chunk's end characteristics."""
    print("="*80)
    print("CHUNK END ANALYSIS")
    print("="*80)

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

    chunk_outputs = []  # Store each chunk's output
    pos = 0
    chunk_idx = 0

    while pos < len(audio):
        block = audio[pos:pos + input_block]
        if len(block) < input_block:
            block = np.pad(block, (0, input_block - len(block)))
        pos += input_block

        changer.process_input_chunk(block)

        # Process and capture chunk output
        while changer.process_next_chunk():
            # Get the output that was just added to queue
            try:
                chunk_output = changer._output_queue.get_nowait()
                chunk_outputs.append(chunk_output)

                # Analyze chunk end
                if len(chunk_output) > 100:
                    # Check last 100 samples for discontinuities
                    end_region = chunk_output[-100:]
                    diff = np.abs(np.diff(end_region))
                    max_disc = np.max(diff)
                    disc_count = np.sum(diff > 0.05)

                    if chunk_idx < 10 or disc_count > 0:  # Log first 10 or any with discontinuities
                        print(f"\nChunk {chunk_idx}:")
                        print(f"  Length: {len(chunk_output)} samples")
                        print(f"  End max discontinuity: {max_disc:.4f}")
                        print(f"  End discontinuities (>0.05): {disc_count}")
                        print(f"  Last 5 values: {chunk_output[-5:]}")

                        if max_disc > 0.10:
                            # Find position of max discontinuity
                            max_pos = np.argmax(diff)
                            print(f"  Max at position: -{len(end_region)-max_pos} from end ({(len(end_region)-max_pos)/48000*1000:.2f}ms)")

                chunk_idx += 1
            except:
                break

        changer.get_output_chunk(0)

    # Final processing
    while changer.process_next_chunk():
        try:
            chunk_output = changer._output_queue.get_nowait()
            chunk_outputs.append(chunk_output)
            chunk_idx += 1
        except:
            break
        changer.get_output_chunk(0)

    changer.flush_final_sola_buffer()
    changer.get_output_chunk(0)

    # Use the collected chunks directly
    output = np.concatenate(chunk_outputs) if chunk_outputs else np.array([], dtype=np.float32)

    print(f"\n{'='*80}")
    print(f"Total chunks processed: {len(chunk_outputs)}")
    print(f"Total output: {len(output)} samples ({len(output)/48000:.2f}s)")

    # Analyze chunk boundaries in final output
    context_samples = int(0.10 * 48000)
    chunk_samples = int(0.15 * 48000)
    output_per_chunk = chunk_samples - context_samples

    boundaries = []
    pos = chunk_samples  # First chunk
    boundaries.append(pos)
    while pos < len(output):
        pos += output_per_chunk
        boundaries.append(pos)

    # Check for discontinuities near each boundary
    print(f"\nBoundary analysis:")
    diff = np.abs(np.diff(output))

    for i, boundary in enumerate(boundaries[:10]):
        # Check region around boundary
        start = max(0, boundary - 500)  # 10ms before
        end = min(len(diff), boundary + 100)  # 2ms after

        if end > start:
            region_diff = diff[start:end]
            max_disc = np.max(region_diff)
            max_pos = np.argmax(region_diff) + start

            if max_disc > 0.10:
                distance_from_boundary = max_pos - boundary
                print(f"  Boundary {i} (sample {boundary}):")
                print(f"    Max discontinuity: {max_disc:.4f} at sample {max_pos}")
                print(f"    Distance: {distance_from_boundary:+d} samples ({distance_from_boundary/48000*1000:+.2f}ms)")

    return output, chunk_outputs


def main():
    pipeline = RVCPipeline("sample_data/hogaraka/hogarakav2.pth", use_compile=False)
    pipeline.load()
    audio = load_audio("sample_data/seki.wav", max_sec=2.0)  # Shorter for detailed analysis
    print(f"Input: {len(audio)/48000:.2f}s @ 48kHz\n")

    output, chunks = test_with_chunk_logging(pipeline, audio)

    print("\n" + "="*80)
    print("HYPOTHESIS:")
    print("="*80)
    print("If discontinuities are at chunk output ends:")
    print("  → Problem in inference.py output processing")
    print("  → Or in resampling (40kHz→48kHz)")
    print("If discontinuities appear after concatenation:")
    print("  → Problem in SOLA buffer handling")
    print("  → Or in output buffer concatenation")
    print("="*80)


if __name__ == "__main__":
    main()
