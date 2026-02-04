"""Simple RMS boundary test with debug audio."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from scipy.io import wavfile
from rcwx.pipeline.inference import RVCPipeline
from rcwx.pipeline.realtime import RealtimeVoiceChanger, RealtimeConfig
from rcwx.audio.resample import resample


def main():
    import sys
    from pathlib import Path

    # Write output to file for monitoring
    log_file = Path("test_output/test_progress.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_file, "w", encoding="utf-8")

    def log_print(msg):
        print(msg)
        log.write(msg + "\n")
        log.flush()

    log_print("=" * 80)
    log_print("RMS MATCHING TEST - debug_audio/01_input_raw.wav")
    log_print("=" * 80)

    # Load audio
    sr, audio = wavfile.read("debug_audio/01_input_raw.wav")
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    if audio.ndim > 1:
        audio = audio[:, 0]

    # Limit to 2 seconds for quick test
    max_samples = int(sr * 2.0)
    audio = audio[:max_samples]

    print(f"Input: {len(audio)/sr:.2f}s @ {sr}Hz")

    # Resample to 48kHz
    if sr != 48000:
        audio = resample(audio, sr, 48000)
        sr = 48000

    # Load model
    print("Loading model...")
    import time
    start_time = time.time()
    pipeline = RVCPipeline("sample_data/hogaraka/hogarakav2.pth", use_compile=False)
    print("Calling pipeline.load()...")
    pipeline.load()
    print(f"Model loaded in {time.time() - start_time:.1f}s")

    # Config with feature cache enabled
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

    print(f"\nConfig: chunk_sec={config.chunk_sec}, use_feature_cache={config.use_feature_cache}")

    # Process
    changer = RealtimeVoiceChanger(pipeline, config=config)
    pipeline.clear_cache()
    changer._recalculate_buffers()
    changer._running = True

    expected_chunks = int(len(audio) / 48000 / config.chunk_sec) + 10
    changer.output_buffer.set_max_latency(expected_chunks * int(48000 * config.chunk_sec) * 3)

    print("Processing...")
    input_block = int(48000 * 0.02)
    pos = 0
    chunks_processed = 0

    while pos < len(audio):
        block = audio[pos:pos + input_block]
        if len(block) < input_block:
            block = np.pad(block, (0, input_block - len(block)))
        pos += input_block
        changer.process_input_chunk(block)
        if changer.process_next_chunk():
            chunks_processed += 1
            if chunks_processed % 5 == 0:
                print(f"  Processed {chunks_processed} chunks ({pos/len(audio)*100:.0f}%)")
        changer.get_output_chunk(0)

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
    print(f"Output: {len(output)/48000:.2f}s")

    # Analyze chunk boundaries
    chunk_samples = int(48000 * config.chunk_sec)
    num_chunks = len(output) // chunk_samples

    print(f"\n" + "=" * 80)
    print(f"CHUNK BOUNDARY ANALYSIS ({num_chunks} chunks)")
    print("=" * 80)

    energies = []
    for i in range(num_chunks):
        start = i * chunk_samples
        end = start + chunk_samples
        if end > len(output):
            break
        chunk = output[start:end]
        rms = np.sqrt(np.mean(chunk**2))
        energies.append(rms)

    # Print energies
    print("\nRMS Energy per chunk:")
    for i, rms in enumerate(energies):
        print(f"  Chunk {i:2d}: {rms:.6f}")

    # Calculate boundary changes
    print("\nBoundary energy changes:")
    significant_drops = 0
    for i in range(1, len(energies)):
        prev_rms = energies[i-1]
        curr_rms = energies[i]
        change_percent = (curr_rms - prev_rms) / prev_rms * 100 if prev_rms > 0 else 0

        if abs(change_percent) > 5.0:
            marker = "⚠"
            significant_drops += 1
        else:
            marker = "✓"

        print(f"  {marker} Boundary {i}: {change_percent:+6.1f}% ({prev_rms:.6f} -> {curr_rms:.6f})")

    # Statistics
    mean_rms = np.mean(energies)
    std_rms = np.std(energies)
    cv = (std_rms / mean_rms * 100) if mean_rms > 0 else 0

    print(f"\nStatistics:")
    print(f"  Mean RMS: {mean_rms:.6f}")
    print(f"  Std RMS:  {std_rms:.6f}")
    print(f"  CV:       {cv:.2f}%")
    print(f"  Significant drops (>5%): {significant_drops}/{len(energies)-1}")

    # Save output
    output_path = "test_output/rms_test_output.wav"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output_int16 = (output * 32767).astype(np.int16)
    wavfile.write(output_path, 48000, output_int16)
    print(f"\nSaved: {output_path}")

    # Verdict
    print(f"\n" + "=" * 80)
    if significant_drops == 0:
        print("✅ SUCCESS: No significant energy drops!")
        print("   RMS matching is working correctly.")
    elif significant_drops <= (len(energies)-1) * 0.1:
        print("⚠ GOOD: Minimal energy variations.")
        print(f"   Only {significant_drops} out of {len(energies)-1} boundaries affected.")
    else:
        print("❌ ISSUE: Energy drops still present.")
        print(f"   {significant_drops} out of {len(energies)-1} boundaries affected.")
    print("=" * 80)


if __name__ == "__main__":
    main()
