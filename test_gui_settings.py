"""Test with GUI default settings to reproduce puchi-puchi issue.

GUI defaults:
- chunk_sec: 0.10 (100ms, FCPE minimum)
- f0_method: fcpe
- context_sec: 0.10
- crossfade_sec: 0.05
- buffer_margin: 0.3 (tight)
- prebuffer_chunks: 1
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from scipy.io import wavfile
from rcwx.pipeline.inference import RVCPipeline
from rcwx.pipeline.realtime import RealtimeVoiceChanger, RealtimeConfig
from rcwx.audio.resample import resample, StatefulResampler
import time


def load_audio(path: str, max_sec: float = 10.0) -> np.ndarray:
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


def test_with_settings(pipeline, audio, chunk_sec, f0_method, description):
    """Test streaming with given settings."""
    print(f"\n{'='*80}")
    print(f"Testing: {description}")
    print(f"  chunk_sec: {chunk_sec}s")
    print(f"  f0_method: {f0_method}")
    print(f"{'='*80}")

    rt_config = RealtimeConfig(
        mic_sample_rate=48000,
        input_sample_rate=16000,
        output_sample_rate=48000,
        chunk_sec=chunk_sec,
        f0_method=f0_method,
        chunking_mode="wokada",
        context_sec=0.10,
        crossfade_sec=0.05,
        use_sola=True,
        rvc_overlap_sec=0.0,
        index_rate=0.0,
        voice_gate_mode="expand",
        use_feature_cache=True,
        use_adaptive_parameters=False,
        prebuffer_chunks=1,
        buffer_margin=0.3,
    )

    changer = RealtimeVoiceChanger(pipeline, config=rt_config)
    pipeline.clear_cache()
    changer._recalculate_buffers()
    changer._running = True

    expected_chunks = int(len(audio) / 48000 / chunk_sec) + 10
    changer.output_buffer.set_max_latency(expected_chunks * int(48000 * chunk_sec) * 3)

    # Simulate realtime: 20ms blocks (like sounddevice callback)
    input_block = int(48000 * 0.02)
    output_block = int(48000 * 0.02)
    outputs = []
    pos = 0

    # Statistics
    total_infer_time = 0
    chunks_processed = 0
    underruns = 0
    last_buffer_size = 0

    start_time = time.perf_counter()

    while pos < len(audio):
        block = audio[pos:pos + input_block]
        if len(block) < input_block:
            block = np.pad(block, (0, input_block - len(block)))
        pos += input_block

        # Add input
        changer.process_input_chunk(block)

        # Process chunks
        while changer.process_next_chunk():
            chunks_processed += 1

        # Get output
        output_available = changer.output_buffer.available
        if output_available < output_block and chunks_processed > 0:
            underruns += 1
            if underruns <= 5:
                print(f"  [UNDERRUN #{underruns}] Output buffer: {output_available} < {output_block} samples")

        changer.get_output_chunk(0)

    # Final processing
    while changer.process_next_chunk():
        chunks_processed += 1
        changer.get_output_chunk(0)

    changer.flush_final_sola_buffer()
    changer.get_output_chunk(0)

    total_time = (time.perf_counter() - start_time) * 1000

    # Retrieve output
    while changer.output_buffer.available > 0:
        outputs.append(changer.get_output_chunk(output_block))

    output = np.concatenate(outputs) if outputs else np.array([], dtype=np.float32)

    # Statistics
    avg_chunk_time = total_time / chunks_processed if chunks_processed > 0 else 0
    realtime_factor = (len(audio) / 48000) / (total_time / 1000)

    print(f"\nResults:")
    print(f"  Chunks processed:  {chunks_processed}")
    print(f"  Total time:        {total_time:.0f}ms")
    print(f"  Avg chunk time:    {avg_chunk_time:.1f}ms")
    print(f"  Realtime factor:   {realtime_factor:.2f}x (>1.0 = faster than realtime)")
    print(f"  Buffer underruns:  {underruns}")
    print(f"  Output length:     {len(output)} samples ({len(output)/48000:.2f}s)")

    if underruns > 0:
        print(f"\n  [!] PROBLEM DETECTED: {underruns} buffer underruns")
        print(f"      This causes puchi-puchi audio glitches!")

    if realtime_factor < 1.0:
        print(f"\n  [!] PROBLEM: Processing slower than realtime ({realtime_factor:.2f}x)")
        print(f"      Cannot sustain {chunk_sec*1000:.0f}ms chunks!")

    return {
        'chunks': chunks_processed,
        'time_ms': total_time,
        'avg_chunk_ms': avg_chunk_time,
        'realtime_factor': realtime_factor,
        'underruns': underruns,
        'output_len': len(output),
    }


def main():
    print("="*80)
    print("GUI SETTINGS TEST: Reproducing puchi-puchi issue")
    print("="*80)

    # Load
    print("\nLoading...")
    pipeline = RVCPipeline("sample_data/hogaraka/hogarakav2.pth", use_compile=False)
    audio = load_audio("sample_data/seki.wav", max_sec=10.0)
    print(f"Audio: {len(audio)/48000:.2f}s @ 48kHz")

    results = []

    # Test 1: GUI defaults (100ms + FCPE)
    results.append(('GUI Default (100ms+FCPE)', test_with_settings(
        pipeline, audio, chunk_sec=0.10, f0_method="fcpe",
        description="GUI Default (100ms + FCPE)"
    )))

    # Test 2: GUI with RMVPE (should fail - too small chunk)
    results.append(('GUI with RMVPE (100ms)', test_with_settings(
        pipeline, audio, chunk_sec=0.10, f0_method="rmvpe",
        description="GUI with RMVPE (100ms, TOO SMALL)"
    )))

    # Test 3: Recommended (350ms + RMVPE)
    results.append(('Recommended (350ms+RMVPE)', test_with_settings(
        pipeline, audio, chunk_sec=0.35, f0_method="rmvpe",
        description="Recommended (350ms + RMVPE)"
    )))

    # Test 4: Conservative (500ms + RMVPE)
    results.append(('Conservative (500ms+RMVPE)', test_with_settings(
        pipeline, audio, chunk_sec=0.50, f0_method="rmvpe",
        description="Conservative (500ms + RMVPE)"
    )))

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Configuration':<30} {'RT Factor':>10} {'Underruns':>10} {'Avg/Chunk':>12}")
    print("-"*80)

    for name, result in results:
        rt_factor = result['realtime_factor']
        underruns = result['underruns']
        avg_ms = result['avg_chunk_ms']
        status = "[OK]" if underruns == 0 and rt_factor >= 1.0 else "[NG]"
        print(f"{name:<30} {rt_factor:>10.2f}x {underruns:>10} {avg_ms:>10.1f}ms {status}")

    print("="*80)

    print("\nCONCLUSION:")
    gui_result = results[0][1]
    if gui_result['underruns'] > 0 or gui_result['realtime_factor'] < 1.0:
        print("  [X] GUI defaults (100ms) cause problems!")
        print("      - Increase chunk_sec to 0.35 or more")
        print("      - Or increase buffer_margin to 0.5-1.0")
        print("      - Or increase prebuffer_chunks to 2-3")
    else:
        print("  [OK] GUI defaults work fine on this system")
        print("       Problem may be elsewhere (device settings, etc.)")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
