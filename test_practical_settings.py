"""Test practical settings to fix puchi-puchi issue."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from scipy.io import wavfile
from rcwx.pipeline.inference import RVCPipeline
from rcwx.pipeline.realtime import RealtimeVoiceChanger, RealtimeConfig
from rcwx.audio.resample import resample
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


def test_config(pipeline, audio, chunk_sec, f0_method, buffer_margin, prebuffer, name):
    """Test a configuration."""
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
        index_rate=0.0,
        voice_gate_mode="expand",
        use_feature_cache=True,
        prebuffer_chunks=prebuffer,
        buffer_margin=buffer_margin,
    )

    changer = RealtimeVoiceChanger(pipeline, config=rt_config)
    pipeline.clear_cache()
    changer._recalculate_buffers()
    changer._running = True

    expected_chunks = int(len(audio) / 48000 / chunk_sec) + 10
    changer.output_buffer.set_max_latency(expected_chunks * int(48000 * chunk_sec) * 3)

    input_block = int(48000 * 0.02)
    output_block = int(48000 * 0.02)
    pos = 0
    chunks_processed = 0
    underruns = 0

    start_time = time.perf_counter()

    while pos < len(audio):
        block = audio[pos:pos + input_block]
        if len(block) < input_block:
            block = np.pad(block, (0, input_block - len(block)))
        pos += input_block

        changer.process_input_chunk(block)
        while changer.process_next_chunk():
            chunks_processed += 1

        if changer.output_buffer.available < output_block and chunks_processed > 0:
            underruns += 1

        changer.get_output_chunk(0)

    while changer.process_next_chunk():
        chunks_processed += 1
        changer.get_output_chunk(0)

    changer.flush_final_sola_buffer()

    total_time = (time.perf_counter() - start_time) * 1000
    avg_chunk_time = total_time / chunks_processed if chunks_processed > 0 else 0
    realtime_factor = (len(audio) / 48000) / (total_time / 1000)

    return {
        'name': name,
        'chunk_sec': chunk_sec,
        'f0': f0_method,
        'margin': buffer_margin,
        'prebuf': prebuffer,
        'chunks': chunks_processed,
        'avg_ms': avg_chunk_time,
        'rt_factor': realtime_factor,
        'underruns': underruns,
        'status': 'OK' if underruns == 0 and realtime_factor >= 1.0 else 'NG',
    }


def main():
    print("="*80)
    print("PRACTICAL SETTINGS TEST")
    print("="*80)

    pipeline = RVCPipeline("sample_data/hogaraka/hogarakav2.pth", use_compile=False)
    audio = load_audio("sample_data/seki.wav", max_sec=10.0)
    print(f"\nAudio: {len(audio)/48000:.2f}s @ 48kHz\n")

    configs = [
        # (chunk_sec, f0_method, buffer_margin, prebuffer, name)
        (0.10, 'fcpe', 0.3, 1, 'Current GUI Default'),
        (0.15, 'fcpe', 0.3, 1, 'FCPE 150ms (recommended)'),
        (0.20, 'fcpe', 0.3, 1, 'FCPE 200ms (safe)'),
        (0.10, 'fcpe', 0.5, 2, 'FCPE 100ms + buffer boost'),
        (0.35, 'rmvpe', 0.3, 1, 'RMVPE 350ms (high quality)'),
    ]

    results = []
    for chunk_sec, f0_method, margin, prebuf, name in configs:
        print(f"Testing: {name}...")
        try:
            result = test_config(pipeline, audio, chunk_sec, f0_method, margin, prebuf, name)
            results.append(result)
            print(f"  RT: {result['rt_factor']:.2f}x, Underruns: {result['underruns']}, "
                  f"Avg: {result['avg_ms']:.1f}ms [{result['status']}]")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                'name': name, 'chunk_sec': chunk_sec, 'f0': f0_method,
                'margin': margin, 'prebuf': prebuf, 'chunks': 0,
                'avg_ms': 0, 'rt_factor': 0, 'underruns': 999, 'status': 'ERROR'
            })

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Configuration':<35} {'Chunk':>8} {'RT':>8} {'Avg/ms':>8} {'Under':>6} {'Status':>8}")
    print("-"*80)

    for r in results:
        chunk_ms = r['chunk_sec'] * 1000
        print(f"{r['name']:<35} {chunk_ms:>6.0f}ms {r['rt_factor']:>7.2f}x "
              f"{r['avg_ms']:>7.1f}ms {r['underruns']:>6} {r['status']:>8}")

    print("="*80)

    # Recommendations
    print("\nRECOMMENDATIONS:")
    ok_results = [r for r in results if r['status'] == 'OK']

    if not ok_results:
        print("  [!] NO configuration worked reliably!")
        print("      Your system may be too slow for realtime processing.")
        print("      Consider: faster GPU, close background apps, disable antivirus")
    else:
        best = min(ok_results, key=lambda x: x['chunk_sec'])
        print(f"  [OK] Minimum working: {best['name']}")
        print(f"       chunk_sec={best['chunk_sec']}, buffer_margin={best['margin']}, "
              f"prebuffer_chunks={best['prebuf']}")
        print(f"       Latency: ~{best['chunk_sec']*1000:.0f}ms + {best['avg_ms']:.0f}ms = "
              f"{best['chunk_sec']*1000 + best['avg_ms']:.0f}ms total")

        if best['chunk_sec'] > 0.10:
            print(f"\n  [!] GUI default (100ms) is TOO SMALL for this system")
            print(f"      Update config.py: chunk_sec = {best['chunk_sec']}")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
