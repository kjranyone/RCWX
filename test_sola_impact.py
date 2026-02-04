"""Measure the impact of SOLA on final correlation.

Test the same chunking with/without SOLA to quantify its contribution.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from scipy.io import wavfile
from rcwx.pipeline.inference import RVCPipeline
from rcwx.pipeline.realtime import RealtimeVoiceChanger, RealtimeConfig
from rcwx.audio.resample import resample, StatefulResampler


def load_audio(path: str, max_sec: float = 20.0) -> np.ndarray:
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


def correlation(a, b, frame=480):
    """Envelope correlation."""
    min_len = min(len(a), len(b))
    a, b = a[:min_len], b[:min_len]
    n = min_len // frame
    if n == 0:
        return 0.0
    e1 = np.array([np.sqrt(np.mean(a[i*frame:(i+1)*frame]**2)) for i in range(n)])
    e2 = np.array([np.sqrt(np.mean(b[i*frame:(i+1)*frame]**2)) for i in range(n)])
    if np.std(e1) < 1e-8 or np.std(e2) < 1e-8:
        return 0.0
    return float(np.corrcoef(e1, e2)[0, 1])


def process_batch(pipeline, audio):
    """True batch processing."""
    pipeline.clear_cache()
    resampler = StatefulResampler(48000, 16000)
    audio_16k = resampler.resample_chunk(audio)
    output = pipeline.infer(audio_16k, input_sr=16000, pitch_shift=0,
                           f0_method="rmvpe", index_rate=0.0,
                           voice_gate_mode="off", use_feature_cache=False)
    out_resampler = StatefulResampler(pipeline.sample_rate, 48000)
    return out_resampler.resample_chunk(output)


def process_streaming(pipeline, audio, use_sola, use_cache):
    """Streaming with RealtimeVoiceChanger."""
    rt_config = RealtimeConfig(
        mic_sample_rate=48000,
        input_sample_rate=16000,
        output_sample_rate=48000,
        chunk_sec=0.35,
        f0_method="rmvpe",
        chunking_mode="wokada",
        context_sec=0.119,
        crossfade_sec=0.119,
        use_sola=use_sola,
        rvc_overlap_sec=0.0,
        index_rate=0.0,
        voice_gate_mode="off",
        use_feature_cache=use_cache,
        use_adaptive_parameters=False,
    )

    changer = RealtimeVoiceChanger(pipeline, config=rt_config)
    pipeline.clear_cache()
    changer._recalculate_buffers()
    changer._running = True

    expected_chunks = int(len(audio) / 48000 / 0.35) + 10
    changer.output_buffer.set_max_latency(expected_chunks * int(48000 * 0.35) * 3)

    input_block = int(48000 * 0.02)
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
        outputs.append(changer.get_output_chunk(int(48000 * 0.02)))

    return np.concatenate(outputs) if outputs else np.array([], dtype=np.float32)


def main():
    print("="*80)
    print("SOLA IMPACT ANALYSIS")
    print("="*80)
    print("\nMeasuring contribution of each optimization:\n")

    # Load
    print("Loading...")
    pipeline = RVCPipeline("sample_data/hogaraka/hogarakav2.pth", use_compile=False)
    audio = load_audio("sample_data/seki.wav", max_sec=20.0)
    print(f"Audio: {len(audio)/48000:.2f}s\n")

    # Batch
    print("[1/5] Batch processing (gold standard)...")
    batch = process_batch(pipeline, audio)
    print(f"  Output: {len(batch)} samples\n")

    # Test 1: No SOLA, No Cache
    print("[2/5] Streaming: NO SOLA, NO Cache...")
    stream1 = process_streaming(pipeline, audio, use_sola=False, use_cache=False)
    corr1 = correlation(batch, stream1)
    print(f"  Output: {len(stream1)} samples")
    print(f"  Correlation: {corr1:.4f}")
    print(f"  Length: {len(stream1)/len(batch)*100:.1f}%\n")

    # Test 2: No SOLA, With Cache
    print("[3/5] Streaming: NO SOLA, WITH Cache...")
    stream2 = process_streaming(pipeline, audio, use_sola=False, use_cache=True)
    corr2 = correlation(batch, stream2)
    improvement2 = corr2 - corr1
    print(f"  Output: {len(stream2)} samples")
    print(f"  Correlation: {corr2:.4f} (+{improvement2:+.4f})")
    print(f"  Length: {len(stream2)/len(batch)*100:.1f}%\n")

    # Test 3: With SOLA, No Cache
    print("[4/5] Streaming: WITH SOLA, NO Cache...")
    stream3 = process_streaming(pipeline, audio, use_sola=True, use_cache=False)
    corr3 = correlation(batch, stream3)
    improvement3 = corr3 - corr1
    print(f"  Output: {len(stream3)} samples")
    print(f"  Correlation: {corr3:.4f} (+{improvement3:+.4f})")
    print(f"  Length: {len(stream3)/len(batch)*100:.1f}%\n")

    # Test 4: With SOLA, With Cache
    print("[5/5] Streaming: WITH SOLA, WITH Cache (current best)...")
    stream4 = process_streaming(pipeline, audio, use_sola=True, use_cache=True)
    corr4 = correlation(batch, stream4)
    improvement4 = corr4 - corr1
    print(f"  Output: {len(stream4)} samples")
    print(f"  Correlation: {corr4:.4f} (+{improvement4:+.4f})")
    print(f"  Length: {len(stream4)/len(batch)*100:.1f}%\n")

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"  Batch (gold):        1.0000")
    print(f"  Base (no opt):       {corr1:.4f} (baseline)")
    print(f"  + Cache only:        {corr2:.4f} (+{improvement2:+.4f})")
    print(f"  + SOLA only:         {corr3:.4f} (+{improvement3:+.4f})")
    print(f"  + Both (current):    {corr4:.4f} (+{improvement4:+.4f})")
    print(f"  Target:              0.9300")
    print(f"  Remaining gap:       {0.93 - corr4:+.4f}")
    print("="*80)

    print("\nANALYSIS:")
    total_loss = 1.0 - corr1
    cache_gain = improvement2
    sola_gain = improvement3
    synergy = improvement4 - improvement2 - improvement3

    print(f"  Total loss (baseline):    {total_loss:.4f} ({total_loss*100:.1f}%)")
    print(f"  Cache contribution:       +{cache_gain:.4f} ({cache_gain/total_loss*100:.1f}% of loss)")
    print(f"  SOLA contribution:        +{sola_gain:.4f} ({sola_gain/total_loss*100:.1f}% of loss)")
    if abs(synergy) > 0.001:
        print(f"  Synergy (Cache+SOLA):     {synergy:+.4f}")
    print(f"  Total recovered:          {improvement4:.4f} ({improvement4/total_loss*100:.1f}% of loss)")
    print(f"  Unrecovered:              {total_loss - improvement4:.4f} ({(total_loss - improvement4)/total_loss*100:.1f}% of loss)")

    print("\nCONCLUSION:")
    if sola_gain > cache_gain:
        print("  -> SOLA is MORE important than cache")
    elif cache_gain > sola_gain:
        print("  -> Cache is MORE important than SOLA")
    else:
        print("  -> SOLA and cache are EQUALLY important")

    remaining = 0.93 - corr4
    if remaining < 0.01:
        print(f"  -> We're VERY CLOSE to 0.93 target!")
    elif remaining < 0.02:
        print(f"  -> Minor improvements needed: {remaining:.4f}")
        print("     Candidates: F0 smoothing, better length adjustment")
    else:
        print(f"  -> Major improvements needed: {remaining:.4f}")
        print("     Unrecovered baseline loss represents fundamental limit")
        print("     Solutions: learned boundary refinement, causal models")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
