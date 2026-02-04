"""Test independent optimization of context_sec and crossfade_sec."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from scipy.io import wavfile
from rcwx.pipeline.inference import RVCPipeline
from rcwx.pipeline.realtime import RealtimeVoiceChanger, RealtimeConfig
from rcwx.audio.resample import resample, StatefulResampler


def load_audio(path: str, target_sr: int = 48000, max_sec: float = 20.0) -> np.ndarray:
    """Load audio file."""
    sr, audio = wavfile.read(path)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    else:
        audio = audio.astype(np.float32)

    if audio.ndim > 1:
        audio = audio[:, 0]
    max_samples = int(sr * max_sec)
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    if sr != target_sr:
        audio = resample(audio, sr, target_sr)
    return audio.astype(np.float32)


def process_batch(pipeline, audio):
    """Process in batch."""
    pipeline.clear_cache()
    input_resampler = StatefulResampler(48000, 16000)
    audio_16k = input_resampler.resample_chunk(audio)
    output = pipeline.infer(audio_16k, input_sr=16000, pitch_shift=0,
                           f0_method="rmvpe", index_rate=0.0,
                           voice_gate_mode="off", use_feature_cache=False)
    output_resampler = StatefulResampler(pipeline.sample_rate, 48000)
    return output_resampler.resample_chunk(output)


def process_streaming(pipeline, audio, crossfade, context):
    """Process streaming."""
    rt_config = RealtimeConfig(
        mic_sample_rate=48000, input_sample_rate=16000, output_sample_rate=48000,
        chunk_sec=0.35, f0_method="rmvpe", chunking_mode="wokada",
        context_sec=context, crossfade_sec=crossfade, use_sola=True,
        rvc_overlap_sec=0.0, index_rate=0.0, voice_gate_mode="off",
        use_feature_cache=False, use_adaptive_parameters=False,
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


def compute_correlation(a1, a2, frame=480):
    """Compute envelope correlation."""
    min_len = min(len(a1), len(a2))
    a1, a2 = a1[:min_len], a2[:min_len]
    n = min_len // frame
    if n == 0:
        return 0.0
    e1 = np.array([np.sqrt(np.mean(a1[i*frame:(i+1)*frame]**2)) for i in range(n)])
    e2 = np.array([np.sqrt(np.mean(a2[i*frame:(i+1)*frame]**2)) for i in range(n)])
    if np.std(e1) < 1e-8 or np.std(e2) < 1e-8:
        return 0.0
    return float(np.corrcoef(e1, e2)[0, 1])


def main():
    print("Loading...")
    pipeline = RVCPipeline("sample_data/hogaraka/hogarakav2.pth", use_compile=False)
    audio = load_audio("sample_data/seki.wav", max_sec=20.0)
    print(f"Audio: {len(audio)/48000:.2f}s\n")

    print("Batch processing...")
    batch = process_batch(pipeline, audio)
    print(f"Batch: {len(batch)} samples\n")

    # Test 1: Fix context=0.119, vary crossfade
    print("=" * 80)
    print("TEST 1: Fixed context=0.119s, varying crossfade")
    print("=" * 80)

    tests1 = [
        (0.08, 0.119), (0.10, 0.119), (0.119, 0.119),
        (0.13, 0.119), (0.15, 0.119),
    ]

    for crossfade, context in tests1:
        try:
            out = process_streaming(pipeline, audio, crossfade, context)
            corr = compute_correlation(batch, out)
            print(f"  crossfade={crossfade:.3f}s: corr={corr:.4f}, len={len(out)/len(batch)*100:.1f}%")
        except Exception as e:
            print(f"  crossfade={crossfade:.3f}s: ERROR - {e}")

    # Test 2: Fix crossfade=0.119, vary context
    print("\n" + "=" * 80)
    print("TEST 2: Fixed crossfade=0.119s, varying context")
    print("=" * 80)

    tests2 = [
        (0.119, 0.08), (0.119, 0.10), (0.119, 0.119),
        (0.119, 0.13), (0.119, 0.15),
    ]

    for crossfade, context in tests2:
        try:
            out = process_streaming(pipeline, audio, crossfade, context)
            corr = compute_correlation(batch, out)
            print(f"  context={context:.3f}s: corr={corr:.4f}, len={len(out)/len(batch)*100:.1f}%")
        except Exception as e:
            print(f"  context={context:.3f}s: ERROR - {e}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
