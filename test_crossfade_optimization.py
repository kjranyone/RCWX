"""Test different crossfade_sec values to find optimal setting."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from scipy.io import wavfile
from rcwx.pipeline.inference import RVCPipeline
from rcwx.pipeline.realtime import RealtimeVoiceChanger, RealtimeConfig
from rcwx.audio.resample import resample, StatefulResampler


def load_audio(path: str, target_sr: int = 48000, max_sec: float = 20.0) -> np.ndarray:
    """Load audio file and prepare for testing."""
    sr, audio = wavfile.read(path)
    # Convert to float32
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


def process_batch(pipeline, audio, mic_sr=48000, output_sr=48000):
    """Process entire audio in one batch."""
    pipeline.clear_cache()
    input_resampler = StatefulResampler(mic_sr, 16000)
    audio_16k = input_resampler.resample_chunk(audio)
    output = pipeline.infer(
        audio_16k,
        input_sr=16000,
        pitch_shift=0,
        f0_method="rmvpe",
        index_rate=0.0,
        voice_gate_mode="off",
        use_feature_cache=False,
    )
    if pipeline.sample_rate != output_sr:
        output_resampler = StatefulResampler(pipeline.sample_rate, output_sr)
        output = output_resampler.resample_chunk(output)
    return output


def process_streaming(pipeline, audio, crossfade_sec, context_sec, mic_sr=48000, output_sr=48000):
    """Process audio in streaming mode with given crossfade."""
    rt_config = RealtimeConfig(
        mic_sample_rate=mic_sr,
        input_sample_rate=16000,
        output_sample_rate=output_sr,
        chunk_sec=0.35,
        f0_method="rmvpe",
        chunking_mode="wokada",
        context_sec=context_sec,
        crossfade_sec=crossfade_sec,
        use_sola=True,
        rvc_overlap_sec=0.0,
        index_rate=0.0,
        voice_gate_mode="off",
        use_feature_cache=False,
        use_adaptive_parameters=False,
    )

    changer = RealtimeVoiceChanger(pipeline, config=rt_config)
    pipeline.clear_cache()
    changer._recalculate_buffers()
    changer._running = True

    # Large output buffer
    expected_chunks = int(len(audio) / mic_sr / rt_config.chunk_sec) + 10
    max_output = expected_chunks * int(output_sr * rt_config.chunk_sec) * 3
    changer.output_buffer.set_max_latency(max_output)

    # Process in 20ms blocks
    input_block_size = int(mic_sr * 0.02)
    output_block_size = int(output_sr * 0.02)
    outputs = []

    pos = 0
    while pos < len(audio):
        block = audio[pos:pos + input_block_size]
        if len(block) < input_block_size:
            block = np.pad(block, (0, input_block_size - len(block)))
        pos += input_block_size

        changer.process_input_chunk(block)
        while changer.process_next_chunk():
            pass
        changer.get_output_chunk(0)

    # Final processing
    while changer.process_next_chunk():
        changer.get_output_chunk(0)
    changer.flush_final_sola_buffer()
    changer.get_output_chunk(0)

    # Retrieve output
    while changer.output_buffer.available > 0:
        outputs.append(changer.get_output_chunk(output_block_size))

    if outputs:
        return np.concatenate(outputs)
    return np.array([], dtype=np.float32)


def compute_envelope_correlation(audio1, audio2, frame_size=480):
    """Compute correlation between energy envelopes."""
    min_len = min(len(audio1), len(audio2))
    audio1 = audio1[:min_len]
    audio2 = audio2[:min_len]

    num_frames = min_len // frame_size
    if num_frames == 0:
        return 0.0

    env1 = np.zeros(num_frames, dtype=np.float32)
    env2 = np.zeros(num_frames, dtype=np.float32)

    for i in range(num_frames):
        frame1 = audio1[i * frame_size:(i + 1) * frame_size]
        frame2 = audio2[i * frame_size:(i + 1) * frame_size]
        env1[i] = np.sqrt(np.mean(frame1 ** 2))
        env2[i] = np.sqrt(np.mean(frame2 ** 2))

    if np.std(env1) < 1e-8 or np.std(env2) < 1e-8:
        return 0.0

    return float(np.corrcoef(env1, env2)[0, 1])


def main():
    print("Loading model...")
    model_path = "sample_data/hogaraka/hogarakav2.pth"
    pipeline = RVCPipeline(model_path, use_compile=False)

    print("Loading audio...")
    audio = load_audio("sample_data/seki.wav", max_sec=20.0)

    print(f"Audio: {len(audio)/48000:.2f}s @ 48kHz")
    print()

    # Process batch (gold standard)
    print("Processing batch...")
    batch_output = process_batch(pipeline, audio)
    print(f"Batch output: {len(batch_output)} samples")
    print()

    # Test different crossfade values
    test_configs = [
        # (crossfade_sec, context_sec, description)
        (0.05, 0.05, "Short (50ms)"),
        (0.08, 0.08, "Medium-Short (80ms)"),
        (0.10, 0.10, "Medium (100ms)"),
        (0.119, 0.119, "Current Default (119ms)"),
        (0.15, 0.15, "Medium-Long (150ms)"),
        (0.20, 0.20, "Long (200ms)"),
    ]

    results = []

    for crossfade_sec, context_sec, desc in test_configs:
        print(f"Testing {desc}: crossfade={crossfade_sec}s, context={context_sec}s")

        try:
            streaming_output = process_streaming(
                pipeline, audio, crossfade_sec, context_sec
            )

            # Compute correlation
            corr = compute_envelope_correlation(batch_output, streaming_output)
            length_ratio = len(streaming_output) / len(batch_output) * 100

            results.append({
                'crossfade': crossfade_sec,
                'context': context_sec,
                'desc': desc,
                'corr': corr,
                'length': length_ratio,
            })

            print(f"  Correlation: {corr:.4f}")
            print(f"  Length: {length_ratio:.1f}%")
            print()
        except Exception as e:
            print(f"  ERROR: {e}")
            print()

    # Print summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Config':<30} {'Correlation':>12} {'Length %':>10}")
    print("-" * 80)

    for r in results:
        marker = " *BEST*" if r['corr'] == max(x['corr'] for x in results) else ""
        print(f"{r['desc']:<30} {r['corr']:>12.4f} {r['length']:>10.1f}{marker}")

    print("=" * 80)


if __name__ == "__main__":
    main()
