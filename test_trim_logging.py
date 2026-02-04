"""Test with detailed trimming logs to identify double-trimming."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import logging
import numpy as np
from scipy.io import wavfile
from rcwx.pipeline.inference import RVCPipeline
from rcwx.pipeline.realtime import RealtimeVoiceChanger, RealtimeConfig
from rcwx.audio.resample import resample

# Enable detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s - %(name)s - %(message)s'
)


def load_audio(path: str, max_sec: float = 2.0) -> np.ndarray:
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


def main():
    print("=" * 80)
    print("TRIMMING LOG TEST")
    print("=" * 80)

    pipeline = RVCPipeline("sample_data/hogaraka/hogarakav2.pth", use_compile=False)
    pipeline.load()
    audio = load_audio("sample_data/seki.wav", max_sec=1.0)  # Very short for focused analysis
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

    pos = 0
    chunks_processed = 0

    print("\nProcessing chunks...\n")

    while pos < len(audio) and chunks_processed < 5:  # Process only first 5 chunks
        block = audio[pos:pos + input_block]
        if len(block) < input_block:
            block = np.pad(block, (0, input_block - len(block)))
        pos += input_block

        changer.process_input_chunk(block)

        while changer.process_next_chunk():
            chunks_processed += 1
            if chunks_processed >= 5:
                break

        changer.get_output_chunk(0)

    print("\n" + "=" * 80)
    print("Look for [TRIM] and [SOLA] logs above")
    print("If both appear for the same chunk → double trimming!")
    print("=" * 80)


if __name__ == "__main__":
    main()
