"""
Reproduce GUI realtime pipeline on a WAV file.

This simulates the same chunking, buffering, and SOLA behavior as the GUI
by using RealtimeVoiceChanger with the current config.json.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from rcwx.audio.resample import resample
from rcwx.config import RCWXConfig
from rcwx.pipeline.inference import RVCPipeline
from rcwx.pipeline.realtime import RealtimeConfig, RealtimeVoiceChanger


def _load_audio(path: Path, target_sr: int) -> np.ndarray:
    sr, audio = wavfile.read(path)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    else:
        audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != target_sr:
        audio = resample(audio, sr, target_sr)
    return audio


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2))) if len(x) else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce GUI realtime pipeline")
    parser.add_argument("input", type=Path, help="Input WAV file")
    parser.add_argument("--output", type=Path, default=Path("test_output/gui_repro.wav"))
    parser.add_argument("--seconds", type=float, default=0.0, help="Trim input to N seconds (0 = full)")
    args = parser.parse_args()

    config = RCWXConfig.load()
    if not config.last_model_path:
        raise SystemExit("No model configured in config.json")

    mic_sr = config.audio.sample_rate
    out_sr = config.audio.output_sample_rate

    audio = _load_audio(args.input, mic_sr)
    if args.seconds and args.seconds > 0:
        audio = audio[: int(mic_sr * args.seconds)]

    pipeline = RVCPipeline(
        config.last_model_path,
        device=config.device,
        use_compile=False,
    )
    pipeline.load()

    rt_config = RealtimeConfig(
        mic_sample_rate=mic_sr,
        input_sample_rate=16000,
        output_sample_rate=out_sr,
        chunk_sec=config.audio.chunk_sec,
        context_sec=config.inference.context_sec,
        lookahead_sec=config.inference.lookahead_sec,
        crossfade_sec=config.inference.crossfade_sec,
        use_sola=config.inference.use_sola,
        prebuffer_chunks=config.audio.prebuffer_chunks,
        buffer_margin=config.audio.buffer_margin,
        input_gain_db=config.audio.input_gain_db,
        input_channel_selection=config.audio.input_channel_selection,
        pitch_shift=config.inference.pitch_shift,
        use_f0=config.inference.use_f0,
        f0_method=config.inference.f0_method,
        index_rate=config.inference.index_ratio if config.inference.use_index else 0.0,
        index_k=config.inference.index_k,
        voice_gate_mode=config.inference.voice_gate_mode,
        energy_threshold=config.inference.energy_threshold,
        use_feature_cache=config.inference.use_feature_cache,
        use_parallel_extraction=config.inference.use_parallel_extraction,
        chunking_mode=config.inference.chunking_mode,
        rvc_overlap_sec=config.inference.crossfade_sec,
    )

    changer = RealtimeVoiceChanger(pipeline, config=rt_config)
    pipeline.clear_cache()
    changer._recalculate_buffers()
    changer._running = True

    block_size = int(mic_sr * 0.02)  # 20ms callback blocks
    outputs: list[np.ndarray] = []
    output_frames = int(out_sr * 0.02)

    pos = 0
    while pos < len(audio):
        block = audio[pos : pos + block_size]
        if len(block) < block_size:
            block = np.pad(block, (0, block_size - len(block)))
        pos += block_size
        changer.process_input_chunk(block)
        while changer.process_next_chunk():
            # Drain to output buffer and render like GUI
            out_block = changer.get_output_chunk(output_frames)
            if len(out_block):
                outputs.append(out_block)

    changer.flush_final_sola_buffer()
    # Drain remaining output
    for _ in range(1000):
        out_block = changer.get_output_chunk(output_frames)
        if len(out_block) == 0:
            break
        outputs.append(out_block)

    if not outputs:
        raise SystemExit("No output produced")

    output = np.concatenate(outputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(args.output, out_sr, (output * 32767).astype(np.int16))

    # Boundary RMS analysis (20ms windows)
    window = int(out_sr * 0.02)
    ratios = []
    for i in range(1, len(outputs)):
        prev = outputs[i - 1]
        curr = outputs[i]
        prev_tail = prev[-window:] if len(prev) >= window else prev
        curr_head = curr[:window] if len(curr) >= window else curr
        r_prev = _rms(prev_tail)
        r_curr = _rms(curr_head)
        ratios.append((r_curr / r_prev) if r_prev > 1e-8 else 0.0)

    if ratios:
        print("Boundary RMS ratio stats:")
        print("  min:", min(ratios))
        print("  mean:", sum(ratios) / len(ratios))
    else:
        print("Boundary RMS ratio stats: n/a")

    print("Output:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
