"""Simulate the full realtime pipeline offline and compare with batch.

Feeds debug_audio/01_input_raw.wav through the same processing stages
as RealtimeVoiceChangerUnified (resample, chunk, overlap, infer_streaming,
resample back, SOLA) and writes both batch and simulated-realtime outputs
for listening comparison.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def main():
    from rcwx.audio.resample import StatefulResampler, resample
    from rcwx.audio.sola import SolaState, sola_crossfade
    from rcwx.config import RCWXConfig
    from rcwx.pipeline.inference import RVCPipeline

    # ---- load audio ----
    audio_path = Path(__file__).parent.parent / "debug_audio" / "01_input_raw.wav"
    if not audio_path.exists():
        print(f"ERROR: {audio_path} not found")
        return
    file_sr, raw = wavfile.read(str(audio_path))
    if raw.dtype == np.int16:
        raw = raw.astype(np.float32) / 32768.0
    if raw.ndim > 1:
        raw = raw.mean(axis=1)
    raw = raw.astype(np.float32)
    print(f"Input: {audio_path.name}, sr={file_sr}, duration={len(raw)/file_sr:.2f}s")

    # ---- load pipeline ----
    config = RCWXConfig.load()
    pipeline = RVCPipeline(config.last_model_path, device=config.device, use_compile=False)
    pipeline.load()
    model_sr = pipeline.sample_rate
    print(f"Model: sr={model_sr}, device={pipeline.device}")

    # ---- config (same defaults as GUI) ----
    mic_sr = 48000
    output_sr = 48000
    chunk_sec = 0.15
    overlap_sec = 0.16
    crossfade_sec = 0.05
    sola_search_ms = 10.0
    f0_method = "fcpe"
    hubert_hop = 320

    def align_to_hop(samples: int, hop: int) -> int:
        return ((samples + hop - 1) // hop) * hop

    hop_16k = align_to_hop(int(chunk_sec * 16000), hubert_hop)
    overlap_16k = align_to_hop(int(overlap_sec * 16000), hubert_hop)

    print(f"Config: chunk={chunk_sec}s, overlap={overlap_sec}s, "
          f"hop_16k={hop_16k}, overlap_16k={overlap_16k}")

    # ==================================================================
    # Batch processing (gold standard)
    # ==================================================================
    print("\n--- Batch processing ---")
    audio_16k_full = resample(raw, file_sr, 16000)
    pipeline.clear_cache()
    t0 = time.perf_counter()
    batch_out = pipeline.infer(
        audio_16k_full, input_sr=16000, pitch_shift=0, f0_method=f0_method,
        use_feature_cache=False, voice_gate_mode="off",
    )
    batch_ms = (time.perf_counter() - t0) * 1000
    # Resample to output rate
    batch_48k = resample(batch_out, model_sr, output_sr)
    print(f"  batch output: {len(batch_out)} @ {model_sr}Hz -> {len(batch_48k)} @ {output_sr}Hz "
          f"({batch_ms:.0f}ms)")

    # ==================================================================
    # Simulated realtime pipeline
    # ==================================================================
    print("\n--- Simulated realtime pipeline ---")

    # Step 1: Resample file_sr -> mic_sr (simulate mic input at 48kHz)
    audio_mic = resample(raw, file_sr, mic_sr)

    # Step 2: StatefulResampler 48k -> 16k (same as realtime)
    input_resampler = StatefulResampler(mic_sr, 16000)
    output_resampler = StatefulResampler(model_sr, output_sr)

    # Step 3: SOLA state
    sola_state = SolaState(
        crossfade_samples=int(output_sr * crossfade_sec),
        search_samples=int(output_sr * sola_search_ms / 1000),
    )

    # Step 4: SOLA extra samples (at model_sr) — crossfade + search
    # Matches realtime_unified.py: produce enough extra output so SOLA has
    # a full crossfade+search region overlapping with previous chunk's tail.
    crossfade_out = int(output_sr * crossfade_sec)
    search_out = int(output_sr * sola_search_ms / 1000)
    sola_extra_out = crossfade_out + search_out
    sola_extra_model = int(sola_extra_out * model_sr / output_sr)
    print(f"SOLA extra: {sola_extra_model} samples @ {model_sr}Hz "
          f"({sola_extra_model * 1000 / model_sr:.1f}ms) "
          f"[cf={crossfade_out}+search={search_out}={sola_extra_out} @ {output_sr}Hz]")

    # Step 5: Simulate chunked input (hop-sized pieces at mic rate)
    hop_mic = int(hop_16k * mic_sr / 16000)
    overlap_buf = None
    chunk_idx = 0
    realtime_outputs = []
    total_infer_ms = 0.0

    pos = 0
    while pos + hop_mic <= len(audio_mic):
        hop_audio = audio_mic[pos:pos + hop_mic]
        pos += hop_mic

        # Resample 48k -> 16k (stateful, like realtime)
        hop_16k_audio = input_resampler.resample_chunk(hop_audio)

        # Align to HuBERT hop
        aligned_len = align_to_hop(len(hop_16k_audio), hubert_hop)
        if len(hop_16k_audio) < aligned_len:
            hop_16k_audio = np.pad(hop_16k_audio, (0, aligned_len - len(hop_16k_audio)))
        elif len(hop_16k_audio) > aligned_len:
            hop_16k_audio = hop_16k_audio[:aligned_len]

        # Assemble chunk with overlap
        if overlap_buf is not None:
            overlap = overlap_buf
            if len(overlap) % hubert_hop != 0:
                al = align_to_hop(len(overlap), hubert_hop)
                if len(overlap) < al:
                    overlap = np.pad(overlap, (al - len(overlap), 0))
                else:
                    overlap = overlap[-al:]
            chunk_16k = np.concatenate([overlap, hop_16k_audio])
            overlap_samples = len(overlap)
        else:
            # First chunk: reflection padding
            # Use min(overlap_16k, hop_16k) to handle overlap > hop case
            actual_overlap = min(overlap_16k, len(hop_16k_audio))
            if actual_overlap > 0:
                reflection = hop_16k_audio[:actual_overlap][::-1].copy()
                chunk_16k = np.concatenate([reflection, hop_16k_audio])
                overlap_samples = actual_overlap
            else:
                chunk_16k = hop_16k_audio
                overlap_samples = 0

        # Store tail for next overlap (accumulating for overlap > hop)
        if overlap_16k > 0:
            if overlap_buf is not None:
                combined = np.concatenate([overlap_buf, hop_16k_audio])
                overlap_buf = combined[-overlap_16k:]
            else:
                tail = min(overlap_16k, len(hop_16k_audio))
                overlap_buf = hop_16k_audio[-tail:].copy()
        else:
            overlap_buf = None

        # Inference
        t0 = time.perf_counter()
        out_model = pipeline.infer_streaming(
            audio_16k=chunk_16k,
            overlap_samples=overlap_samples,
            pitch_shift=0,
            f0_method=f0_method,
            voice_gate_mode="off",
            use_parallel_extraction=True,
            sola_extra_samples=sola_extra_model,
        )
        infer_ms = (time.perf_counter() - t0) * 1000
        total_infer_ms += infer_ms

        # Resample model_sr -> 48k (stateful)
        out_48k = output_resampler.resample_chunk(out_model)

        # Soft clip
        max_val = np.max(np.abs(out_48k)) if len(out_48k) else 0.0
        if max_val > 1.0:
            out_48k = np.tanh(out_48k)

        # SOLA crossfade
        out_sola = sola_crossfade(out_48k, sola_state)

        # No hard trim — let ring buffer absorb small surplus
        realtime_outputs.append(out_sola)
        chunk_idx += 1

        if chunk_idx <= 3 or chunk_idx % 5 == 0:
            print(f"  chunk {chunk_idx:2d}: hop_16k={len(hop_16k_audio)}, "
                  f"overlap={overlap_samples}, out_model={len(out_model)}, "
                  f"out_48k={len(out_48k)}, out_sola={len(out_sola)}, "
                  f"infer={infer_ms:.0f}ms")

    realtime_48k = np.concatenate(realtime_outputs)
    print(f"\n  Total chunks: {chunk_idx}")
    print(f"  Total inference: {total_infer_ms:.0f}ms "
          f"(avg {total_infer_ms/max(chunk_idx,1):.0f}ms/chunk, "
          f"budget={chunk_sec*1000:.0f}ms)")
    print(f"  Realtime output: {len(realtime_48k)} samples @ {output_sr}Hz "
          f"= {len(realtime_48k)/output_sr:.3f}s")
    print(f"  Batch output:    {len(batch_48k)} samples @ {output_sr}Hz "
          f"= {len(batch_48k)/output_sr:.3f}s")

    # ==================================================================
    # Comparison
    # ==================================================================
    print("\n--- Comparison ---")
    n = min(len(batch_48k), len(realtime_48k))
    if n > 0:
        corr = np.corrcoef(batch_48k[:n], realtime_48k[:n])[0, 1]
        max_diff = np.max(np.abs(batch_48k[:n] - realtime_48k[:n]))
        rms_batch = np.sqrt(np.mean(batch_48k[:n] ** 2))
        rms_rt = np.sqrt(np.mean(realtime_48k[:n] ** 2))
        print(f"  Overall corr={corr:.4f} (context-limited, <1.0 expected)")
        print(f"  max_diff={max_diff:.4f}")
        print(f"  RMS batch={rms_batch:.4f}, realtime={rms_rt:.4f}")
        print(f"  Length diff: {len(batch_48k) - len(realtime_48k)} samples "
              f"({(len(batch_48k) - len(realtime_48k))/output_sr*1000:.1f}ms)")

        # Check for silence (underrun symptom)
        chunk_48k = int(output_sr * chunk_sec)
        silent_chunks = 0
        total_check = min(len(realtime_48k), len(batch_48k))
        for i in range(0, total_check, chunk_48k):
            seg = realtime_48k[i:i + chunk_48k]
            if np.max(np.abs(seg)) < 0.001:
                silent_chunks += 1
        print(f"  Silent chunks in realtime: {silent_chunks}")
    else:
        print("  ERROR: no output to compare")

    # ==================================================================
    # Save outputs
    # ==================================================================
    out_dir = Path("test_output/realtime_simulation")
    out_dir.mkdir(parents=True, exist_ok=True)

    wavfile.write(str(out_dir / "batch_48k.wav"), output_sr, batch_48k)
    wavfile.write(str(out_dir / "realtime_48k.wav"), output_sr, realtime_48k)

    # Also save a "difference" wav for analysis
    if n > 0:
        diff_signal = batch_48k[:n] - realtime_48k[:n]
        wavfile.write(str(out_dir / "diff_48k.wav"), output_sr, diff_signal)

    print(f"\n  Saved to {out_dir}/")
    print(f"    batch_48k.wav     - batch (gold standard)")
    print(f"    realtime_48k.wav  - simulated realtime pipeline")
    print(f"    diff_48k.wav      - difference signal")
    print()
    print("Listen to both files. realtime_48k.wav should sound clean")
    print("without pitch artifacts, glitches, or silence gaps.")


if __name__ == "__main__":
    main()
