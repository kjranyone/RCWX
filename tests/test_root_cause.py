"""Root cause: HuBERT -1 frame offset causes 12.5% time-stretch per chunk.

HuBERT produces (input_samples / 320) - 1 frames for input_samples.
infer_streaming() trims output assuming (input_samples / 320) frames.
The 1-frame deficit (= 800 samples @ 40kHz) is "fixed" by resample(),
which time-stretches each chunk by ~12.5% for typical 150ms chunks.
This shifts pitch down by ~2.3 semitones per chunk.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def main():
    from rcwx.config import RCWXConfig
    from rcwx.pipeline.inference import RVCPipeline, highpass_filter

    config = RCWXConfig.load()
    pipeline = RVCPipeline(config.last_model_path, device=config.device, use_compile=False)
    pipeline.load()
    model_sr = pipeline.sample_rate
    hubert_hop = 320

    print("=" * 70)
    print("Root cause: HuBERT -1 frame offset -> per-chunk time-stretch")
    print("=" * 70)

    # ================================================================
    # Test 1: HuBERT frame count formula verification
    # ================================================================
    print("\n--- Test 1: HuBERT frame count = input/320 - 1 ---")
    for n_samples in [3200, 4160, 5760, 8000, 16000, 17600]:
        audio_t = torch.randn(1, n_samples).to(pipeline.device)
        with torch.no_grad(), torch.autocast(device_type=pipeline.device, dtype=pipeline.dtype):
            feats = pipeline.hubert.extract(audio_t, output_layer=12, output_dim=768)
        naive = n_samples // hubert_hop
        actual = feats.shape[1]
        print(f"  {n_samples:6d} samples -> {actual:3d} frames (naive={naive}, diff={actual - naive})")

    # ================================================================
    # Test 2: Trimming deficit per chunk size
    # ================================================================
    print("\n--- Test 2: Per-chunk output deficit ---")
    print(f"  model_sr={model_sr}, HuBERT hop={hubert_hop}")

    overlap_16k = 1600  # 100ms overlap
    t_pad = 800

    for chunk_sec in [0.10, 0.15, 0.20, 0.30, 0.40]:
        hop_16k = ((int(chunk_sec * 16000) + hubert_hop - 1) // hubert_hop) * hubert_hop
        total_16k = overlap_16k + hop_16k
        padded = t_pad + total_16k + t_pad
        remainder = padded % hubert_hop
        extra_pad = (hubert_hop - remainder) if remainder != 0 else 0

        # HuBERT frame count
        hubert_frames = (padded + extra_pad) // hubert_hop - 1  # -1!
        feature_frames = hubert_frames * 2  # 2x interpolation
        synth_samples = feature_frames * (model_sr // 100)

        # Trim (infer_streaming style)
        t_pad_tgt = int(t_pad * model_sr / 16000)
        overlap_tgt = int(overlap_16k * model_sr / 16000)
        trim_left = t_pad_tgt + overlap_tgt
        trim_right = t_pad_tgt
        remaining = synth_samples - trim_left - trim_right

        # Expected
        expected = int(hop_16k * model_sr / 16000)
        deficit = expected - remaining
        deficit_pct = deficit / expected * 100

        # Length adjustment behavior
        if remaining >= expected:
            adj = "TRIM (truncate)"
        elif deficit < expected * 0.1:
            adj = f"ZERO-PAD +{deficit} samples ({deficit * 1000 / model_sr:.1f}ms silence)"
        else:
            stretch_pct = (expected / remaining - 1) * 100
            adj = f"RESAMPLE {remaining}->{expected} ({stretch_pct:.1f}% stretch = {-12 * np.log2(remaining / expected):.1f} semitones)"

        print(
            f"  chunk={chunk_sec:.2f}s: hop={hop_16k}, total={total_16k}, "
            f"hubert={hubert_frames}fr, synth={synth_samples}, "
            f"trim={trim_left}+{trim_right}, remain={remaining}, "
            f"expected={expected}, deficit={deficit} ({deficit_pct:.1f}%) -> {adj}"
        )

    # ================================================================
    # Test 3: Actual infer_streaming() output length check
    # ================================================================
    print("\n--- Test 3: Actual infer_streaming() output lengths ---")

    # Load real audio
    test_file = Path(__file__).parent.parent / "sample_data" / "seki.wav"
    if not test_file.exists():
        test_file = Path(__file__).parent.parent / "sample_data" / "sustained_voice.wav"
    if test_file.exists():
        from scipy.io import wavfile
        from rcwx.audio.resample import resample

        sr, data = wavfile.read(str(test_file))
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != 16000:
            data = resample(data, sr, 16000)
        audio_16k = data[:32000]  # 2 seconds
    else:
        audio_16k = np.random.randn(32000).astype(np.float32) * 0.1

    chunk_sec = 0.15
    overlap_sec = 0.10
    hop_samples = ((int(chunk_sec * 16000) + hubert_hop - 1) // hubert_hop) * hubert_hop
    overlap_samples = ((int(overlap_sec * 16000) + hubert_hop - 1) // hubert_hop) * hubert_hop
    expected_per_hop = int(hop_samples * model_sr / 16000)

    aligned_len = (len(audio_16k) // hubert_hop) * hubert_hop
    audio = audio_16k[:aligned_len]

    pipeline.clear_cache()
    pos = 0
    chunk_idx = 0
    output_lengths = []
    while pos + hop_samples <= len(audio):
        new_hop = audio[pos:pos + hop_samples]

        if chunk_idx == 0:
            reflection = new_hop[:overlap_samples][::-1].copy()
            chunk = np.concatenate([reflection, new_hop])
            ovl = overlap_samples
        else:
            ovl_start = max(0, pos - overlap_samples)
            overlap_audio = audio[ovl_start:pos]
            al = (len(overlap_audio) // hubert_hop) * hubert_hop
            overlap_audio = overlap_audio[-al:] if al > 0 else overlap_audio
            chunk = np.concatenate([overlap_audio, new_hop])
            ovl = len(overlap_audio)

        out = pipeline.infer_streaming(
            audio_16k=chunk, overlap_samples=ovl,
            pitch_shift=0, f0_method="fcpe", voice_gate_mode="off",
        )
        output_lengths.append(len(out))
        pos += hop_samples
        chunk_idx += 1

    print(f"  chunk_sec={chunk_sec}, overlap_sec={overlap_sec}")
    print(f"  hop_16k={hop_samples}, expected_output={expected_per_hop}")
    print(f"  Chunks processed: {chunk_idx}")
    print(f"  Output lengths: {output_lengths[:10]}...")
    print(f"  All equal to expected? {all(l == expected_per_hop for l in output_lengths)}")
    unique_lengths = set(output_lengths)
    for ul in sorted(unique_lengths):
        count = output_lengths.count(ul)
        diff = ul - expected_per_hop
        print(f"    length={ul} (diff={diff:+d}): {count} chunks")

    # ================================================================
    # Test 4: Impact of resampling on pitch
    # ================================================================
    print("\n--- Test 4: Pitch impact of length adjustment ---")

    # Generate a pure 220Hz tone, process through pipeline, measure output pitch
    sr_16k = 16000
    tone_dur = 0.5  # 500ms
    t = np.arange(int(sr_16k * tone_dur), dtype=np.float32) / sr_16k
    tone_220 = 0.5 * np.sin(2 * np.pi * 220 * t)
    tone_aligned = tone_220[: (len(tone_220) // hubert_hop) * hubert_hop]

    # Batch
    pipeline.clear_cache()
    batch_out = pipeline.infer(
        tone_aligned, input_sr=16000, pitch_shift=0,
        f0_method="none", use_feature_cache=False, voice_gate_mode="off",
    )

    # Single streaming (no overlap)
    pipeline.clear_cache()
    stream_out = pipeline.infer_streaming(
        tone_aligned, overlap_samples=0, pitch_shift=0,
        f0_method="none", voice_gate_mode="off",
    )

    # Measure fundamental frequency via autocorrelation
    def estimate_f0(audio, sr):
        if len(audio) < sr // 50:
            return 0.0
        # Use center portion
        center = len(audio) // 4
        seg = audio[center:center + sr // 4]
        if len(seg) < 100:
            return 0.0
        # Normalize
        seg = seg - np.mean(seg)
        if np.max(np.abs(seg)) < 1e-6:
            return 0.0
        seg = seg / np.max(np.abs(seg))
        # Autocorrelation
        corr_full = np.correlate(seg, seg, mode="full")
        corr_half = corr_full[len(seg):]
        # Find first peak after zero crossing
        min_lag = sr // 500  # 500Hz max
        max_lag = sr // 50   # 50Hz min
        if max_lag > len(corr_half):
            max_lag = len(corr_half)
        search = corr_half[min_lag:max_lag]
        if len(search) == 0:
            return 0.0
        peak = np.argmax(search) + min_lag
        if peak > 0:
            return sr / peak
        return 0.0

    batch_f0 = estimate_f0(batch_out, model_sr)
    stream_f0 = estimate_f0(stream_out, model_sr)
    print(f"  Input: 220Hz pure tone")
    print(f"  Batch output F0: {batch_f0:.1f}Hz")
    print(f"  Stream output F0: {stream_f0:.1f}Hz")
    if batch_f0 > 0 and stream_f0 > 0:
        ratio = stream_f0 / batch_f0
        semitones = 12 * np.log2(ratio) if ratio > 0 else 0
        print(f"  Ratio: {ratio:.4f} ({semitones:+.2f} semitones)")

    # ================================================================
    # Test 5: Per-chunk pitch measurement
    # ================================================================
    print("\n--- Test 5: Per-chunk pitch of streaming output ---")

    # Use longer tone
    tone_2s = np.sin(2 * np.pi * 220 * np.arange(32000, dtype=np.float32) / 16000).astype(np.float32) * 0.5
    tone_2s_aligned = tone_2s[:(len(tone_2s) // hubert_hop) * hubert_hop]

    pipeline.clear_cache()
    pos = 0
    chunk_idx = 0
    chunk_pitches = []
    while pos + hop_samples <= len(tone_2s_aligned):
        new_hop = tone_2s_aligned[pos:pos + hop_samples]
        if chunk_idx == 0:
            reflection = new_hop[:overlap_samples][::-1].copy()
            chunk = np.concatenate([reflection, new_hop])
            ovl = overlap_samples
        else:
            ovl_start = max(0, pos - overlap_samples)
            ov = tone_2s_aligned[ovl_start:pos]
            al = (len(ov) // hubert_hop) * hubert_hop
            ov = ov[-al:] if al > 0 else ov
            chunk = np.concatenate([ov, new_hop])
            ovl = len(ov)

        out = pipeline.infer_streaming(
            chunk, overlap_samples=ovl, pitch_shift=0,
            f0_method="none", voice_gate_mode="off",
        )
        f0_est = estimate_f0(out, model_sr)
        chunk_pitches.append(f0_est)
        pos += hop_samples
        chunk_idx += 1

    print(f"  Per-chunk F0 estimates (input=220Hz):")
    for i, p in enumerate(chunk_pitches):
        if p > 0:
            ratio = p / 220
            st = 12 * np.log2(ratio)
            print(f"    chunk {i}: {p:.1f}Hz (ratio={ratio:.4f}, {st:+.2f} semitones)")
        else:
            print(f"    chunk {i}: undetected")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"HuBERT frame count: input/320 - 1 (always 1 frame less than naive)")
    print(f"Per-chunk output deficit: 800 samples ({800 * 1000 / model_sr:.0f}ms) at model_sr={model_sr}")
    print(f"For chunk_sec=0.15: deficit/expected = 800/{expected_per_hop} = {800 / expected_per_hop * 100:.1f}%")
    print(f"Length adjustment: resample (time-stretch) for deficit > 10%")
    print(f"                  zero-pad (silence) for deficit <= 10%")
    print()
    print("ROOT CAUSE:")
    print("  infer_streaming() trim calculation assumes HuBERT produces")
    print("  input/320 frames, but actual is input/320 - 1.")
    print("  The 1-frame deficit becomes 800 output samples that are")
    print("  compensated by resampling (time-stretch), causing")
    print(f"  ~{800 / (expected_per_hop - 800) * 100:.1f}% pitch shift per chunk.")


if __name__ == "__main__":
    main()
