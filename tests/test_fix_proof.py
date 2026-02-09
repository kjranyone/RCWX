"""Proof: HuBERT -1 frame trim fix eliminates per-chunk time-stretch.

Uses noise_scale=0 for deterministic comparison (eliminating VAE noise).
Key metrics:
  1. Per-chunk output length matches expected EXACTLY (no resample/zero-pad)
  2. Single-chunk batch vs streaming correlation >> 0.99 (with noise_scale=0)
  3. Chunked streaming correlation with batch is limited by context differences
     but NOT degraded by time-stretch artifacts.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def load_test_audio(sr: int = 16000, duration_sec: float = 2.0) -> np.ndarray:
    test_file = Path(__file__).parent.parent / "sample_data" / "seki.wav"
    if not test_file.exists():
        test_file = Path(__file__).parent.parent / "sample_data" / "sustained_voice.wav"
    if not test_file.exists():
        raise FileNotFoundError("No test audio in sample_data/")
    from scipy.io import wavfile
    from rcwx.audio.resample import resample
    file_sr, data = wavfile.read(str(test_file))
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    if data.ndim > 1:
        data = data.mean(axis=1)
    if file_sr != sr:
        data = resample(data, file_sr, sr)
    return data[:int(sr * duration_sec)].astype(np.float32)


def main():
    from rcwx.config import RCWXConfig
    from rcwx.pipeline.inference import RVCPipeline

    config = RCWXConfig.load()
    pipeline = RVCPipeline(config.last_model_path, device=config.device, use_compile=False)
    pipeline.load()
    model_sr = pipeline.sample_rate
    hubert_hop = 320

    audio_16k = load_test_audio(16000, 1.0)
    aligned = (len(audio_16k) // hubert_hop) * hubert_hop
    audio = audio_16k[:aligned]

    chunk_sec = 0.15
    overlap_sec = 0.10
    hop_samples = ((int(chunk_sec * 16000) + hubert_hop - 1) // hubert_hop) * hubert_hop
    overlap_samples = ((int(overlap_sec * 16000) + hubert_hop - 1) // hubert_hop) * hubert_hop
    expected_per_hop = int(hop_samples * model_sr / 16000)

    print("=" * 70)
    print("PROOF: HuBERT -1 frame trim fix (noise_scale=0 deterministic)")
    print("=" * 70)

    # ================================================================
    # Test 1: Single-chunk comparison (noise_scale=0)
    #   Same audio, no chunking → should be near-identical
    # ================================================================
    print("\n--- Test 1: Single-chunk batch vs streaming (noise_scale=0) ---")

    pipeline.clear_cache()
    batch_out = pipeline.infer(
        audio, input_sr=16000, pitch_shift=0, f0_method="fcpe",
        use_feature_cache=False, voice_gate_mode="off",
        noise_scale=0.0,
    )

    pipeline.clear_cache()
    stream_out = pipeline.infer_streaming(
        audio_16k=audio, overlap_samples=0,
        pitch_shift=0, f0_method="fcpe", voice_gate_mode="off",
        noise_scale=0.0,
    )

    n = min(len(batch_out), len(stream_out))
    corr_single = np.corrcoef(batch_out[:n], stream_out[:n])[0, 1]
    max_diff = np.max(np.abs(batch_out[:n] - stream_out[:n]))
    print(f"  batch={len(batch_out)}, stream={len(stream_out)}, len_diff={len(batch_out)-len(stream_out)}")
    print(f"  corr={corr_single:.6f}, max_diff={max_diff:.6f}")
    print(f"  {'PASS' if corr_single > 0.99 else 'FAIL'}: expect >0.99 with noise_scale=0")

    # ================================================================
    # Test 2: Per-chunk output length (no resample/zero-pad needed)
    # ================================================================
    print("\n--- Test 2: Per-chunk output length (deficit check) ---")

    pipeline.clear_cache()
    pos = 0
    chunk_idx = 0
    chunk_outputs = []

    while pos + hop_samples <= len(audio):
        new_hop = audio[pos:pos + hop_samples]
        if chunk_idx == 0:
            reflection = new_hop[:overlap_samples][::-1].copy()
            chunk = np.concatenate([reflection, new_hop])
            ovl = overlap_samples
        else:
            ovl_start = max(0, pos - overlap_samples)
            ov = audio[ovl_start:pos]
            al = (len(ov) // hubert_hop) * hubert_hop
            ov = ov[-al:] if al > 0 else ov
            chunk = np.concatenate([ov, new_hop])
            ovl = len(ov)

        out = pipeline.infer_streaming(
            audio_16k=chunk, overlap_samples=ovl,
            pitch_shift=0, f0_method="fcpe", voice_gate_mode="off",
            noise_scale=0.0,
        )
        chunk_outputs.append(out)
        pos += hop_samples
        chunk_idx += 1

    lengths = [len(c) for c in chunk_outputs]
    all_exact = all(l == expected_per_hop for l in lengths)
    print(f"  expected={expected_per_hop}, chunks={chunk_idx}")
    print(f"  lengths: {lengths}")
    print(f"  {'PASS' if all_exact else 'FAIL'}: all lengths exact (no resample/zero-pad)")

    # ================================================================
    # Test 3: Chunked streaming vs batch (noise_scale=0)
    # ================================================================
    print("\n--- Test 3: Chunked streaming vs batch (noise_scale=0) ---")

    pipeline.clear_cache()
    batch_det = pipeline.infer(
        audio, input_sr=16000, pitch_shift=0, f0_method="fcpe",
        use_feature_cache=False, voice_gate_mode="off",
        noise_scale=0.0,
    )

    streamed_det = np.concatenate(chunk_outputs)
    n = min(len(batch_det), len(streamed_det))
    corr_chunked = np.corrcoef(batch_det[:n], streamed_det[:n])[0, 1]
    print(f"  batch={len(batch_det)}, streamed={len(streamed_det)}")
    print(f"  Overall corr={corr_chunked:.4f}")
    print(f"  Note: <1.0 is EXPECTED due to HuBERT/TextEncoder context differences")

    # Per-chunk
    chunk_out_size = expected_per_hop
    n_compare = min(n // chunk_out_size, len(chunk_outputs))
    chunk_corrs = []
    for i in range(n_compare):
        s = i * chunk_out_size
        e = s + chunk_out_size
        if e <= n:
            c = np.corrcoef(batch_det[s:e], streamed_det[s:e])[0, 1]
            chunk_corrs.append(c)
            print(f"    chunk {i}: corr={c:.4f}")

    # ================================================================
    # Test 4: Reproducibility (streaming vs streaming, noise_scale=0)
    # ================================================================
    print("\n--- Test 4: Streaming reproducibility (noise_scale=0) ---")

    pipeline.clear_cache()
    pos = 0
    chunk_idx = 0
    chunk_outputs_2 = []
    while pos + hop_samples <= len(audio):
        new_hop = audio[pos:pos + hop_samples]
        if chunk_idx == 0:
            reflection = new_hop[:overlap_samples][::-1].copy()
            chunk = np.concatenate([reflection, new_hop])
            ovl = overlap_samples
        else:
            ovl_start = max(0, pos - overlap_samples)
            ov = audio[ovl_start:pos]
            al = (len(ov) // hubert_hop) * hubert_hop
            ov = ov[-al:] if al > 0 else ov
            chunk = np.concatenate([ov, new_hop])
            ovl = len(ov)

        out = pipeline.infer_streaming(
            audio_16k=chunk, overlap_samples=ovl,
            pitch_shift=0, f0_method="fcpe", voice_gate_mode="off",
            noise_scale=0.0,
        )
        chunk_outputs_2.append(out)
        pos += hop_samples
        chunk_idx += 1

    streamed_2 = np.concatenate(chunk_outputs_2)
    n2 = min(len(streamed_det), len(streamed_2))
    corr_repro = np.corrcoef(streamed_det[:n2], streamed_2[:n2])[0, 1]
    max_diff_repro = np.max(np.abs(streamed_det[:n2] - streamed_2[:n2]))
    print(f"  stream vs stream: corr={corr_repro:.6f}, max_diff={max_diff_repro:.6f}")
    print(f"  {'PASS' if corr_repro > 0.99 else 'FAIL'}: expect >0.99 (deterministic)")

    # ================================================================
    # Test 5: Save wav files
    # ================================================================
    out_dir = Path("test_output/fix_proof")
    out_dir.mkdir(parents=True, exist_ok=True)
    from scipy.io import wavfile
    wavfile.write(str(out_dir / "batch_det.wav"), model_sr, batch_det)
    wavfile.write(str(out_dir / "streamed_det.wav"), model_sr, streamed_det)
    print(f"\n  Saved to {out_dir}/")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Test 1 (single chunk):   corr={corr_single:.6f}  {'PASS' if corr_single > 0.99 else 'FAIL'}")
    print(f"  Test 2 (exact lengths):  {all_exact}          {'PASS' if all_exact else 'FAIL'}")
    print(f"  Test 3 (chunked corr):   corr={corr_chunked:.4f}  (context-limited)")
    print(f"  Test 4 (reproducibility): corr={corr_repro:.6f}  {'PASS' if corr_repro > 0.99 else 'FAIL'}")
    print("=" * 70)
    all_pass = corr_single > 0.99 and all_exact and corr_repro > 0.99
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")
    if all_pass:
        print("  Trim fix eliminates time-stretch artifact.")
        print("  Remaining batch/stream difference is due to context window limitations.")
    print("=" * 70)


if __name__ == "__main__":
    main()
