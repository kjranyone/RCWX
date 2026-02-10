"""Spectral analysis of synthesizer output — diagnose hoarse voice.

Measures:
  1. HNR (Harmonic-to-Noise Ratio) across noise_scale values
  2. Spectral tilt (high-freq energy relative to low)
  3. Batch vs streaming spectral difference
  4. F0 tracking accuracy on known input
  5. HuBERT feature quality: full vs short context
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import welch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

MODEL_PATH = Path(r"C:\lib\github\grand2-products\RCWX\model\kurumi\kurumi.pth")


def spectral_analysis(wav, sr, label):
    """Analyze spectral characteristics of output waveform."""
    freqs, psd = welch(wav, fs=sr, nperseg=2048, noverlap=1024)
    psd_db = 10 * np.log10(psd + 1e-12)

    # Estimate f0 from spectrum peak
    voice_band = (freqs > 80) & (freqs < 500)
    if np.any(voice_band):
        f0_est = freqs[voice_band][np.argmax(psd[voice_band])]
    else:
        f0_est = 200.0

    # HNR: harmonic power vs inter-harmonic noise power
    harmonic_power = 0.0
    noise_power = 0.0
    for i, f in enumerate(freqs):
        if f < 50 or f > 8000:
            continue
        nearest_harmonic = round(f / f0_est) * f0_est
        if abs(f - nearest_harmonic) < 20:  # within 20Hz of harmonic
            harmonic_power += psd[i]
        else:
            noise_power += psd[i]

    hnr = 10 * np.log10(harmonic_power / (noise_power + 1e-12))

    # Spectral tilt
    valid = (freqs > 100) & (freqs < 8000)
    if np.sum(valid) > 10:
        log_f = np.log10(freqs[valid])
        log_p = psd_db[valid]
        coeffs = np.polyfit(log_f, log_p, 1)
        tilt = coeffs[0]
    else:
        tilt = float("nan")

    # Band energy
    def band_energy(lo, hi):
        mask = (freqs >= lo) & (freqs < hi)
        return 10 * np.log10(np.sum(psd[mask]) + 1e-12)

    e_low = band_energy(50, 500)
    e_mid = band_energy(500, 2000)
    e_hi = band_energy(2000, 8000)
    e_vhi = band_energy(8000, sr // 2)

    rms = np.sqrt(np.mean(wav ** 2))
    peak = np.max(np.abs(wav))
    crest = peak / (rms + 1e-12)

    print(f"  {label}:")
    print(f"    RMS={rms:.4f}  Peak={peak:.4f}  Crest={crest:.1f}")
    print(f"    HNR={hnr:.1f}dB  Tilt={tilt:.1f}dB/dec")
    print(
        f"    Band: low(50-500)={e_low:.1f}dB  mid(500-2k)={e_mid:.1f}dB  "
        f"hi(2k-8k)={e_hi:.1f}dB  vhi(8k+)={e_vhi:.1f}dB"
    )
    return {
        "hnr": hnr, "tilt": tilt, "rms": rms, "crest": crest,
        "e_low": e_low, "e_mid": e_mid, "e_hi": e_hi, "e_vhi": e_vhi,
        "f0_est": f0_est,
    }


def _load_pipeline():
    if not MODEL_PATH.exists():
        return None
    from rcwx.pipeline.inference import RVCPipeline
    p = RVCPipeline(str(MODEL_PATH), device="xpu", dtype=torch.float16, use_compile=False)
    p.load()
    return p


def _make_test_audio(sr=16000, dur=1.0):
    """Rich harmonic signal simulating a vowel at 200Hz."""
    t = np.arange(int(sr * dur)) / sr
    audio = np.zeros_like(t, dtype=np.float32)
    f0 = 200.0
    for h in range(1, 8):
        audio += (0.5 / h) * np.sin(2 * np.pi * f0 * h * t)
    return audio.astype(np.float32) * 0.3


# ======================================================================
# Test 1: noise_scale sweep — spectral quality
# ======================================================================

def test_noise_scale_spectral_sweep():
    pipeline = _load_pipeline()
    if pipeline is None:
        print("SKIP: model not found")
        return

    audio = _make_test_audio()
    model_sr = pipeline.sample_rate

    print("=" * 70)
    print("BATCH INFERENCE: noise_scale spectral sweep")
    print("=" * 70)

    results = {}
    for ns in [0.0, 0.2, 0.4, 0.66666]:
        out = pipeline.infer(
            audio, input_sr=16000, pitch_shift=0, f0_method="rmvpe",
            noise_scale=ns, f0_lowpass_cutoff_hz=16.0,
        )
        r = spectral_analysis(out, model_sr, f"ns={ns:.3f}")
        results[ns] = r

    # Summary comparison
    print("\n--- Summary ---")
    print(f"  {'ns':>8} | {'HNR':>7} | {'Tilt':>7} | {'hi-band':>8} | {'vhi-band':>9}")
    print(f"  {'-'*50}")
    for ns, r in results.items():
        print(
            f"  {ns:8.3f} | {r['hnr']:6.1f}dB | {r['tilt']:6.1f} | {r['e_hi']:7.1f}dB | {r['e_vhi']:8.1f}dB"
        )


# ======================================================================
# Test 2: Batch vs Streaming spectral comparison
# ======================================================================

def test_batch_vs_streaming_spectral():
    pipeline = _load_pipeline()
    if pipeline is None:
        print("SKIP: model not found")
        return

    audio = _make_test_audio()
    model_sr = pipeline.sample_rate

    print("\n" + "=" * 70)
    print("STREAMING vs BATCH comparison (ns=0.4)")
    print("=" * 70)

    # Batch
    out_batch = pipeline.infer(
        audio, input_sr=16000, pitch_shift=0, f0_method="rmvpe",
        noise_scale=0.4, f0_lowpass_cutoff_hz=16.0,
    )
    r_batch = spectral_analysis(out_batch, model_sr, "batch")

    # Streaming: simulate multiple chunks
    pipeline._streaming_prev_audio = None
    pipeline._streaming_prev_features = None
    pipeline._streaming_prev_f0 = None

    chunk_frames = 8   # 160ms
    overlap_frames = 5  # 100ms
    hop_samples = 320 * chunk_frames
    overlap_samples = 320 * overlap_frames
    total_samples = 320 * (chunk_frames + overlap_frames)

    streaming_out = []
    for i in range(5):
        start = i * hop_samples
        end = start + total_samples
        if end > len(audio):
            break
        chunk = audio[start:end]
        out_s = pipeline.infer_streaming(
            chunk, overlap_samples=overlap_samples,
            noise_scale=0.4, f0_lowpass_cutoff_hz=16.0,
        )
        streaming_out.append(out_s)

    if streaming_out:
        streaming_concat = np.concatenate(streaming_out)
        r_stream = spectral_analysis(streaming_concat, model_sr, "streaming")

        print(f"\n  --- Delta (streaming - batch) ---")
        print(f"  HNR:      {r_stream['hnr'] - r_batch['hnr']:+.1f}dB")
        print(f"  Tilt:     {r_stream['tilt'] - r_batch['tilt']:+.1f}dB/dec")
        print(f"  hi-band:  {r_stream['e_hi'] - r_batch['e_hi']:+.1f}dB")
        print(f"  vhi-band: {r_stream['e_vhi'] - r_batch['e_vhi']:+.1f}dB")


# ======================================================================
# Test 3: F0 tracking accuracy
# ======================================================================

def test_f0_tracking_accuracy():
    pipeline = _load_pipeline()
    if pipeline is None:
        print("SKIP: model not found")
        return

    print("\n" + "=" * 70)
    print("F0 TRACKING ACCURACY")
    print("=" * 70)

    audio = _make_test_audio()
    expected_f0 = 200.0

    audio_t = torch.from_numpy(audio).unsqueeze(0).to(pipeline.device)
    with torch.no_grad():
        with torch.autocast(device_type=pipeline.device, dtype=pipeline.dtype):
            f0_raw = pipeline.rmvpe.infer(audio_t)

    f0_np = f0_raw.cpu().float().numpy().squeeze()
    voiced = f0_np > 0

    if np.any(voiced):
        f0_voiced = f0_np[voiced]
        err = np.abs(f0_voiced - expected_f0)
        print(f"  Input: 200Hz harmonic signal")
        print(f"  Detected F0: mean={f0_voiced.mean():.1f}Hz  std={f0_voiced.std():.1f}Hz")
        print(f"  Range: [{f0_voiced.min():.1f}, {f0_voiced.max():.1f}]Hz")
        print(f"  Voiced: {np.sum(voiced)}/{len(f0_np)} ({np.mean(voiced)*100:.0f}%)")
        print(f"  Error: mean={err.mean():.1f}Hz  max={err.max():.1f}Hz")

        # F0 stability (frame-to-frame jitter)
        f0_diff = np.abs(np.diff(f0_voiced))
        print(f"  Frame jitter: mean={f0_diff.mean():.2f}Hz  max={f0_diff.max():.2f}Hz")
    else:
        print("  WARNING: No voiced frames detected!")


# ======================================================================
# Test 4: HuBERT feature quality — context dependence
# ======================================================================

def test_hubert_context_quality():
    pipeline = _load_pipeline()
    if pipeline is None:
        print("SKIP: model not found")
        return

    print("\n" + "=" * 70)
    print("HuBERT FEATURE QUALITY: context length dependence")
    print("=" * 70)

    audio = _make_test_audio()

    # Full audio
    audio_t_full = torch.from_numpy(audio).unsqueeze(0).float().to(pipeline.device)
    with torch.no_grad():
        with torch.autocast(device_type=pipeline.device, dtype=pipeline.dtype):
            feat_full = pipeline.hubert.infer(audio_t_full)
    feat_full_np = feat_full.cpu().float().numpy().squeeze()

    # Various context lengths
    contexts_ms = [160, 260, 500, 1000]
    for ctx_ms in contexts_ms:
        n_samples = int(16000 * ctx_ms / 1000)
        if n_samples > len(audio):
            continue
        chunk = audio[:n_samples]
        audio_t = torch.from_numpy(chunk).unsqueeze(0).float().to(pipeline.device)
        with torch.no_grad():
            with torch.autocast(device_type=pipeline.device, dtype=pipeline.dtype):
                feat_short = pipeline.hubert.infer(audio_t)
        feat_short_np = feat_short.cpu().float().numpy().squeeze()

        n_compare = min(feat_short_np.shape[0], feat_full_np.shape[0])
        if n_compare > 0:
            from numpy.linalg import norm
            cos_sims = []
            for i in range(n_compare):
                a = feat_full_np[i]
                b = feat_short_np[i]
                cos = np.dot(a, b) / (norm(a) * norm(b) + 1e-8)
                cos_sims.append(cos)
            cos_sims = np.array(cos_sims)
            print(
                f"  {ctx_ms:4d}ms ({feat_short_np.shape[0]:3d} frames): "
                f"cos_sim mean={cos_sims.mean():.4f}  min={cos_sims.min():.4f}  "
                f"norm_ratio={norm(feat_short_np, axis=1).mean() / (norm(feat_full_np[:n_compare], axis=1).mean() + 1e-8):.4f}"
            )


# ======================================================================
# Test 5: Output spectral comparison with input
# ======================================================================

def test_input_output_spectral_comparison():
    pipeline = _load_pipeline()
    if pipeline is None:
        print("SKIP: model not found")
        return

    print("\n" + "=" * 70)
    print("INPUT vs OUTPUT spectral comparison")
    print("=" * 70)

    audio_16k = _make_test_audio()
    model_sr = pipeline.sample_rate

    # Resample input to model_sr for comparison
    from scipy.signal import resample_poly
    audio_out_sr = resample_poly(audio_16k, model_sr, 16000).astype(np.float32)

    r_input = spectral_analysis(audio_out_sr, model_sr, "input (resampled)")

    out = pipeline.infer(
        audio_16k, input_sr=16000, pitch_shift=0, f0_method="rmvpe",
        noise_scale=0.4, f0_lowpass_cutoff_hz=16.0,
    )
    r_output = spectral_analysis(out, model_sr, "output (ns=0.4)")

    print(f"\n  --- Delta (output - input) ---")
    print(f"  HNR:      {r_output['hnr'] - r_input['hnr']:+.1f}dB")
    print(f"  Tilt:     {r_output['tilt'] - r_input['tilt']:+.1f}dB/dec")
    print(f"  hi-band:  {r_output['e_hi'] - r_input['e_hi']:+.1f}dB")
    print(f"  vhi-band: {r_output['e_vhi'] - r_input['e_vhi']:+.1f}dB")
    if r_output["hnr"] < r_input["hnr"] - 5:
        print(f"  ** HNR dropped by {r_input['hnr'] - r_output['hnr']:.1f}dB — significant noise added by synthesis **")
    if r_output["e_hi"] > r_input["e_hi"] + 5:
        print(f"  ** High-freq energy increased by {r_output['e_hi'] - r_input['e_hi']:.1f}dB — synthesis adds high-freq noise **")


# ======================================================================
# Runner
# ======================================================================

if __name__ == "__main__":
    tests = [
        test_noise_scale_spectral_sweep,
        test_batch_vs_streaming_spectral,
        test_f0_tracking_accuracy,
        test_hubert_context_quality,
        test_input_output_spectral_comparison,
    ]
    for t in tests:
        try:
            t()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ERROR: {e}")
