"""Test: Does float16 vs float32 explain the high-frequency noise?

Compares spectral characteristics of synthesis output at different precisions.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import welch, resample_poly

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.WARNING)

MODEL_PATH = Path(r"C:\lib\github\grand2-products\RCWX\model\kurumi\kurumi.pth")


def _make_test_audio(sr=16000, dur=1.0):
    t = np.arange(int(sr * dur)) / sr
    audio = np.zeros_like(t, dtype=np.float32)
    for h in range(1, 8):
        audio += (0.5 / h) * np.sin(2 * np.pi * 200 * h * t)
    return audio.astype(np.float32) * 0.3


def spectral_metrics(wav, sr):
    freqs, psd = welch(wav, fs=sr, nperseg=2048, noverlap=1024)
    psd_db = 10 * np.log10(psd + 1e-12)

    voice_band = (freqs > 80) & (freqs < 500)
    f0_est = freqs[voice_band][np.argmax(psd[voice_band])] if np.any(voice_band) else 200.0

    harm_p, noise_p = 0.0, 0.0
    for i, f in enumerate(freqs):
        if f < 50 or f > 8000:
            continue
        nearest = round(f / f0_est) * f0_est
        if abs(f - nearest) < 20:
            harm_p += psd[i]
        else:
            noise_p += psd[i]
    hnr = 10 * np.log10(harm_p / (noise_p + 1e-12))

    valid = (freqs > 100) & (freqs < 8000)
    tilt = np.polyfit(np.log10(freqs[valid]), psd_db[valid], 1)[0] if np.sum(valid) > 10 else float("nan")

    def be(lo, hi):
        m = (freqs >= lo) & (freqs < hi)
        return 10 * np.log10(np.sum(psd[m]) + 1e-12)

    return {
        "hnr": hnr, "tilt": tilt, "rms": np.sqrt(np.mean(wav ** 2)),
        "e_hi": be(2000, 8000), "e_vhi": be(8000, sr // 2),
    }


def main():
    if not MODEL_PATH.exists():
        print("SKIP: model not found")
        return

    from rcwx.pipeline.inference import RVCPipeline

    audio = _make_test_audio()

    # Input reference
    audio_48k = resample_poly(audio, 48000, 16000).astype(np.float32)
    ref = spectral_metrics(audio_48k, 48000)

    print("=" * 100)
    print("DTYPE COMPARISON: float16 vs float32 vs bfloat16")
    print("=" * 100)
    print(f"  Input ref: HNR={ref['hnr']:.1f}dB  hi={ref['e_hi']:.1f}dB  vhi={ref['e_vhi']:.1f}dB")
    print()

    configs = [
        ("float16 (current)", torch.float16),
        ("float32", torch.float32),
    ]

    # Check bfloat16 support
    try:
        if torch.xpu.is_available() and torch.xpu.is_bf16_supported():
            configs.append(("bfloat16", torch.bfloat16))
    except Exception:
        pass

    hdr = f"  {'config':<20} | {'RMS':>6} | {'HNR':>7} {'dHNR':>6} | {'Tilt':>6} | {'hi(2-8k)':>9} {'dHi':>6} | {'vhi(8k+)':>9} {'dVhi':>6}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for label, dtype in configs:
        try:
            pipeline = RVCPipeline(str(MODEL_PATH), device="xpu", dtype=dtype, use_compile=False)
            pipeline.load()

            out = pipeline.infer(
                audio, input_sr=16000, pitch_shift=0, f0_method="rmvpe",
                noise_scale=0.4, f0_lowpass_cutoff_hz=16.0,
            )

            model_sr = pipeline.sample_rate
            m = spectral_metrics(out, model_sr)

            # Reference at model SR
            if model_sr != 48000:
                audio_ref_sr = resample_poly(audio, model_sr, 16000).astype(np.float32)
                ref_m = spectral_metrics(audio_ref_sr, model_sr)
            else:
                ref_m = ref

            d_hnr = m["hnr"] - ref_m["hnr"]
            d_hi = m["e_hi"] - ref_m["e_hi"]
            d_vhi = m["e_vhi"] - ref_m["e_vhi"]

            print(
                f"  {label:<20} | {m['rms']:.4f} | "
                f"{m['hnr']:6.1f}dB {d_hnr:+5.1f} | "
                f"{m['tilt']:5.1f} | "
                f"{m['e_hi']:8.1f}dB {d_hi:+5.1f} | "
                f"{m['e_vhi']:8.1f}dB {d_vhi:+5.1f}"
            )

            del pipeline
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.empty_cache()

        except Exception as e:
            print(f"  {label:<20} | ERROR: {e}")

    # === Test 2: float16 HuBERT + float32 synthesis ===
    print()
    print("=" * 100)
    print("MIXED PRECISION: float16 HuBERT + float32 synthesis")
    print("=" * 100)

    try:
        # Load as float32
        pipeline = RVCPipeline(str(MODEL_PATH), device="xpu", dtype=torch.float32, use_compile=False)
        pipeline.load()

        # But extract HuBERT features in float16 (like original pipeline)
        audio_t = torch.from_numpy(audio).unsqueeze(0).to("xpu")
        with torch.no_grad():
            # float16 HuBERT
            with torch.autocast(device_type="xpu", dtype=torch.float16):
                features_f16 = pipeline.hubert.extract(audio_t, output_layer=12, output_dim=768)

            # float32 HuBERT
            features_f32 = pipeline.hubert.extract(audio_t.float(), output_layer=12, output_dim=768)

        # Compare features
        f16_np = features_f16.float().cpu().numpy().squeeze()
        f32_np = features_f32.float().cpu().numpy().squeeze()

        n = min(f16_np.shape[0], f32_np.shape[0])
        diff = np.abs(f16_np[:n] - f32_np[:n])

        from numpy.linalg import norm
        cos_sims = []
        for i in range(n):
            cos = np.dot(f16_np[i], f32_np[i]) / (norm(f16_np[i]) * norm(f32_np[i]) + 1e-8)
            cos_sims.append(cos)

        print(f"  HuBERT feature comparison (f16 vs f32):")
        print(f"    Cosine similarity: mean={np.mean(cos_sims):.6f}  min={np.min(cos_sims):.6f}")
        print(f"    Abs diff: mean={diff.mean():.6f}  max={diff.max():.6f}")
        print(f"    f16 norm: {norm(f16_np, axis=1).mean():.4f}  f32 norm: {norm(f32_np, axis=1).mean():.4f}")

        del pipeline
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
