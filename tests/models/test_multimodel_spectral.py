"""Multi-model spectral analysis — is hoarseness model-specific or architectural?

Runs the same input through all available models and compares:
  - HNR (Harmonic-to-Noise Ratio)
  - Spectral tilt
  - High-frequency energy delta vs input
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

MODEL_DIR = Path(r"C:\lib\github\grand2-products\RCWX\model")

MODELS = [
    ("kurumi", MODEL_DIR / "kurumi" / "kurumi.pth"),
    ("kana", MODEL_DIR / "kana" / "kana" / "voice.pth"),
    ("tsukuyomi-1", MODEL_DIR / "つくよみちゃん公式RVCモデル" / "01 つくよみちゃん公式RVCモデル 通常1.pth"),
    ("tsukuyomi-2", MODEL_DIR / "つくよみちゃん公式RVCモデル" / "02 つくよみちゃん公式RVCモデル 通常2.pth"),
    ("tsukuyomi-3", MODEL_DIR / "つくよみちゃん公式RVCモデル" / "03 つくよみちゃん公式RVCモデル 通常3.pth"),
    ("tsukuyomi-strong", MODEL_DIR / "つくよみちゃん公式RVCモデル" / "04 つくよみちゃん公式RVCモデル 強.pth"),
    ("tsukuyomi-weak", MODEL_DIR / "つくよみちゃん公式RVCモデル" / "05 つくよみちゃん公式RVCモデル 弱.pth"),
]


def _make_test_audio(sr=16000, dur=1.0):
    t = np.arange(int(sr * dur)) / sr
    audio = np.zeros_like(t, dtype=np.float32)
    f0 = 200.0
    for h in range(1, 8):
        audio += (0.5 / h) * np.sin(2 * np.pi * f0 * h * t)
    return audio.astype(np.float32) * 0.3


def spectral_metrics(wav, sr):
    freqs, psd = welch(wav, fs=sr, nperseg=2048, noverlap=1024)
    psd_db = 10 * np.log10(psd + 1e-12)

    # Estimate f0
    voice_band = (freqs > 80) & (freqs < 500)
    f0_est = freqs[voice_band][np.argmax(psd[voice_band])] if np.any(voice_band) else 200.0

    # HNR
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

    # Tilt
    valid = (freqs > 100) & (freqs < 8000)
    tilt = np.polyfit(np.log10(freqs[valid]), psd_db[valid], 1)[0] if np.sum(valid) > 10 else float("nan")

    # Band energies
    def be(lo, hi):
        m = (freqs >= lo) & (freqs < hi)
        return 10 * np.log10(np.sum(psd[m]) + 1e-12)

    rms = np.sqrt(np.mean(wav ** 2))
    return {
        "hnr": hnr, "tilt": tilt, "rms": rms,
        "e_low": be(50, 500), "e_mid": be(500, 2000),
        "e_hi": be(2000, 8000), "e_vhi": be(8000, sr // 2),
    }


def main():
    from rcwx.pipeline.inference import RVCPipeline

    audio_16k = _make_test_audio()

    # Input reference at 48kHz
    audio_48k = resample_poly(audio_16k, 48000, 16000).astype(np.float32)
    ref = spectral_metrics(audio_48k, 48000)

    print("=" * 110)
    print("MULTI-MODEL SPECTRAL ANALYSIS")
    print("=" * 110)
    print(f"  Input reference: HNR={ref['hnr']:.1f}dB  Tilt={ref['tilt']:.1f}  "
          f"hi={ref['e_hi']:.1f}dB  vhi={ref['e_vhi']:.1f}dB")
    print()

    hdr = (f"  {'model':<18} | {'sr':>5} | {'RMS':>6} | {'HNR':>7} {'dHNR':>6} | "
           f"{'Tilt':>6} {'dTilt':>6} | {'hi(2-8k)':>9} {'dHi':>6} | {'vhi(8k+)':>9} {'dVhi':>6}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for name, path in MODELS:
        if not path.exists():
            print(f"  {name:<18} | SKIP (not found)")
            continue

        try:
            pipeline = RVCPipeline(str(path), device="xpu", dtype=torch.float16, use_compile=False)
            pipeline.load()
            model_sr = pipeline.sample_rate

            out = pipeline.infer(
                audio_16k, input_sr=16000, pitch_shift=0, f0_method="rmvpe",
                noise_scale=0.4, f0_lowpass_cutoff_hz=16.0,
            )

            m = spectral_metrics(out, model_sr)

            # Compute reference at the model's sample rate for fair comparison
            if model_sr != 48000:
                audio_ref = resample_poly(audio_16k, model_sr, 16000).astype(np.float32)
                ref_m = spectral_metrics(audio_ref, model_sr)
            else:
                ref_m = ref

            d_hnr = m["hnr"] - ref_m["hnr"]
            d_tilt = m["tilt"] - ref_m["tilt"]
            d_hi = m["e_hi"] - ref_m["e_hi"]
            d_vhi = m["e_vhi"] - ref_m["e_vhi"]

            print(
                f"  {name:<18} | {model_sr:>5} | {m['rms']:.4f} | "
                f"{m['hnr']:6.1f}dB {d_hnr:+5.1f} | "
                f"{m['tilt']:5.1f} {d_tilt:+5.1f} | "
                f"{m['e_hi']:8.1f}dB {d_hi:+5.1f} | "
                f"{m['e_vhi']:8.1f}dB {d_vhi:+5.1f}"
            )

            # Cleanup
            del pipeline
            torch.xpu.empty_cache() if hasattr(torch, "xpu") and torch.xpu.is_available() else None

        except Exception as e:
            print(f"  {name:<18} | ERROR: {e}")

    print()
    print("  dHNR: delta HNR vs input (positive=cleaner, negative=noisier)")
    print("  dHi/dVhi: delta high-freq energy vs input (large positive=synthesis adds noise)")


if __name__ == "__main__":
    main()
