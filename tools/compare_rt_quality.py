"""rt --virtual 出力とオフライン run 出力の品質比較 (dev 専用)。

fc68405 (Python 時代の チャンク/バッチ品質差 検証) の Rust 版。
同一入力・同一モデルで 2 経路を回し、時間整列後に:

- 波形 cosine / 相関
- エンベロープ相関 (RMS 20ms 窓 — fc68405 と同じ発想)
- セグメント別 (0.5s) 相関の最小値 (局所劣化の検出)

使い方:
    uv run python tools/compare_rt_quality.py <offline.wav> <rt.wav> [--sr 48000]
"""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    w = wave.open(str(path), "r")
    x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64) / 32768.0
    if w.getnchannels() > 1:
        x = x.reshape(-1, w.getnchannels()).mean(axis=1)
    return x, w.getframerate()


def align(a: np.ndarray, b: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, int]:
    """相互相関で最適ラグを探して共通区間を返す (±0.5s)。"""
    n = min(len(a), len(b))
    max_lag = int(sr * 0.5)
    a2, b2 = a[:n], b[:n]
    fa = np.fft.rfft(a2, 2 * n)
    fb = np.fft.rfft(b2, 2 * n)
    cc = np.fft.irfft(fa.conj() * fb)
    lags = np.concatenate([np.arange(0, max_lag), np.arange(-max_lag, 0)])
    vals = np.concatenate([cc[:max_lag], cc[-max_lag:]])
    lag = int(lags[np.argmax(vals)])
    if lag >= 0:
        a2, b2 = a2[lag:], b2
    else:
        a2, b2 = a2, b2[-lag:]
    m = min(len(a2), len(b2))
    return a2[:m], b2[:m], lag


def envelope(x: np.ndarray, sr: int, win_ms: float = 20.0) -> np.ndarray:
    k = max(1, int(sr * win_ms / 1000))
    pad = len(x) % k
    if pad:
        x = x[: len(x) - pad]
    rms = np.sqrt((x.reshape(-1, k) ** 2).mean(axis=1))
    return rms


def corr(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(a @ b / (na * nb))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("offline")
    ap.add_argument("rt")
    args = ap.parse_args()
    a, sr_a = load_wav(Path(args.offline))
    b, sr_b = load_wav(Path(args.rt))
    if sr_a != sr_b:
        print(f"sample rate mismatch: {sr_a} vs {sr_b}", file=sys.stderr)
        sys.exit(1)
    a, b, lag = align(a, b, sr_a)
    print(f"整列: lag={lag} samples ({lag / sr_a * 1000:.1f}ms)  長さ={len(a) / sr_a:.2f}s")
    print(f"波形 cosine      : {corr(a, b):.4f}")
    ea, eb = envelope(a, sr_a), envelope(b, sr_a)
    print(f"エンベロープ相関 : {corr(ea, eb):.4f}")
    seg = int(sr_a * 0.5)
    nseg = len(a) // seg
    worst, worst_i = 1.0, -1
    segs = []
    for i in range(nseg):
        sa, sb = a[i * seg : (i + 1) * seg], b[i * seg : (i + 1) * seg]
        c = corr(sa, sb)
        segs.append(c)
        if c < worst:
            worst, worst_i = c, i
    print("セグメント (0.5s) 波形相関:", " ".join(f"{c:.3f}" for c in segs))
    print(f"最小セグメント   : {worst:.4f} @ {worst_i * 0.5:.1f}s")


if __name__ == "__main__":
    main()
