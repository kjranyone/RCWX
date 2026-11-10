"""M-T3 検証: 前処理 (lfilter / STFT / slicer) の scipy / torch golden 生成。

- hpf:  scipy.signal.butter(5, 48/(sr/2), 'high') + lfilter — Rust butter5_hpf と maxdiff 1e-5
- spec: torch.stft(n_fft 2048, hop 480, hann, center=False) の
        sqrt(re²+im²+2e-7) magnitude — Rust spectrogram と rtol 1e-4
- slicer: Rust slice_and_chunk (per 3.0s / overlap 0.3s) のアルゴリズムを
        py 側に鏡像実装して境界 (start, len) を保存 — 完全一致

使い方: uv run python tools/dump_preproc_golden.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SR = 48000


def slice_and_chunk(x_len: int, sr: int, per: float, overlap: float):
    """rcwx-train data/slicer.rs の鏡像。境界 (start, length) の list を返す。"""
    per_n = int(per * sr)
    ovl_n = int(overlap * sr)
    out: list[tuple[int, int]] = []
    if x_len <= per_n:
        if x_len > sr:
            out.append((0, x_len))
        return out
    step = per_n - min(ovl_n, per_n - 1)
    i = 0
    while i + per_n <= x_len:
        out.append((i, per_n))
        i += step
    if x_len - i > per_n // 2:
        out.append((x_len - per_n, per_n))
    return out


def main() -> None:
    import torch
    from scipy.signal import butter, lfilter

    rng = np.random.RandomState(1234)
    # per 3.0s (144000) / step 2.7s (129600) で複数チャンク + 末尾チャンク発火
    n = 144_000 + 129_600 + 91_000
    t = np.arange(n, dtype=np.float64)
    x = (
        0.4 * np.sin(2 * np.pi * 220 * t / SR)
        + 0.15 * np.sin(2 * np.pi * 613 * t / SR + 0.7)
        + 0.05 * rng.randn(n)
    ).astype(np.float32)

    # ---- lfilter golden (scipy butter5 48Hz HPF) ----
    b, a = butter(5, 48.0 / (SR / 2), btype="high")
    hpf = lfilter(b, a, x).astype(np.float32)

    # ---- STFT golden (torch.stft, center=False, hann 2048 / hop 480) ----
    # rust 側 hann は 0.5-0.5cos(2πk/N) (= torch periodic 既定)。f32 で計算すると
    # FFT 累積誤差 ~5e-4 で 1e-4 ゲートを超えるため **f64 で計算**して f32 保存。
    n_fft, hop = 2048, 480
    window = torch.hann_window(n_fft, dtype=torch.float64)
    stft = torch.stft(
        torch.from_numpy(x.astype(np.float64)),
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=False,
        return_complex=True,
    )
    spec = torch.sqrt(stft.real**2 + stft.imag**2 + 2e-7).float().numpy()
    # rust は [frames, bins] 平坦
    spec_flat = spec.T.reshape(-1).astype(np.float32)

    # ---- slicer golden ----
    chunks = slice_and_chunk(n, SR, per=3.0, overlap=0.3)
    sl = np.array([[s, l] for s, l in chunks], dtype=np.float32).reshape(-1)

    out_dir = os.path.join(os.path.expanduser("~"), ".cache", "rcwx", "train", "golden")
    os.makedirs(out_dir, exist_ok=True)
    from safetensors.numpy import save_file

    save_file(
        {
            "x": x,
            "hpf": hpf,
            "spec": spec_flat,
            "slice": sl,
        },
        os.path.join(out_dir, "preproc_golden.safetensors"),
    )
    print(
        f"x={n} hpf={len(hpf)} spec={spec.shape} (frames={spec.shape[1]}) "
        f"chunks={len(chunks)} → preproc_golden.safetensors"
    )


if __name__ == "__main__":
    main()
