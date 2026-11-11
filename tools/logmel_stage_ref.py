"""logmel 構成段階別の逆伝達参照 (prefix bisect 用)。

P1: reflect → frames → window → (l1 vs 同 prefix の b)
P2: P1 + DFT mm → mag
P3: P2 + mel mm
P4: P3 + log  (= logmel)

各段階の loss は mean|prefix(a) − prefix(b)| で a の勾配を保存。
使い方: uv run python tools/logmel_stage_ref.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BATCH, SAMPLES = 2, 18400
N_FFT, HOP, N_MELS, SR = 2048, 480, 128, 48000


def mk(seed_a: bool) -> np.ndarray:
    n = BATCH * SAMPLES
    i = np.arange(n)
    if seed_a:
        v = (i * 0.37 + (i % 7) * 0.11) % 1.0 - 0.5
    else:
        v = (i * 0.71 + (i % 5) * 0.13) % 1.0 - 0.5
    return v.astype(np.float32)


def stages(y: torch.Tensor):
    """段階辞書を返す (各値は tensor)。"""
    pad = (N_FFT - HOP) // 2
    y2 = y.view(BATCH, SAMPLES)
    yy = F.pad(y2, (pad, pad), mode="reflect").reshape(-1)
    out = {"pad": yy}
    # frames: unfold
    # per-batch frames (rust と同一構造): [B, n_frames, N_FFT]
    fr = yy.view(BATCH, -1).unfold(1, N_FFT, HOP)
    out["frames"] = fr
    win = torch.hann_window(N_FFT)
    fw = fr * win  # [B, n_frames, N_FFT]
    out["windowed"] = fw
    # rust と同一: f64 で cos まで計算して f32 化 (MKL twiddle 相当)
    ang = torch.cos(
        -torch.arange(N_FFT).double()[:, None]
        * torch.arange(N_FFT // 2 + 1).double()[None, :]
        * (2 * torch.pi / N_FFT)
    ).float()
    spec_r = fw @ ang  # cos 基底 (上で f64 計算→f32)
    spec_i = fw @ torch.sin(
        -torch.arange(N_FFT).double()[:, None]
        * torch.arange(N_FFT // 2 + 1).double()[None, :]
        * (2 * torch.pi / N_FFT)
    ).float()
    out["re"] = spec_r
    mag = torch.sqrt(spec_r**2 + spec_i**2 + 2e-7)  # [B, n_frames, bins]
    out["mag"] = mag
    from librosa_mel import slaney_mel

    basis = torch.from_numpy(slaney_mel(N_MELS, N_FFT // 2 + 1, SR, N_FFT)).float()
    mel = mag @ basis.T  # [B, n_frames, mels]
    out["mel"] = mel
    lm = torch.log(torch.clamp(mel, min=2e-6))
    out["log"] = lm
    return out


def main() -> None:
    from safetensors.numpy import save_file

    data = {}
    for tag in ["a", "b"]:
        y = torch.from_numpy(mk(tag == "a"))
        if tag == "a":
            y = y.requires_grad_(True)
        st = stages(y)
        if tag == "a":
            for name, val in st.items():
                if val.requires_grad:
                    # この段階までの l1 損失 (vs b の同段階) の勾配
                    pass
            # b 側を再計算して l1
            yb_t = torch.from_numpy(mk(False))
            stb = stages(yb_t)
            for name in ["pad", "frames", "windowed", "re", "mag", "mel", "log"]:
                la = st[name].reshape(-1)
                lb = stb[name].reshape(-1).detach()  # pad は平坦・他は [B, ...] 平坦化
                data[f"fwd_{name}"] = la.detach().float().numpy()
                loss = torch.mean(torch.abs(la - lb))
                gy = torch.autograd.grad(loss, y, retain_graph=True)[0]
                data[f"dy_{name}"] = gy.float().numpy()
            # re 損失の windowed/frames レベル勾配 (mm bwd 分離)
            la_r = st["re"].reshape(-1)
            lb_r = stb["re"].reshape(-1).detach()
            loss_r = torch.mean(torch.abs(la_r - lb_r))
            for name in ["windowed", "frames", "pad"]:
                gn = torch.autograd.grad(loss_r, st[name], retain_graph=True)[0]
                data[f"dyR_at_{name}"] = gn.reshape(-1).float().numpy()
            data["ya"] = y.detach().numpy()
            data["yb"] = yb_t.numpy()
    out_dir = os.path.join(os.path.expanduser("~"), ".cache", "rcwx", "train", "golden")
    os.makedirs(out_dir, exist_ok=True)
    save_file(data, os.path.join(out_dir, "logmel_stages.safetensors"))
    print("keys:", sorted(data.keys()))


if __name__ == "__main__":
    main()
