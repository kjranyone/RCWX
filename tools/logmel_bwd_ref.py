"""logmel 逆伝達の py 参照: rust analytic dy との直接比較用 golden。

Rust: crates/rcwx-train/tests/mpd_gradcheck.rs logmel_backward_fd と同一の
ya/yb (決定的生成) で mean|logmel(a) − logmel(b)| の a 勾配を保存。

使い方: uv run python tools/logmel_bwd_ref.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def logmel_torch(y: torch.Tensor) -> torch.Tensor:
    """StftMel::logmel と同一式 (reflect pad → hann → |STFT|+2e-7 → slaney mel → log clamp 2e-6)。"""
    import torch.nn.functional as F

    pad = (N_FFT - HOP) // 2
    yy = F.pad(y.unsqueeze(0), (pad, pad), mode="reflect").squeeze(0)
    window = torch.hann_window(N_FFT)
    spec = torch.stft(
        yy, N_FFT, hop_length=HOP, win_length=N_FFT, window=window,
        center=False, return_complex=True,
    )
    mag = torch.sqrt(spec.real**2 + spec.imag**2 + 2e-7)
    # slaney mel basis は dump_mel_golden と同一生成
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from librosa_mel import slaney_mel

    basis = torch.from_numpy(slaney_mel(N_MELS, N_FFT // 2 + 1, SR, N_FFT)).float()
    mel = torch.matmul(basis, mag)
    return torch.log(torch.clamp(mel, min=2e-6))


def main() -> None:
    ya = torch.from_numpy(mk(True)).requires_grad_(True)
    yb = torch.from_numpy(mk(False))
    la = logmel_torch(ya)
    lb = logmel_torch(yb)
    loss = torch.mean(torch.abs(la - lb))
    loss.backward()

    out_dir = os.path.join(os.path.expanduser("~"), ".cache", "rcwx", "train", "golden")
    os.makedirs(out_dir, exist_ok=True)
    from safetensors.numpy import save_file

    save_file(
        {
            "ya": ya.detach().numpy(),
            "yb": yb.numpy(),
            "dy": ya.grad.numpy(),
        },
        os.path.join(out_dir, "logmel_bwd.safetensors"),
    )
    print(f"loss={loss.item():.6f} |dy|max={ya.grad.abs().max().item():.6f} → logmel_bwd.safetensors")


if __name__ == "__main__":
    main()
