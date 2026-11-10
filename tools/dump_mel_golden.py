"""M-T4 検証: 学習 mel 損失 (logmel) の PyTorch golden 生成。

RVC mel_processing.mel_spectrogram_torch と同一パラメータ (n_fft 2048 /
hop 480 / 128 mels / fmin 0 / fmax None / reflect pad (n_fft-hop)/2 / log clamp)
で合成信号の log-mel を保存し、Rust 側 stft_mel と突き合わせる。

使い方: uv run python tools/dump_mel_golden.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def mel_spectrogram_torch(
    y: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: int,
    fmax,
    center: bool = False,
) -> torch.Tensor:
    """RVC WebUI mel_processing と同一実装。"""
    import torch.nn.functional as F

    if torch.min(y) < -1.0:
        print("min", torch.min(y))
    if torch.max(y) > 1.0:
        print("max", torch.max(y))
    global mel_basis, hann_window
    mel_basis = {}
    hann_window = {}
    dtype_device = str(y.dtype) + "_" + str(y.device)
    if dtype_device not in mel_basis:
        try:
            import librosa

            mel = librosa.filters.mel(
                sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
            )
        except ModuleNotFoundError:
            from librosa_mel import slaney_mel as _slaney

            mel = _slaney(num_mels, n_fft // 2 + 1, sampling_rate, n_fft, fmin, fmax)
        mel_basis[dtype_device] = torch.from_numpy(mel).float().to(y.dtype).to(y.device)
        hann_window[dtype_device] = torch.hann_window(win_size).to(y.dtype).to(y.device)
    y = F.pad(
        y.unsqueeze(0).unsqueeze(0),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    ).squeeze(0)
    y = y.squeeze(1)
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    mel = torch.matmul(mel_basis[dtype_device], spec)
    log_spec = torch.log(torch.clamp(mel, min=2e-6))
    return log_spec


def main() -> None:
    g = torch.Generator().manual_seed(2024)
    y = (torch.randn(17280, generator=g) * 0.3).float()
    # 簡易有声明: 正弦波 + ノイズ
    t = torch.arange(17280).float()
    y = 0.4 * torch.sin(2 * np.pi * 220 * t / 48000) + 0.1 * y
    mel = mel_spectrogram_torch(
        y, 2048, 128, 48000, 480, 2048, 0, None, center=False
    )
    out_dir = os.path.join(
        os.path.expanduser("~"), ".cache", "rcwx", "train", "golden"
    )
    os.makedirs(out_dir, exist_ok=True)
    from safetensors.torch import save_file

    save_file(
        {"y": y.detach(), "mel": mel.detach()},
        os.path.join(out_dir, "mel_golden.safetensors"),
    )
    print(f"mel shape: {tuple(mel.shape)} → mel_golden.safetensors")


if __name__ == "__main__":
    main()
