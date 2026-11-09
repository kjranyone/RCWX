"""RMVPE 重み変換 (dev 専用)。

- rmvpe.pt → rmvpe.safetensors (state_dict そのまま・BatchNorm は running stats 込み)
- mel_basis [128, 513] と hann 窓 [1024] も保存 (librosa htk=True 相当)
- golden: 決定論合成音声に対する f0 出力
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    from safetensors.torch import save_file

    from rcwx.models.rmvpe import RMVPE

    model_path = Path.home() / ".cache/rcwx/models/rmvpe/rmvpe.pt"
    out_dir = Path.home() / ".cache/rcwx/models/rust"
    out_dir.mkdir(parents=True, exist_ok=True)

    rm = RMVPE(str(model_path), device="cpu", dtype=torch.float32)
    rm.eval()

    t: dict[str, torch.Tensor] = {}
    for k, v in rm.model.state_dict().items():
        t[k] = v.detach().cpu().to(torch.float32).contiguous()
    t["mel_basis"] = rm.mel_extractor.mel_basis.cpu()
    t["hann_window"] = rm.mel_extractor.hann_window.cpu()
    t["cents_mapping"] = rm.cents_mapping.cpu()

    save_file(t, str(out_dir / "rmvpe.safetensors"))
    print(f"exported {out_dir / 'rmvpe.safetensors'} ({sum(v.numel() for v in t.values()) / 1e6:.2f}M params)")

    # golden
    rng = np.random.default_rng(42)
    n = 8960  # RMVPE 最小窓 (0.56s)
    f0_true = 180.0
    phase = np.cumsum(2 * np.pi * f0_true / 16000.0 + 0.0 * rng.standard_normal(n))
    sig = 0.3 * np.sin(phase) + 0.02 * rng.standard_normal(n)
    # 無声区間を混ぜる
    sig[n // 3 : n // 2] = 0.01 * rng.standard_normal(n // 2 - n // 3)
    audio = torch.from_numpy(sig.astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        f0 = rm.infer(audio, threshold=0.03)

    g = {"audio": audio, "f0": f0.cpu().float()}
    save_file(g, str(out_dir / "rmvpe_golden.safetensors"))
    print(f"golden: audio {tuple(audio.shape)} -> f0 {tuple(f0.shape)}")


if __name__ == "__main__":
    main()
