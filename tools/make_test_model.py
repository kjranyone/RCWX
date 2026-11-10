"""ランダム重みの RVC v1/v2 テストモデル生成 (dev 専用)。

実モデルが手元にない環境で v1 (256-dim 特徴・n_layers=3・kernel 5) と
v2 (768-dim・n_layers=6・kernel 3) のパリティ検証を行うための決定論的
チェックポイント (RVC 形式: weight/config/version/sr/f0) を書き出す。
重みは乱数なので音質は無意味だが、Rust/Python 間の構造パリティ検証には十分。

使い方:
    uv run python tools/make_test_model.py [--out-dir target/testmodels] [--seed 20241110]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rcwx.models.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs768NSFsid,
)


def rvc_config(n_layers: int, kernel_size: int) -> list:
    """RVC 40k config (SynthesizerLoader._parse_config と同じ 18 項目順)。"""
    return [
        1025,  # spec_channels
        1024,  # segment_size
        192,  # inter_channels
        192,  # hidden_channels
        768,  # filter_channels
        2,  # n_heads
        n_layers,
        kernel_size,
        0.1,  # p_dropout
        "1",  # resblock
        [3, 7, 11],
        [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        [10, 10, 2, 2],
        512,  # upsample_initial_channel
        [16, 16, 4, 4],
        256,  # spk_embed_dim
        256,  # gin_channels
        40000,  # sr
    ]


def make(version: int, seed: int, out: Path) -> None:
    torch.manual_seed(seed)
    if version == 1:
        cfg = rvc_config(n_layers=3, kernel_size=5)
        model = SynthesizerTrnMs256NSFsid(*cfg)
    else:
        cfg = rvc_config(n_layers=6, kernel_size=3)
        model = SynthesizerTrnMs768NSFsid(*cfg)
    model.eval()
    ckpt = {
        "weight": model.state_dict(),
        "config": cfg,
        "version": f"v{version}",
        "sr": 40000,
        "f0": 1,
        "info": f"RCWX random test model v{version} (seed={seed})",
    }
    torch.save(ckpt, out)
    n_params = sum(v.numel() for v in model.state_dict().values())
    print(f"v{version}: {out} ({n_params / 1e6:.1f}M params, n_layers={cfg[6]}, kernel={cfg[7]})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="target/testmodels", help="出力ディレクトリ")
    ap.add_argument("--seed", type=int, default=20241110, help="乱数シード")
    ap.add_argument("--versions", default="1,2", help="生成するバージョン (カンマ区切り)")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for v in [int(s) for s in args.versions.split(",") if s.strip()]:
        make(v, args.seed + v, out_dir / f"rand_v{v}.pth")


if __name__ == "__main__":
    main()
