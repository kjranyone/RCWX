"""学習済み重み (folded fp16 バイナリ + meta.json) から RVC 互換 .pth を組む。

Rust 側 (rcwx-cli train-export --pth) が:
  - weights.f16   : 全テンソルの fp16 連結
  - meta.json     : [{"name","shape","offset"(要素),"dtype"}] + config + sr
を書き出す。本スクリプトはそれを torch.save 形式にパックする。

使い方: uv run python tools/export_pth.py --dir <export_dir> --out model.pth
"""
from __future__ import annotations

import argparse
import json
import struct
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = Path(args.dir)
    meta = json.loads((d / "meta.json").read_text())
    raw = np.frombuffer((d / "weights.f16").read_bytes(), dtype="<f2")
    sd = OrderedDict()
    for t in meta["tensors"]:
        n = int(np.prod(t["shape"])) if t["shape"] else 1
        arr = raw[t["offset"]: t["offset"] + n].astype(np.float32)
        sd[t["name"]] = torch.from_numpy(arr.reshape(t["shape"] or [1]))
    ckpt = {
        "weight": sd,
        "config": meta.get("config"),
        "info": meta.get("info", "rcwx-train"),
        "sr": meta.get("sr"),
        "f0": meta.get("f0", 1),
        "version": meta.get("version", "v2"),
    }
    torch.save(ckpt, args.out)
    print(f"saved {args.out}: {len(sd)} tensors")


if __name__ == "__main__":
    main()
