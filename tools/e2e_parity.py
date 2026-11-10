"""M-T5 E2E ゲート: 学習済み .pth を **製品経路そのまま** (pipeline.infer,
fp32, monkeypatch なし) で推論し、Rust `rcwx-cli run` 出力と波形 corr を比較。

重要 (2026-11 調査結果):
- py CLI は既定 dtype=float16。fp16 のまま比較すると py(fp16) vs py(fp32)
  自体が corr 0.988 しか出ない (fp16 ノイズフロア)。**ゲートは fp32 同士で**
  比較すること (本スクリプトは float32 固定)。
- 残る差の主因は RMVPE 乱数でなく (a) 末尾 f0 の stabilize_f0_boundaries
  クリフ (最終有声フレームが n-1-edge 境界を跨ぐと片側だけ平滑化が発火)、
  (b) 1 sample 未満の先頭トリムずれ。実測 corr ≈ 0.9956。

使い方:
  # Rust 側 (native bundle = train-export の model/):
  RCWX_MODELS_DIR=.../e2e_test/model rcwx-cli run in.wav -o out_rust.wav \
      --noise 0.0 --f0-method rmvpe --index-rate 0.0
  # Python 側 (本スクリプト):
  uv run python tools/e2e_parity.py --model model.pth --wav in.wav \
      --rust out_rust.wav [--gate 0.98]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--rust", required=True, help="Rust 出力 wav")
    ap.add_argument("--gate", type=float, default=0.98)
    ap.add_argument("--device", default="xpu")
    args = ap.parse_args()

    import torch
    from scipy.io import wavfile

    from rcwx.pipeline import inference as inf

    sr, xi = wavfile.read(args.wav)
    x = xi.astype(np.float32) / 32768.0
    if x.ndim == 2:
        x = x.mean(axis=1)

    pipeline = inf.RVCPipeline(
        args.model, device=args.device, dtype=torch.float32, use_f0=True,
        use_compile=False,
    )
    pipeline.load()
    with torch.no_grad():
        out = pipeline.infer(
            x,
            input_sr=sr,
            pitch_shift=0,
            f0_method="rmvpe",
            index_rate=0.0,
            noise_scale=0.0,
            moe_boost=0.0,
            pad_mode="batch",
            use_feature_cache=False,
        )

    sr_r, rust = wavfile.read(args.rust)
    assert sr_r == pipeline.sample_rate, f"sr mismatch: {sr_r} vs {pipeline.sample_rate}"
    a = out.astype(np.float64)
    r = rust.astype(np.float64) / 32768.0
    n = min(len(a), len(r))
    a, r = a[:n], r[:n]

    def corr(x: np.ndarray, y: np.ndarray) -> float:
        d = np.linalg.norm(x) * np.linalg.norm(y)
        return float(np.dot(x, y) / d) if d > 0 else float("nan")

    c0 = corr(a, r)
    # ±2 sample のオフセット掃引 (先頭トリム丸め差を許容する参考値)
    lags = range(-2, 3)
    cl = max(lags, key=lambda l: corr(a[l:] if l >= 0 else a[:l],
                                      r[:len(r) - l] if l > 0 else (r if l == 0 else r[-l:])))
    cb = corr(a[cl:] if cl >= 0 else a[:cl],
              r[:len(r) - cl] if cl > 0 else (r if cl == 0 else r[-cl:]))
    print(f"E2E corr (product path, fp32) = {c0:.5f}  [best lag {cl:+d}: {cb:.5f}]"
          f"  (gate >= {args.gate})")
    if c0 >= args.gate:
        print("[OK] E2E パリティ合格")
    else:
        print("[NG] E2E パリティ不合格")
        sys.exit(1)


if __name__ == "__main__":
    main()
