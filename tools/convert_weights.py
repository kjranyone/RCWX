"""RCWX 重み変換ツール (dev 専用・実行時には Python 不要)。

PyTorch 側 (RCWX 既存ローダ) でモデルをロードし、Rust (burn) 側の名前規約で
safetensors へ書き出す。数値正規化はここで完了させる:
  - pos_conv の weight_norm (dim=2) を解決済み重みに畳み込む
  - attention の q/k/v を 1 本の fused QKV Linear に連結する
  - torch レイアウトを維持 (Linear は [out, in]。Rust 側ローダで転置する)

同時に golden 入出力 (決定論的合成音声 × 2 長) を fp32 でダンプし、
Rust 側パリティテストの基準にする。

使い方:
    uv run python tools/convert_weights.py hubert [--model <hubert_base.pt>] [--out ~/.cache/rcwx/models/rust]
    uv run python tools/convert_weights.py synth   <model.pth> [--out ...]   # M2 で実装
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def convert_hubert(model_path: Path, out_dir: Path) -> None:
    from safetensors.torch import save_file

    from rcwx.models.hubert_loader import HuBERTLoader

    out_dir.mkdir(parents=True, exist_ok=True)
    loader = HuBERTLoader(str(model_path), device="cpu", dtype=torch.float32)
    m = loader.model
    m.eval()

    t: dict[str, torch.Tensor] = {}

    def put(name: str, v: torch.Tensor | None) -> None:
        if v is None:
            return  # fairseq の conv は bias 無し
        t[name] = v.detach().cpu().to(torch.float32).contiguous()

    # conv frontend: 7段 (padding なし, k/s = (10,3,3,3,3,2,2)/(5,2,2,2,2,2,2))
    # GN は第 1 段のみ (transformers group mode)
    for i, layer in enumerate(m.feature_extractor.conv_layers):
        put(f"conv.{i}.weight", layer.conv.weight)
        put(f"conv.{i}.bias", layer.conv.bias)
    gn0 = m.feature_extractor.conv_layers[0].layer_norm
    put("gn0.weight", gn0.weight)
    put("gn0.bias", gn0.bias)

    fp_ln = m.feature_projection.layer_norm
    put("fp_ln.weight", fp_ln.weight)
    put("fp_ln.bias", fp_ln.bias)
    proj = m.feature_projection.projection
    put("fp_proj.weight", proj.weight)
    put("fp_proj.bias", proj.bias)

    # pos_conv: parametrization 解決後 (weight_norm dim=2 畳込済み) の重み
    pos = m.encoder.pos_conv_embed.conv
    put("pos_conv.weight", pos.weight)
    put("pos_conv.bias", pos.bias)

    enc_ln = m.encoder.layer_norm
    put("enc_ln.weight", enc_ln.weight)
    put("enc_ln.bias", enc_ln.bias)

    for i, layer in enumerate(m.encoder.layers):
        a = layer.attention
        put(
            f"layers.{i}.qkv.weight",
            torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight], dim=0),
        )
        put(
            f"layers.{i}.qkv.bias",
            torch.cat([a.q_proj.bias, a.k_proj.bias, a.v_proj.bias], dim=0),
        )
        put(f"layers.{i}.o.weight", a.out_proj.weight)
        put(f"layers.{i}.o.bias", a.out_proj.bias)
        put(f"layers.{i}.ln_a.weight", layer.layer_norm.weight)
        put(f"layers.{i}.ln_a.bias", layer.layer_norm.bias)
        put(f"layers.{i}.ff1.weight", layer.feed_forward.intermediate_dense.weight)
        put(f"layers.{i}.ff1.bias", layer.feed_forward.intermediate_dense.bias)
        put(f"layers.{i}.ff2.weight", layer.feed_forward.output_dense.weight)
        put(f"layers.{i}.ff2.bias", layer.feed_forward.output_dense.bias)
        put(f"layers.{i}.ln_f.weight", layer.final_layer_norm.weight)
        put(f"layers.{i}.ln_f.bias", layer.final_layer_norm.bias)

    put("final_proj.weight", loader.final_proj.weight)
    put("final_proj.bias", loader.final_proj.bias)

    weights_path = out_dir / "hubert.safetensors"
    save_file(t, str(weights_path))
    n_params = sum(v.numel() for v in t.values())
    print(f"exported {weights_path} ({n_params / 1e6:.1f}M params, {len(t)} tensors)")

    # ---- golden: 決定論的合成入力 × (v2 layer12/768, v1 layer9/256) ----
    goldens: dict[str, torch.Tensor] = {}
    for name, n_samples in (("aggr", 10880), ("norm", 17920)):
        rng = np.random.default_rng(42)
        wav = (rng.standard_normal(n_samples) * 0.1).astype(np.float32)
        x = torch.from_numpy(wav).unsqueeze(0)
        with torch.no_grad():
            v2 = loader.extract(x.clone(), output_layer=12, output_dim=768)
            v1 = loader.extract(x.clone(), output_layer=9, output_dim=256)
        goldens[f"{name}.wav"] = x
        goldens[f"{name}.v2"] = v2.cpu().to(torch.float32)
        goldens[f"{name}.v1"] = v1.cpu().to(torch.float32)
        print(f"golden {name}: wav {tuple(x.shape)} v2 {tuple(v2.shape)} v1 {tuple(v1.shape)}")

    golden_path = out_dir / "hubert_golden.safetensors"
    save_file(goldens, str(golden_path))
    print(f"exported {golden_path}")


def export_hubert_onnx(model_path: Path, out_dir: Path) -> None:
    """実 HuBERT を ONNX 化 (v2: last_hidden_state, v1: layer9+LN+final_proj)。"""
    import torch.nn as nn
    from rcwx.models.hubert_loader import HuBERTLoader

    out_dir.mkdir(parents=True, exist_ok=True)
    loader = HuBERTLoader(str(model_path), device="cpu", dtype=torch.float32)
    m = loader.model
    m.eval()

    class WrapV2(nn.Module):
        def forward(self, wav):
            return m(wav).last_hidden_state

    class WrapV1(nn.Module):
        def forward(self, wav):
            out = m(wav, output_hidden_states=True)
            feats = m.encoder.layer_norm(out.hidden_states[9])
            return loader.final_proj(feats)

    wav = torch.randn(1, 10880)
    for name, mod in (("hubert_v2.onnx", WrapV2().eval()), ("hubert_v1.onnx", WrapV1().eval())):
        path = out_dir / name
        torch.onnx.export(
            mod, (wav,), path,
            input_names=["wav"], output_names=["feats"],
            dynamic_axes={"wav": {1: "N"}, "feats": {1: "T"}},
            opset_version=17, dynamo=False,
        )
        print(f"exported {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", choices=["hubert", "synth", "hubert-onnx"])
    ap.add_argument("model", nargs="?", default=None, help="モデルファイルパス")
    ap.add_argument("--out", default=None, help="出力ディレクトリ")
    args = ap.parse_args()

    default_out = Path.home() / ".cache/rcwx/models/rust"
    out = Path(args.out) if args.out else default_out

    if args.target == "hubert-onnx":
        model = Path(args.model) if args.model else Path.home() / ".cache/rcwx/models/hubert/hubert_base.pt"
        if not model.exists():
            print(f"model not found: {model}", file=sys.stderr)
            sys.exit(1)
        export_hubert_onnx(model, out)
    elif args.target == "hubert":
        model = Path(args.model) if args.model else Path.home() / ".cache/rcwx/models/hubert/hubert_base.pt"
        if not model.exists():
            print(f"model not found: {model}", file=sys.stderr)
            sys.exit(1)
        convert_hubert(model, out)
    else:
        print("synth converter is M2 — 未実装", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
