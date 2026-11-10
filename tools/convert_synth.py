"""Synthesizer (RVC .pth) 重み変換 + golden ダンプ (dev 専用)。

- weight_norm はロード時に解決済みの重みを保存 (SynthesizerLoader が remove_weight_norm 済み)
- enc_q (PosteriorEncoder) は推論未使用のため除外
- golden: 決定論入力 (features/pitch/pitchf/sid, noise_scale=0) に対する infer() 出力

使い方:
    uv run python tools/convert_weights.py synth model/kurumi/kurumi.pth
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch


def convert_synth(model_path: Path, out_dir: Path) -> None:
    from safetensors.torch import save_file

    from rcwx.models.synthesizer import SynthesizerLoader

    out_dir.mkdir(parents=True, exist_ok=True)
    loader = SynthesizerLoader(model_path=str(model_path), device="cpu", dtype=torch.float32)
    model = loader.load()
    model.eval()

    sd = model.state_dict()
    skip_prefixes = ("enc_q.",)
    t: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if any(k.startswith(p) for p in skip_prefixes):
            continue
        t[k] = v.detach().cpu().to(torch.float32).contiguous()

    weights_path = out_dir / "synth.safetensors"
    save_file(t, str(weights_path))

    ckpt = torch.load(str(model_path), map_location="cpu", weights_only=True)
    cfg_list = ckpt.get("config", [])
    keys18 = ["spec_channels", "segment_size", "inter_channels", "hidden_channels",
              "filter_channels", "n_heads", "n_layers", "kernel_size", "p_dropout",
              "resblock", "resblock_kernel_sizes", "resblock_dilation_sizes",
              "upsample_rates", "upsample_initial_channel", "upsample_kernel_sizes",
              "spk_embed_dim", "gin_channels", "sr", "extra"]
    parsed = {k: cfg_list[i] for i, k in enumerate(keys18) if i < len(cfg_list)}
    config = {
        "version": getattr(loader, "version", 2) if hasattr(loader, "version") else 2,
        "has_f0": bool(getattr(loader, "has_f0", True)),
        "sr": int(getattr(loader, "sample_rate", 40000)),
        "n_speakers": int(sd["emb_g.weight"].shape[0]),
        # make_test_model.py 由来のランダム重みモデル (パリティ gate 緩和の判定用)
        "random_test": "RCWX random test" in str(ckpt.get("info", "")),
        **parsed,
    }
    cfg_path = out_dir / "synth.json"
    cfg_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"exported {weights_path} ({sum(v.numel() for v in t.values()) / 1e6:.1f}M params)")
    print(f"config: {config}")

    # ---- golden: 決定論 infer (noise_scale=0, SineGen 励振ノイズ 0) ----
    # SineGen の randn 励振ノイズは torch RNG 依存のため golden では 0 化する
    # (Rust 側も noise=0 で実装し、production では独自 RNG で加算)
    _sg = model.dec.m_source.l_sin_gen

    def _det_sinegen_forward(f0, upp, _sg=_sg):
        import torch.nn.functional as F
        with torch.no_grad():
            f0 = f0.unsqueeze(-1)
            sine_waves = _sg._f02sine_legacy(f0, upp)
            sine_waves = sine_waves * _sg.sine_amp
            uv = _sg._f02uv(f0)
            uv = F.interpolate(uv.transpose(2, 1), scale_factor=float(upp), mode="nearest").transpose(2, 1)
            sine_waves = sine_waves * uv
        return sine_waves, uv, torch.zeros_like(sine_waves)

    model.dec.m_source.l_sin_gen.forward = _det_sinegen_forward
    torch.manual_seed(0)
    frames = 33
    # v1 は 256-dim / v2 は 768-dim 特徴
    in_ch = 256 if getattr(loader, "version", 2) == 1 else 768
    feats = torch.randn(1, frames, in_ch) * 0.5
    pitch_hz = torch.where(
        torch.rand(1, frames) > 0.2,
        100.0 + 180.0 * torch.rand(1, frames),
        torch.zeros(1, frames),
    )
    # quantize (mel 逆写像) — f0_processing.quantize_f0_to_pitch と同じ式
    mel_min = 1127.0 * np.log(1.0 + 50.0 / 700.0)
    mel_max = 1127.0 * np.log(1.0 + 1100.0 / 700.0)
    mel = 1127.0 * torch.log1p(pitch_hz / 700.0)
    idx = ((mel - mel_min) * 254.0 / (mel_max - mel_min) + 1.0).round().long().clamp(1, 255)
    pitch_idx = idx

    phone_lengths = torch.tensor([frames])
    sid = torch.zeros(1, dtype=torch.long)

    pitch = pitch_idx  # int64
    with torch.no_grad():
        out = model.infer(
            feats,
            phone_lengths,
            pitch,
            pitch_hz,
            sid,
            skip_head=0,
            return_length=0,
            return_length2=0,
            noise_scale=0.0,
            all_frames_valid=True,
        )

    g: dict[str, torch.Tensor] = {
        "feats": feats,
        "pitch": pitch_idx.float(),
        "pitchf": pitch_hz,
        "audio": out[0, 0].cpu().to(torch.float32),
    }
    golden_path = out_dir / "synth_golden.safetensors"
    save_file(g, str(golden_path))
    print(f"golden: feats {tuple(feats.shape)} pitch {tuple(pitch.shape)} audio {tuple(out.shape)}")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", nargs="?", default="model/kurumi/kurumi.pth", help="RVC .pth モデル")
    ap.add_argument("--out", default=None, help="出力ディレクトリ (既定 ~/.cache/rcwx/models/rust)")
    args = ap.parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"model not found: {model_path}", file=sys.stderr)
        sys.exit(1)
    out_dir = Path(args.out) if args.out else Path.home() / ".cache/rcwx/models/rust"
    convert_synth(model_path, out_dir)


if __name__ == "__main__":
    main()
