"""Synth 段階別 golden ダンプ (dev 専用)。

convert_synth.py と同じ決定論入力 (synth_golden.safetensors の feats/pitch/pitchf)
で PyTorch 側の TextEncoder 中間出力をキャプチャし、Rust 側
(synth.rs enable_debug のフック名) と突き合わせるための
safetensors を書き出す。パリティ劣化の段階特定に使う。

フック名 (Rust 側と同一):
  x_in          : enc_p.encoder 入力 (emb_phone+emb_pitch, √h, lrelu 後) [B,h,T]
  attn0         : attn_layers[0] 出力 (conv_o 後)
  norm1_0       : norm_layers_1[0] 出力 (residual+LN)
  ffn0_gelu_mid : ffn_layers[0].conv_1 出力 (relu 後)
  ffn0          : ffn_layers[0] 出力 (conv_2 後)
  enc_out       : enc_p.encoder 出力 (全層後)
  proj_out      : enc_p.proj 出力 (stats) [B,384,T]

使い方:
    uv run python tools/dump_synth_stages.py <model.pth> \
        [--golden-dir ~/.cache/rcwx/models/rust] \
        [--out target/golden/synth_stages16.safetensors]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    from safetensors.numpy import save_file
    from safetensors.torch import load_file

    from rcwx.models.synthesizer import SynthesizerLoader

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", help="RVC .pth モデル")
    ap.add_argument(
        "--golden-dir",
        default=None,
        help="synth_golden.safetensors があるディレクトリ (既定 ~/.cache/rcwx/models/rust)",
    )
    ap.add_argument(
        "--out",
        default="target/golden/synth_stages16.safetensors",
        help="出力 safetensors",
    )
    args = ap.parse_args()

    golden_dir = (
        Path(args.golden_dir) if args.golden_dir else Path.home() / ".cache/rcwx/models/rust"
    )
    golden = load_file(str(golden_dir / "synth_golden.safetensors"))
    feats = (
        golden["feats"].unsqueeze(0) if golden["feats"].ndim == 2 else golden["feats"]
    )
    loader = SynthesizerLoader(str(args.model), device="cpu", dtype=torch.float32)
    model = loader.load()
    model.eval()

    frames = feats.shape[1]
    pitch_idx = golden["pitch"].long()
    pitch_hz = golden["pitchf"]
    if pitch_idx.ndim == 1:
        pitch_idx = pitch_idx.unsqueeze(0)
    if pitch_hz.ndim == 1:
        pitch_hz = pitch_hz.unsqueeze(0)

    stages: dict[str, np.ndarray] = {}

    def grab(name: str):
        def hook(_m, _i, o):
            t = o[0] if isinstance(o, tuple) else o
            stages[name] = t.detach().float().numpy()

        return hook

    def grab_input(name: str):
        def hook(_m, args):
            t = args[0]
            stages[name] = t.detach().float().numpy()

        return hook

    enc = model.enc_p.encoder
    enc.register_forward_pre_hook(grab_input("x_in"))
    enc.attn_layers[0].register_forward_hook(grab("attn0"))
    enc.norm_layers_1[0].register_forward_hook(grab("norm1_0"))

    def conv1_relu(_m, _i, o):
        stages["ffn0_gelu_mid"] = torch.relu(o.detach()).float().numpy()

    enc.ffn_layers[0].conv_1.register_forward_hook(conv1_relu)
    enc.ffn_layers[0].register_forward_hook(grab("ffn0"))
    enc.register_forward_hook(grab("enc_out"))
    model.enc_p.proj.register_forward_hook(grab("proj_out"))

    # convert_synth.py と同じ決定論 SineGen (励振ノイズ 0)
    _sg = model.dec.m_source.l_sin_gen

    def _det_sinegen_forward(f0, upp, _sg=_sg):
        import torch.nn.functional as F

        with torch.no_grad():
            f0 = f0.unsqueeze(-1)
            sine_waves = _sg._f02sine_legacy(f0, upp)
            sine_waves = sine_waves * _sg.sine_amp
            uv = _sg._f02uv(f0)
            uv = F.interpolate(
                uv.transpose(2, 1), scale_factor=float(upp), mode="nearest"
            ).transpose(2, 1)
            sine_waves = sine_waves * uv
        return sine_waves, uv, torch.zeros_like(sine_waves)

    model.dec.m_source.l_sin_gen.forward = _det_sinegen_forward

    phone_lengths = torch.tensor([frames])
    sid = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        out = model.infer(
            feats,
            phone_lengths,
            pitch_idx,
            pitch_hz,
            sid,
            skip_head=0,
            return_length=0,
            return_length2=0,
            noise_scale=0.0,
            all_frames_valid=True,
        )
    stages["audio"] = out[0, 0].float().numpy()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.astype(np.float32) for k, v in stages.items()}, str(out_path))
    print(f"{out_path}:")
    for k, v in stages.items():
        print(f"  {k:<14} {v.shape}")


if __name__ == "__main__":
    main()
