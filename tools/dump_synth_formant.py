"""Formant shift 経路 (return_length2) の golden ダンプ (dev 専用)。

tools/convert_synth.py と同一の決定論入力 (同 seed) を使い、
formant = ±semitone それぞれについて upstream rtrvc.py と同一の補償:
  factor    = 2^(formant/12)
  pitch_idx = quantize(f0 * 2^(-formant/12))
  pitchf    = f0 * 2^(-formant/12) * factor  (= f0)
  T2        = ceil(T * factor)
で Synthesizer.infer(return_length=T, return_length2=T2) を呼び、
伸長状態の出力 [T2*zc] を保存する。Rust 側は
Synth::infer_formant_stretched と cosine 比較する (リサンプル戻し前)。

使い方:
    uv run python tools/dump_synth_formant.py rvc_models/voice.pth \
        --out ~/.cache/rcwx/models/rust
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch


def quantize_f0(pitch_hz: torch.Tensor) -> torch.Tensor:
    # f0_processing.quantize_f0_to_pitch と同じ式
    mel_min = 1127.0 * np.log(1.0 + 50.0 / 700.0)
    mel_max = 1127.0 * np.log(1.0 + 1100.0 / 700.0)
    mel = 1127.0 * torch.log1p(pitch_hz / 700.0)
    return ((mel - mel_min) * 254.0 / (mel_max - mel_min) + 1.0).round().long().clamp(1, 255)


def dump_case(model, out_dir: Path, tag: str, feats, pitch_hz, frames, formant: float) -> None:
    from safetensors.torch import save_file

    factor = 2.0 ** (formant / 12.0)
    pitch_idx = quantize_f0(pitch_hz * (2.0 ** (-formant / 12.0)))
    pitchf = pitch_hz * (2.0 ** (-formant / 12.0)) * factor
    t2 = int(math.ceil(frames * factor))
    phone_lengths = torch.tensor([frames])
    sid = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        out = model.infer(
            feats,
            phone_lengths,
            pitch_idx,
            pitchf,
            sid,
            skip_head=0,
            return_length=frames,
            return_length2=t2,
            noise_scale=0.0,
            all_frames_valid=True,
        )
    g = {
        "feats": feats,
        "pitch": pitch_idx.float(),
        "pitchf": pitch_hz,
        "audio": out[0, 0].cpu().to(torch.float32),
        "factor": torch.tensor([factor], dtype=torch.float32),
    }
    path = out_dir / f"synth_formant_{tag}.safetensors"
    save_file(g, str(path))
    print(
        f"formant {tag} ({formant:+.0f} st, factor {factor:.5f}): "
        f"audio {tuple(out.shape)} -> {path.name}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out_dir = args.out or Path.home() / ".cache/rcwx/models/rust"
    out_dir.mkdir(parents=True, exist_ok=True)

    from rcwx.models.synthesizer import SynthesizerLoader

    loader = SynthesizerLoader(model_path=str(args.model), device="cpu", dtype=torch.float32)
    model = loader.load()
    model.eval()

    # convert_synth.py と同じ決定論化 (SineGen 励振ノイズ 0)
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

    # convert_synth.py と同一 seed → 同一入力
    torch.manual_seed(0)
    frames = 33
    in_ch = 256 if getattr(loader, "version", 2) == 1 else 768
    feats = torch.randn(1, frames, in_ch) * 0.5
    pitch_hz = torch.where(
        torch.rand(1, frames) > 0.2,
        100.0 + 180.0 * torch.rand(1, frames),
        torch.zeros(1, frames),
    )

    dump_case(model, out_dir, "pos", feats, pitch_hz, frames, 5.0)
    dump_case(model, out_dir, "neg", feats, pitch_hz, frames, -3.0)


if __name__ == "__main__":
    main()
