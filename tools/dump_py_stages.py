"""段階ダンプ: Python run パイプラインの各段階出力を保存し Rust dump と比較する。

Rust 側は RCWX_CLI_DUMP=1 の rcwx-cli run が target/golden/cli_rust_*.bin を
書く。本スクリプトは同じ入力・同じ引数で Python 経路の対応段階を
target/golden/cli_py_*.bin として保存する (f32 バイナリ)。

使い方:
  uv run python tools/dump_py_stages.py <in.wav> <model.pth>
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = "target/golden"


def dump(name: str, x) -> None:
    a = np.asarray(x, dtype=np.float32)
    with open(os.path.join(OUT, f"cli_py_{name}.bin"), "wb") as f:
        f.write(a.tobytes())
    print(f"[dump] cli_py_{name}.bin  shape={a.shape}")


def main() -> None:
    import torch
    from scipy.io import wavfile

    wav_path = sys.argv[1]
    model = sys.argv[2]

    from rcwx.pipeline import inference as inf
    from rcwx.models import synthesizer as synth_mod

    # ---- patch: resample (16k 化) ----
    orig_resample = inf.resample if hasattr(inf, "resample") else None
    if orig_resample is not None:
        def resample_spy(x, in_sr, out_sr, **kw):
            y = orig_resample(x, in_sr, out_sr, **kw)
            if int(out_sr) == 16000 and getattr(resample_spy, "n", 0) == 0:
                dump("hp", y)
                resample_spy.n = 1
            return y
        inf.resample = resample_spy

    # ---- patch: HuBERT 特徴 ----
    orig_hub = inf.RVCPipeline._extract_hubert_features

    def hub_spy(self, audio_t, output_layer, output_dim, prof=None):
        feats = orig_hub(self, audio_t, output_layer, output_dim, prof)
        if not getattr(hub_spy, "done", False):
            dump("feats", feats.detach().float().cpu().numpy())
            hub_spy.done = True
        return feats

    inf.RVCPipeline._extract_hubert_features = hub_spy

    # ---- patch: F0 ----
    orig_f0 = inf.RVCPipeline._extract_f0_raw

    def f0_spy(self, f0_input, f0_method, prof=None):
        r = orig_f0(self, f0_input, f0_method, prof)
        if r is not None and not getattr(f0_spy, "done", False):
            dump("f0raw", r.detach().float().cpu().numpy())
            f0_spy.done = True
        return r

    inf.RVCPipeline._extract_f0_raw = f0_spy

    # ---- patch: 特徴 2x 補間 + synth 呼び出し ----
    orig_interp = inf.RVCPipeline._interpolate_features_2x

    def interp_spy(self, features):
        y = orig_interp(self, features)
        if not getattr(interp_spy, "done", False):
            dump("feats100", y.detach().float().cpu().numpy())
            interp_spy.done = True
        return y

    inf.RVCPipeline._interpolate_features_2x = interp_spy

    orig_infer = synth_mod.SynthesizerLoader.infer

    def synth_infer_spy(self, features, feature_lengths, pitch=None, pitchf=None, **kw):
        if not getattr(synth_infer_spy, "done", False):
            dump("sy_feats", features.detach().float().cpu().numpy())
            if pitch is not None:
                dump("sy_pitch", pitch.detach().float().cpu().numpy())
            if pitchf is not None:
                dump("sy_pitchf", pitchf.detach().float().cpu().numpy())
            synth_infer_spy.done = True
        out = orig_infer(self, features, feature_lengths, pitch=pitch, pitchf=pitchf, **kw)
        if not getattr(synth_infer_spy, "done2", False):
            dump("synthout", out.detach().float().cpu().numpy())
            synth_infer_spy.done2 = True
        return out

    # synthesizer loader が返すのは nn.Module (synthesizer.infer は mm の infer)
    # → モデルインスタンスの infer を patch するため loader をフック
    orig_load = synth_mod.SynthesizerLoader.load

    def load_spy(self):
        m = orig_load(self)
        if hasattr(m, "infer"):
            import types

            bound = m.infer

            def wrapper(*a, **k):
                if not getattr(load_spy, "d1", False) and len(a) >= 4:
                    dump("sy_feats", a[0].detach().float().cpu().numpy())
                    dump("sy_pitch", a[2].detach().float().cpu().numpy())
                    dump("sy_pitchf", a[3].detach().float().cpu().numpy())
                    load_spy.d1 = True
                out = bound(*a, **k)
                if not getattr(load_spy, "d2", False):
                    dump("synthout", out.detach().float().cpu().numpy())
                    load_spy.d2 = True
                return out

            m.infer = wrapper
        return m

    synth_mod.SynthesizerLoader.load = load_spy

    # ---- 本体: pipeline.infer を cmd_run 相当の引数で直接呼ぶ ----
    sr, audio = wavfile.read(wav_path)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147364838.0
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    pipeline = inf.RVCPipeline(
        model, device="xpu", dtype=torch.float32, use_f0=True, use_compile=False
    )
    with torch.no_grad():
        out = pipeline.infer(
            audio,
            input_sr=sr,
            pitch_shift=0,
            f0_method="rmvpe",
            index_rate=0.0,
            index_k=4,
            noise_scale=0.0,
            moe_boost=0.0,
        )
    dump("final", out)
    print("done")


if __name__ == "__main__":
    main()
