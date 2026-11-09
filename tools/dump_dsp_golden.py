"""rcwx-dsp 移植用 golden ダンプ (dev ツール・実行時不要)。

target/golden/dsp_golden.npz に各 DSP モジュールの入出力対を保存し、
Rust 側パリティテスト (crates/rcwx-dsp/tests/) の基準にする。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

GOLDEN = Path(__file__).resolve().parents[1] / "target/golden/dsp_golden.npz"


def main() -> None:
    from rcwx.audio.noise_gate import NoiseGate
    from rcwx.audio.postprocess import Postprocessor
    from rcwx.audio.resample import StatefulResampler, resample
    from rcwx.audio.sola import SolaState, sola_crossfade
    from rcwx.audio.denoise import StreamingSpectralGate
    from rcwx.pipeline.f0_processing import apply_moe_f0_style
    from rcwx.pipeline.realtime_unified import _max_normalized_lag_correlation

    rng = np.random.RandomState(42)
    out: dict[str, np.ndarray] = {}

    # ---- resample: 48k -> 16k (batch + streaming) ----
    sig48 = (rng.randn(24000) * 0.3).astype(np.float32)
    sig48[:200] = 0.3 * np.sin(2 * np.pi * 440 * np.arange(200) / 48000)
    out["rs_in48"] = sig48
    out["rs_batch_48_16"] = resample(sig48, 48000, 16000, "poly")
    out["rs_batch_48_16_lin"] = resample(sig48, 48000, 16000, "linear")
    sr = StatefulResampler(48000, 16000)
    chunks = [sig48[i : i + 1440] for i in range(0, len(sig48), 1440)]
    stream = np.concatenate([sr.resample_chunk(c) for c in chunks])
    out["rs_stream_48_16"] = stream

    # ---- resample: 16k -> 48k (batch + streaming) ----
    sig16 = (rng.randn(8000) * 0.3).astype(np.float32)
    out["rs_in16"] = sig16
    out["rs_batch_16_48"] = resample(sig16, 16000, 48000, "poly")
    sr = StatefulResampler(16000, 48000)
    chunks = [sig16[i : i + 480] for i in range(0, len(sig16), 480)]
    stream = np.concatenate([sr.resample_chunk(c) for c in chunks])
    out["rs_stream_16_48"] = stream

    # ---- SOLA: 周期信号の連続チャンク + target_len 補正 ----
    t = np.arange(1600) / 48000.0
    cf, search, target = 480, 480, 960
    state = SolaState(crossfade_samples=cf, search_samples=search)
    sola_in, sola_out = [], []
    for i in range(8):
        seg = (0.6 * np.sin(2 * np.pi * 220.0 * (t + i * 1600 / 48000))).astype(np.float32)
        seg += (0.05 * rng.randn(1600)).astype(np.float32)
        o = sola_crossfade(seg, state, target_len=target)
        sola_in.append(seg)
        sola_out.append(o)
    out["sola_in"] = np.stack(sola_in)
    out["sola_out_lens"] = np.array([len(o) for o in sola_out], np.int64)
    out["sola_out"] = np.concatenate(sola_out)
    out["sola_flush"] = state.buffer if state.buffer is not None else np.zeros(0, np.float32)

    # ---- StreamingSpectralGate: hop 320 (20ms) 連続 ----
    n = 16000
    t = np.arange(n) / 16000.0
    speech = np.zeros(n, np.float32)
    speech[: n // 2] = 0.5 * np.sin(2 * np.pi * 180 * t[: n // 2]) * (
        0.5 + 0.5 * np.sin(2 * np.pi * 3 * t[: n // 2])
    )
    noisy = speech + (0.1 * rng.randn(n)).astype(np.float32)
    gate = StreamingSpectralGate(threshold_db=6.0, reduction_db=-24.0)
    outs = [gate.process(noisy[i : i + 320]) for i in range(0, n, 320)]
    out["spec_in"] = noisy
    out["spec_out"] = np.concatenate(outs)

    # ---- NoiseGate: manual + auto (hop 320) ----
    env = np.zeros(16000, np.float32)
    env[:2400] = 0.001 * rng.randn(2400)  # ノイズのみ
    lvl = np.concatenate([np.full(1600, 0.02), np.full(1600, 0.002), np.full(1600, 0.05)])
    env[2400:7200] += lvl.astype(np.float32)
    env[7200:] = 0.0005 * rng.randn(8800)
    for gcfg, tag in [(dict(threshold_db=-40.0), "man"), (dict(threshold_db=-40.0), "auto")]:
        ng = NoiseGate(16000, **gcfg)
        if tag == "auto":
            ng.auto_threshold = True
            ng.set_sensitivity("mid")
        outs = [ng.process(env[i : i + 320]) for i in range(0, 16000, 320)]
        out[f"ng_in_{tag}"] = env
        out[f"ng_out_{tag}"] = np.concatenate(outs)

    # ---- Postprocessor: 既定設定でチャンク列 ----
    pp = Postprocessor(48000)
    rng2 = np.random.RandomState(7)
    chunks = []
    for i in range(10):
        c = (0.3 * rng2.randn(1440)).astype(np.float32)
        c[300:700] += 0.8 * np.sin(2 * np.pi * 440 * np.arange(400) / 48000).astype(np.float32)
        chunks.append(c)
    outs = [pp.process(c.copy()) for c in chunks]
    out["pp_in"] = np.stack(chunks)
    out["pp_out"] = np.stack(outs)

    # ---- moe f0 style ----
    import torch

    f0 = np.zeros(200, np.float32)
    f0[:150] = 120 + 40 * np.sin(np.arange(150) / 18.0)
    f0[70:85] = 0.0  # 穴
    styled = apply_moe_f0_style(torch.from_numpy(f0)[None, :], 0.45)[0].numpy()
    out["moe_in"] = f0
    out["moe_out"] = styled

    # ---- feedback correlation ----
    hay = (0.2 * rng.randn(4096)).astype(np.float32)
    needle = hay[1234:1234 + 512].copy()
    out["fb_needle"] = needle
    out["fb_hay"] = hay
    out["fb_corr"] = np.array([_max_normalized_lag_correlation(needle, hay)], np.float64)
    hay2 = (0.2 * rng.randn(4096)).astype(np.float32)
    out["fb_hay2"] = hay2
    out["fb_corr2"] = np.array([_max_normalized_lag_correlation(needle, hay2)], np.float64)

    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    np.savez(GOLDEN, **out)
    for k, v in out.items():
        print(f"{k}: {v.shape} {v.dtype}")


if __name__ == "__main__":
    main()
