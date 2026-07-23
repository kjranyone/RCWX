"""Micro-benchmark for Aggressive 20ms streaming hot path.

Instruments host-side stages that are currently invisible in [PERF]
(highpass, history assembly, F0 postprocess) and measures end-to-end
infer_streaming wall times on XPU with Graph warm-up.

Usage:
    uv run python tests/models/bench_streaming_hotpath.py \\
        --model rvc_models/hogaraka/hogarakav2.pth \\
        [--audio sample_data/seki.wav] [--hops 200]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rcwx.audio.resample import resample
from rcwx.device import get_device, get_dtype
from rcwx.pipeline.inference import (
    RVCPipeline,
    StreamingParams,
    apply_f0_filter_chain,
    highpass_filter,
    quantize_f0_to_pitch,
)


def _pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def _fmt(values: list[float]) -> str:
    if not values:
        return "n/a"
    return (
        f"p50={_pct(values, 50):5.2f}  p95={_pct(values, 95):5.2f}  "
        f"p99={_pct(values, 99):5.2f}  mean={statistics.fmean(values):5.2f}  "
        f"n={len(values)}"
    )


def load_audio_16k(path: str) -> np.ndarray:
    import scipy.io.wavfile

    sr, data = scipy.io.wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    else:
        data = data.astype(np.float32)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != 16000:
        data = resample(data, sr, 16000)
    return data.astype(np.float32)


def micro_host_ops(audio_16k: np.ndarray, hops: int, hop: int = 320) -> None:
    """Isolate host-side costs at Aggressive 20ms shape."""
    context = 8960  # 0.56s @ 16k
    history = np.zeros(context, dtype=np.float32)
    # seed with speech
    history[:] = audio_16k[:context]

    t_copy: list[float] = []
    t_concat: list[float] = []
    t_hp: list[float] = []
    t_pad: list[float] = []
    t_from_np: list[float] = []
    t_h2d: list[float] = []
    t_f0_chain: list[float] = []
    t_quantize: list[float] = []

    # Synthetic F0 track matching streaming length (~56 frames + history)
    f0 = torch.zeros(1, 80, dtype=torch.float32)
    # voiced contour around 200Hz
    f0[:, 10:70] = 200.0 + 20.0 * torch.sin(torch.linspace(0, 6, 60))

    device = get_device("auto")
    for i in range(hops):
        hop_audio = audio_16k[(i * hop) % max(1, len(audio_16k) - hop) :][:hop]
        if len(hop_audio) < hop:
            hop_audio = np.pad(hop_audio, (0, hop - len(hop_audio)))

        t0 = time.perf_counter()
        combined = np.concatenate([history, hop_audio])
        t_concat.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        history = combined[-context:].copy()
        t_copy.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        filtered = highpass_filter(history, sr=16000, cutoff=48)
        t_hp.append((time.perf_counter() - t0) * 1000)

        t_pad_amt = 800
        t0 = time.perf_counter()
        padded = np.pad(filtered, (t_pad_amt, t_pad_amt), mode="reflect")
        # fixed hubert input size like streaming
        fixed = ((context + 2 * t_pad_amt + 319) // 320) * 320
        if len(padded) < fixed:
            padded = np.pad(padded, (0, fixed - len(padded)), mode="reflect")
        t_pad.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        cpu_t = torch.from_numpy(padded).float().unsqueeze(0)
        t_from_np.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        _ = cpu_t.to(device)
        if "xpu" in str(device):
            torch.xpu.synchronize()
        elif "cuda" in str(device):
            torch.cuda.synchronize()
        t_h2d.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        f0_out = apply_f0_filter_chain(
            f0,
            f0_lowpass_cutoff_hz=16.0,
            enable_octave_flip_suppress=True,
            enable_f0_slew_limit=True,
            f0_slew_max_step_st=3.6,
            f0_hole_fill_ms=30.0,
        )
        t_f0_chain.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        _ = quantize_f0_to_pitch(f0_out)
        t_quantize.append((time.perf_counter() - t0) * 1000)

    # skip first 10 for warm caches
    def tail(xs: list[float]) -> list[float]:
        return xs[10:]

    print("=== Host micro-ops (Aggressive 20ms shape, ms) ===")
    print(f"  history concat : {_fmt(tail(t_concat))}")
    print(f"  history copy   : {_fmt(tail(t_copy))}")
    print(f"  highpass 0.56s : {_fmt(tail(t_hp))}")
    print(f"  reflect pad    : {_fmt(tail(t_pad))}")
    print(f"  torch.from_np  : {_fmt(tail(t_from_np))}")
    print(f"  H2D+sync       : {_fmt(tail(t_h2d))}")
    print(f"  F0 filter chain: {_fmt(tail(t_f0_chain))}")
    print(f"  quantize pitch : {_fmt(tail(t_quantize))}")
    host_sum = [
        a + b + c + d + e + f
        for a, b, c, d, e, f in zip(
            tail(t_concat),
            tail(t_copy),
            tail(t_hp),
            tail(t_pad),
            tail(t_from_np),
            tail(t_f0_chain),
        )
    ]
    print(f"  SUM (no H2D)   : {_fmt(host_sum)}")
    print()


def bench_e2e(
    model_path: str,
    audio_16k: np.ndarray,
    hops: int,
    f0_method: str,
    index_rate: float,
    index_path: str | None,
) -> None:
    device = get_device("auto")
    dtype = get_dtype(device, "float16")
    print(f"Device={device} dtype={dtype}")
    print(f"Model={model_path}")
    print(f"F0={f0_method} index_rate={index_rate}")
    print()

    pipe = RVCPipeline(
        model_path,
        index_path=index_path,
        device=str(device),
        dtype=dtype,
        use_compile=False,
    )
    pipe.load()

    hop = 320  # 20ms @ 16k
    overlap = 320  # 100% overlap for aggressive
    params = StreamingParams(
        pitch_shift=0,
        f0_method=f0_method,
        index_rate=index_rate,
        index_k=4,
        voice_gate_mode="off",
        use_parallel_extraction=True,
        noise_scale=0.45,
        sola_extra_samples=1200,  # ~30ms @ 40k
        moe_boost=0.0,
        hubert_context_sec=0.56,
        f0_context_sec=0.10 if f0_method == "swiftf0" else 0.32,
        prime_hubert_history=True,
        output_sample_rate=48000,
        fixed_harmonics=True,
    )

    # Warmup / graph capture
    print("Warming up (graph capture)...")
    pipe.clear_cache()
    seed = audio_16k[: hop + overlap]
    if len(seed) < hop + overlap:
        seed = np.pad(seed, (0, hop + overlap - len(seed)))
    for _ in range(8):
        _ = pipe.infer_streaming(seed, overlap, params)
    if "xpu" in str(device):
        torch.xpu.synchronize()
    print("Warmup done.")
    print()

    wall: list[float] = []
    stages: dict[str, list[float]] = {
        "hubert_ms": [],
        "f0_ms": [],
        "faiss_ms": [],
        "synth_ms": [],
        "output_resample_ms": [],
    }

    # Force profile every hop for this bench
    import rcwx.pipeline.inference as inf_mod

    old_interval = inf_mod.STAGE_PROFILE_INTERVAL
    inf_mod.STAGE_PROFILE_INTERVAL = 1
    try:
        pipe.clear_cache()
        # re-prime
        _ = pipe.infer_streaming(seed, overlap, params)

        pos = 0
        for i in range(hops):
            if pos + hop > len(audio_16k):
                pos = 0
            hop_audio = audio_16k[pos : pos + hop]
            pos += hop
            if len(hop_audio) < hop:
                hop_audio = np.pad(hop_audio, (0, hop - len(hop_audio)))

            # Build [overlap|hop] like realtime (overlap from previous is
            # internal; we pass full window with fixed overlap length).
            # For the bench we feed hop-sized "new" audio by embedding it
            # in a 2-hop window — history handles context.
            window = np.concatenate([hop_audio, hop_audio])  # 40ms window
            # Better: use true consecutive audio
            # Actually realtime passes [overlap_buf | new_hop]. History is
            # inside the pipeline. So pass 2 hops of consecutive audio with
            # overlap_samples = hop.
            if pos >= 2 * hop:
                window = audio_16k[pos - 2 * hop : pos]
            else:
                window = np.concatenate([hop_audio, hop_audio])

            t0 = time.perf_counter()
            _ = pipe.infer_streaming(window, hop, params)
            # Natural D2H already happened; measure wall only
            wall.append((time.perf_counter() - t0) * 1000)

            st = getattr(pipe, "stage_times", {}) or {}
            for k in stages:
                if k in st:
                    stages[k].append(float(st[k]))
    finally:
        inf_mod.STAGE_PROFILE_INTERVAL = old_interval

    # Drop first 20 hops (prime + residual capture)
    def tail(xs: list[float], n: int = 20) -> list[float]:
        return xs[n:] if len(xs) > n else xs

    print("=== E2E infer_streaming wall (ms) ===")
    print(f"  wall           : {_fmt(tail(wall))}")
    for k, xs in stages.items():
        print(f"  {k:18s}: {_fmt(tail(xs))}")

    # Gap analysis: wall - max(hubert,f0) - faiss - synth
    # (synth_ms already includes out_rs + D2H)
    gaps: list[float] = []
    w = tail(wall)
    h = tail(stages["hubert_ms"])
    f = tail(stages["f0_ms"])
    fa = tail(stages["faiss_ms"])
    s = tail(stages["synth_ms"])
    n = min(len(w), len(h), len(s))
    for i in range(n):
        hub = h[i] if i < len(h) else 0.0
        f0v = f[i] if i < len(f) else 0.0
        fav = fa[i] if i < len(fa) else 0.0
        # GPU stages may be event-timed (no host wait). Wall ≈ host + critical path.
        # Approximate host gap = wall - max(hubert,f0) - faiss - synth? No:
        # synth_ms is wall-clock from synth start incl D2H, so:
        # wall ≈ pre_host + max(hubert_wait, f0) + faiss + f0_post + synth_wall
        # We can't separate cleanly; report wall - synth as pre-synth wall.
        gaps.append(w[i] - s[i] if i < len(s) else w[i])
    print(f"  pre-synth wall : {_fmt(gaps)}  (wall - synth_ms; includes hubert/f0/host)")
    print()

    deadline = 20.0
    miss = sum(1 for v in tail(wall) if v > deadline)
    print(
        f"Deadline miss (>20ms): {miss}/{len(tail(wall))} "
        f"({100.0 * miss / max(1, len(tail(wall))):.1f}%)"
    )
    near = sum(1 for v in tail(wall) if v > 16.0)
    print(
        f"Near miss (>16ms):     {near}/{len(tail(wall))} "
        f"({100.0 * near / max(1, len(tail(wall))):.1f}%)"
    )
    pipe.unload()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="rvc_models/hogaraka/hogarakav2.pth",
    )
    parser.add_argument("--audio", default="sample_data/seki.wav")
    parser.add_argument("--hops", type=int, default=200)
    parser.add_argument("--f0-method", default="swiftf0")
    parser.add_argument("--index-rate", type=float, default=0.0)
    parser.add_argument("--index", default=None)
    parser.add_argument("--skip-e2e", action="store_true")
    args = parser.parse_args()

    audio = load_audio_16k(args.audio)
    print(f"Audio: {args.audio}  {len(audio)/16000:.2f}s")
    print()

    micro_host_ops(audio, hops=min(args.hops, 150))

    if not args.skip_e2e:
        index = args.index
        if index is None and args.index_rate > 0:
            cand = Path(args.model).with_suffix(".index")
            # common sibling names
            parent = Path(args.model).parent
            idxs = list(parent.glob("*.index"))
            if idxs:
                index = str(idxs[0])
        bench_e2e(
            args.model,
            audio,
            hops=args.hops,
            f0_method=args.f0_method,
            index_rate=args.index_rate,
            index_path=index,
        )


if __name__ == "__main__":
    main()
