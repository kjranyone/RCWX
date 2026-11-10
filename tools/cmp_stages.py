"""段階比較: cli_rust_*.bin と cli_py_*.bin を突き合わせ corr/長さを表示。

使い方: uv run python tools/cmp_stages.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = "target/golden"


def load(name: str) -> np.ndarray:
    with open(os.path.join(OUT, name), "rb") as f:
        return np.frombuffer(f.read(), dtype=np.float32).copy()


def corr(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    a, b = a[:n].astype(np.float64), b[:n].astype(np.float64)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 0 else float("nan")


def lag_scan(a: np.ndarray, b: np.ndarray, max_lag: int = 960) -> tuple[int, float]:
    n = min(len(a), len(b))
    best = (0, -2.0)
    for lag in range(0, max_lag, 4):
        x, y = a[lag:n], b[: n - lag]
        c = corr(x, y)
        if c > best[1]:
            best = (lag, c)
    return best


def main() -> None:
    py_hp = load("cli_py_hp.bin")
    ru_hp = load("cli_rust_hp.bin")
    print(f"hp: py={len(py_hp)} rust={len(ru_hp)} corr={corr(py_hp, ru_hp):.6f}")

    py_feats = load("cli_py_feats.bin")
    ru_feats = load("cli_rust_feats.bin")
    print(
        f"feats: py={len(py_feats)} rust={len(ru_feats)} corr={corr(py_feats, ru_feats):.6f}"
    )

    py_f0 = load("cli_py_f0raw.bin")
    ru_f0 = load("cli_rust_f0raw.bin")
    print(f"f0raw: py={len(py_f0)} rust={len(ru_f0)} corr={corr(py_f0, ru_f0):.6f}")

    ru_f0c = load("cli_rust_f0.bin")
    print(f"rust f0 chain: corr(raw→chain)={corr(ru_f0, ru_f0c):.6f} len={len(ru_f0c)}")

    py_fe = load("cli_py_sy_feats.bin")
    ru_fe_full = ru_feats  # rust sy_feats は wavpad 全長分 (未 dump)
    print(f"sy_feats(py 入力): len={len(py_fe)}")

    py_pitch = load("cli_py_sy_pitch.bin")
    ru_pitch = load("cli_rust_pitch.bin")
    print(
        f"pitch: py={len(py_pitch)} rust={len(ru_pitch)} corr={corr(py_pitch, ru_pitch):.6f}"
    )
    pf = load("cli_py_sy_pitchf.bin")
    print(
        f"pitchf corr(py vs rust f0chain)={corr(pf, ru_f0c[: len(pf)]):.6f} len py={len(pf)}"
    )

    py_so = load("cli_py_synthout.bin")
    ru_so = load("cli_rust_synthout.bin")
    lag, c = lag_scan(py_so, ru_so)
    print(
        f"synthout: py={len(py_so)} rust={len(ru_so)} corr={corr(py_so, ru_so):.6f} "
        f"best(lag={lag})={c:.6f}"
    )

    py_fin = load("cli_py_final.bin")
    ru_fin = load("cli_rust_synthout.bin")
    print(f"final: corr(py_final vs rust_synthout)={corr(py_fin, ru_so):.6f}")


if __name__ == "__main__":
    main()
