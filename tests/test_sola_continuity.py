"""
SOLAの残存問題調査

不連続性が38箇所残っている原因を特定
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rcwx.audio.crossfade import SOLAState, apply_sola_crossfade


def analyze_discontinuity_patterns():
    """不連続パターンを分析"""

    sr = 48000
    crossfade_samples = int(sr * 0.05)
    context_samples = int(sr * 0.05)

    print(f"\n=== 不連続パターン分析 ===")
    print(f"サンプリングレート: {sr}Hz")
    print(f"クロスフェード長: {crossfade_samples}サンプル")
    print(f"コンテキスト長: {context_samples}サンプル")

    # 実際の推論出力長のバリエーションをシミュレート
    # HuBERT量子化により、チャンクごとに異なる長さになる

    # パターン1: 正常長（19200サンプル @ 40kHz）
    print(f"\n--- パターン1: 正常長 (19200サンプル @ 40kHz -> 23040 @ 48kHz) ---")
    state = SOLAState.create(crossfade_samples, sr)
    chunk = np.sin(2 * np.pi * 440 * np.arange(23040) / sr).astype(np.float32) * 0.5
    result = apply_sola_crossfade(chunk, state, wokada_mode=True, context_samples=context_samples)
    print(f"  入力: {len(chunk)}")
    print(f"  出力: {len(result.audio)}")
    print(f"  出力率: {len(result.audio) / len(chunk) * 100:.1f}%")

    # パターン2: 短い（HuBERT量子化による -800サンプル）
    print(f"\n--- パターン2: 量子化で短い (22400サンプル) ---")
    state = SOLAState.create(crossfade_samples, sr)
    chunk = np.sin(2 * np.pi * 440 * np.arange(22400) / sr).astype(np.float32) * 0.5
    result = apply_sola_crossfade(chunk, state, wokada_mode=True, context_samples=context_samples)
    print(f"  入力: {len(chunk)}")
    print(f"  出力: {len(result.audio)}")
    print(f"  出力率: {len(result.audio) / len(chunk) * 100:.1f}%")

    # パターン3: 非常に短い（推論失敗など）
    print(f"\n--- パターン3: 非常に短い (5000サンプル) ---")
    state = SOLAState.create(crossfade_samples, sr)
    chunk = np.sin(2 * np.pi * 440 * np.arange(5000) / sr).astype(np.float32) * 0.5
    result = apply_sola_crossfade(chunk, state, wokada_mode=True, context_samples=context_samples)
    print(f"  入力: {len(chunk)}")
    print(f"  出力: {len(result.audio)}")
    print(f"  出力率: {len(result.audio) / len(chunk) * 100:.1f}%")

    # パターン4: 長さが変動するシーケンス
    print(f"\n--- パターン4: 長さ変動シーケンス ---")
    state = SOLAState.create(crossfade_samples, sr)
    sequence_lengths = [23040, 22400, 23520, 22400, 23040]  # 実際の出力長
    output_lengths = []
    for i, length in enumerate(sequence_lengths):
        chunk = np.sin(2 * np.pi * 440 * np.arange(length) / sr).astype(np.float32) * 0.5
        result = apply_sola_crossfade(
            chunk, state, wokada_mode=True, context_samples=context_samples
        )
        output_lengths.append(len(result.audio))
        print(
            f"  Chunk {i + 1}: 入力={length}, 出力={output_lengths[-1]}, 差={length - output_lengths[-1]}"
        )

    # 出力長の変動を分析
    print(f"\n出力長の変動:")
    output_var = np.var(output_lengths)
    output_range = max(output_lengths) - min(output_lengths)
    print(f"  分散: {output_var:.1f}")
    print(f"  範囲: {output_range} サンプル")
    print(f"  平均: {np.mean(output_lengths):.1f}")

    # 出力長が一定でない場合、クリックが発生する可能性
    if output_range > 100:
        print(f"  ⚠️  出力長が{output_range}サンプル変動しています")
        print(f"     これがクリックの原因の可能性があります")


def analyze_crossfade_quality():
    """クロスフェードの品質を分析"""

    sr = 48000
    crossfade_samples = int(sr * 0.05)

    print(f"\n=== クロスフェード品質分析 ===")

    # 連続する2つのチャンクでクロスフェードをシミュレート
    state = SOLAState.create(crossfade_samples, sr)

    # チャンク1: 440Hz
    t1 = np.arange(10000) / sr
    chunk1 = (np.sin(2 * np.pi * 440 * t1) * 0.5).astype(np.float32)

    result1 = apply_sola_crossfade(chunk1, state, wokada_mode=True, context_samples=2400)

    # チャンク2: 220Hz（ピッチが変わる）
    t2 = np.arange(10000) / sr
    chunk2 = (np.sin(2 * np.pi * 220 * t2) * 0.5).astype(np.float32)

    result2 = apply_sola_crossfade(chunk2, state, wokada_mode=True, context_samples=2400)

    # 結合
    combined = np.concatenate([result1.audio, result2.audio])

    # 接続点付近の不連続を検出
    joint_point = len(result1.audio)
    window_size = 100

    if joint_point + window_size < len(combined):
        before = combined[joint_point - window_size : joint_point]
        after = combined[joint_point : joint_point + window_size]

        # 不連続度を計算
        discontinuity = abs(after[0] - before[-1])
        print(f"接続点の不連続: {discontinuity:.6f}")

        # 前後のエネルギーを比較
        energy_before = np.mean(before**2)
        energy_after = np.mean(after**2)
        energy_ratio = energy_after / energy_before if energy_before > 0 else 0

        print(f"エネルギー比: {energy_ratio:.2f} ({energy_before:.6f} → {energy_after:.6f})")

        if discontinuity > 0.1:
            print(f"  ⚠️  不連続が大きいです（{discontinuity:.3f}）")

        if energy_ratio < 0.5 or energy_ratio > 2.0:
            print(f"  ⚠️  エネルギーが急変しています（{energy_ratio:.2f}）")


if __name__ == "__main__":
    analyze_discontinuity_patterns()
    analyze_crossfade_quality()
