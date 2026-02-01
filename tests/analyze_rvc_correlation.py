"""
RVC WebUIモードの相関が低い原因調査
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rcwx.config import RCWXConfig
from rcwx.pipeline.inference import RVCPipeline


def analyze_rvc_correlation_issue():
    """
    RVC WebUIモードで不連続が0なのに相関が低い（0.0014）原因を調査
    """

    # RVC WebUIモードの特徴
    # 1. オーバーラップ処理：各チャンクが前のチャンクとオーバーラップ
    # 2. SOLA：オーバーラップ領域で最適な位相位置を探索
    # 3. 出力長：完全一致（length_diff=0）

    print("\n=== RVC WebUIモードの問題分析 ===")

    # 可能な原因
    print("\n原因の分析:")

    print("\n1. オーバーラップ領域の問題")
    print("   RVC WebUIモードでは各チャンクが前のチャンクとオーバーラップ")
    print("   SOLAがオーバーラップ領域で最適な位相位置を探索")
    print("   しかし、元のバッチ処理と同じロジックを使っているか？")

    print("\n2. 位相整合の問題")
    print("   SOLAは相関係数が最大になる位置を探索")
    print("   負の相関（-0.0014）は、信号が逆相関を持っている可能性")
    print("   原因：SOLAオフセット探索が誤っている、または")
    print("         クロスフェード長が長すぎて信号が逆転")

    print("\n3. オーバーラップの相乗効果")
    print("   オーバーラップ領域でSOLAを適用すると、")
    print("   同じ部分が2回処理される可能性がある")
    print("   1回目：前のチャンクのSOLAクロスフェード")
    print("   2回目：現在のチャンクのSOLA（オーバーラップ領域）")
    print("   これがアーティファクトを引き起こす可能性")

    print("\n4. チャンク境界の処理")
    print("   RVC WebUIモードのチャンク構造:")
    print("   - hop_size = chunk_samples - overlap_samples")
    print("   - 各チャンクは hop_samples ずつ進む")
    print("   - オーバーラップ領域 = chunk_samples - hop_samples = overlap_samples")

    print("\n5. 可能な解決策")
    print("   a. SOLAをオーバーラップ領域全体に適用せず、hop境界のみにする")
    print("   b. クロスフェード長を調整（オーバーラップより短く）")
    print("   c. RVC WebUIモードでも相関が低いので、根本的な問題がある可能性")
    print("      （例：リサンプリング、パディング、フィルタリング）")

    print("\n=== 推奨アクション ===")
    print("\n1. w-okadaモードを改善（不連続34→0へ）")
    print("   - SOLAオフセット探索の改善")
    print("   - クロスフェード長の最適化")
    print("   - オーバーラップなしモードでのテスト")

    print("\n2. RVC WebUIモードの相関を改善（0.0014→0.93へ）")
    print("   - リサンプリング方法の確認")
    print("   - パディングの最適化")
    print("   - w-okadaモードとの出力比較")

    print("\n3. 両モードの共通問題の調査")
    print("   - 出力がバッチ処理と完全に一致しない原因")
    print("   - HuBERT特徴量子化の影響")
    print("   - F0量子化の影響")

    return True


if __name__ == "__main__":
    analyze_rvc_correlation_issue()
