"""
SOLA問題の診断スクリプト

クリックノイズ（プチプチ）の原因を調査
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from rcwx.audio.crossfade import SOLAState, apply_sola_crossfade, flush_sola_buffer
from rcwx.pipeline.inference import RVCPipeline
from rcwx.pipeline.realtime import RealtimeConfig, RealtimeVoiceChanger
from rcwx.audio.resample import resample
from scipy.io import wavfile


def load_test_audio(sr: int = 48000) -> np.ndarray:
    """テスト用オーディオを生成（シンプルなトーン）"""
    duration = 2.0  # 秒
    t = np.arange(int(sr * duration)) / sr
    # 440Hz + 低い周波数成分（人声をシミュレート）
    audio = 0.7 * np.sin(2 * np.pi * 440 * t)
    audio += 0.3 * np.sin(2 * np.pi * 220 * t)
    audio += 0.2 * np.sin(2 * np.pi * 880 * t)
    return audio.astype(np.float32)


def analyze_sola_behavior():
    """SOLAの挙動を詳細分析"""

    sr = 48000
    crossfade_sec = 0.05
    crossfade_samples = int(sr * crossfade_sec)

    print(f"\n=== SOLA動作分析 ===")
    print(f"サンプリングレート: {sr}Hz")
    print(f"クロスフェード長: {crossfade_sec}秒 ({crossfade_samples}サンプル)")

    # SOLAステート初期化
    state = SOLAState.create(crossfade_samples, sr)

    print(f"\nSOLA設定:")
    print(f"  sola_buffer_frame: {state.sola_buffer_frame}")
    print(f"  sola_search_frame: {state.sola_search_frame}")

    # テストケース1: 正常長のチャンク
    print(f"\n--- テストケース1: 正常長チャンク (10000サンプル) ---")
    chunk1 = np.sin(2 * np.pi * 440 * np.arange(10000) / sr).astype(np.float32) * 0.5

    result1 = apply_sola_crossfade(chunk1, state, wokada_mode=True, context_samples=2400)
    print(f"  入力長: {len(chunk1)}")
    print(f"  出力長: {len(result1.audio)}")
    print(
        f"  バッファ保存: {state.sola_buffer is not None} (長さ: {len(state.sola_buffer) if state.sola_buffer is not None else 0})"
    )
    print(f"  SOLAオフセット: {result1.sola_offset}")

    # テストケース2: 短いチャンク（問題を引き起こす可能性）
    print(f"\n--- テストケース2: 短いチャンク (3000サンプル) ---")
    chunk2 = np.sin(2 * np.pi * 440 * np.arange(3000) / sr).astype(np.float32) * 0.5

    result2 = apply_sola_crossfade(chunk2, state, wokada_mode=True, context_samples=2400)
    print(f"  入力長: {len(chunk2)}")
    print(f"  出力長: {len(result2.audio)}")
    print(
        f"  バッファ保存: {state.sola_buffer is not None} (長さ: {len(state.sola_buffer) if state.sola_buffer is not None else 0})"
    )
    print(f"  SOLAオフセット: {result2.sola_offset}")

    if len(result2.audio) < len(chunk2) * 0.5:
        print(f"  ⚠️  出力が異常に短いです！バッファが破壊されている可能性")

    # テストケース3: もう一度短いチャンク
    print(f"\n--- テストケース3: 短いチャンク (3000サンプル) - 再度 ---")
    chunk3 = np.sin(2 * np.pi * 440 * np.arange(3000) / sr).astype(np.float32) * 0.5

    result3 = apply_sola_crossfade(chunk3, state, wokada_mode=True, context_samples=2400)
    print(f"  入力長: {len(chunk3)}")
    print(f"  出力長: {len(result3.audio)}")
    print(
        f"  バッファ保存: {state.sola_buffer is not None} (長さ: {len(state.sola_buffer) if state.sola_buffer is not None else 0})"
    )
    print(f"  SOLAオフセット: {result3.sola_offset}")

    if len(result3.audio) < len(chunk3) * 0.5:
        print(f"  ⚠️  出力が異常に短いです！バッファが破壊されている可能性")

    # テストケース4: 非常に短いチャンク（致命的）
    print(f"\n--- テストケース4: 非常に短いチャンク (1000サンプル) ---")
    chunk4 = np.sin(2 * np.pi * 440 * np.arange(1000) / sr).astype(np.float32) * 0.5

    result4 = apply_sola_crossfade(chunk4, state, wokada_mode=True, context_samples=2400)
    print(f"  入力長: {len(chunk4)}")
    print(f"  出力長: {len(result4.audio)}")
    print(
        f"  バッファ保存: {state.sola_buffer is not None} (長さ: {len(state.sola_buffer) if state.sola_buffer is not None else 0})"
    )
    print(f"  SOLAオフセット: {result4.sola_offset}")

    if len(result4.audio) < len(chunk4) * 0.5:
        print(f"  ⚠️  出力が異常に短いです！バッファが破壊されている可能性")


def detect_clicks(audio: np.ndarray, sr: int, threshold: float = 10.0) -> list:
    """クリック（急激な変化）を検出"""

    if len(audio) < 3:
        return []

    # 隣接サンプル間の差を計算
    diffs = np.abs(np.diff(audio))

    # 閾値を超える箇所を検出
    click_positions = np.where(diffs > threshold)[0]

    # クリックのエネルギーを計算
    clicks = []
    for pos in click_positions:
        if pos < len(audio) - 1:
            # クリック前後の差（ピークツーピーク）
            click_energy = abs(audio[pos + 1] - audio[pos])
            clicks.append(
                {
                    "position": pos,
                    "energy": float(click_energy),
                    "time_ms": pos * 1000 / sr,
                }
            )

    return clicks


def analyze_real_world_sola():
    """実際の推論出力でSOLAの問題を分析"""

    print(f"\n\n=== 実際の推論出力のSOLA分析 ===")

    # テストオーディオを生成
    sr = 48000
    audio = load_test_audio(sr)

    # モデルロード（テスト用に既存モデルを使用）
    try:
        from rcwx.config import RCWXConfig
        import os

        # モデルパスの検出
        config = RCWXConfig()
        model_dir = Path(config.model_dir)

        model_files = list(model_dir.glob("*.pth"))
        if not model_files:
            print(f"モデルが見つかりません: {model_dir}")
            return

        model_path = model_files[0]
        print(f"モデル: {model_path}")

        # パイプライン初期化
        pipeline = RVCPipeline(
            model_path=str(model_path),
            device="cpu",  # テストなのでCPUで
        )

        # RealtimeVoiceChangerでストリーミング処理
        rt_config = RealtimeConfig(
            mic_sample_rate=48000,
            output_sample_rate=48000,
            chunk_sec=0.35,
            context_sec=0.05,
            use_sola=True,
            voice_gate_mode="off",
            use_feature_cache=True,
        )

        changer = RealtimeVoiceChanger(pipeline, config=rt_config)
        changer._recalculate_buffers()
        changer._running = True

        # テスト用に大きなバッファ
        changer.output_buffer.set_max_latency(len(audio) * 2)

        # ストリーミング処理
        block_size = int(sr * 0.02)
        pos = 0
        chunks_processed = 0

        while pos < len(audio):
            block = audio[pos : min(pos + block_size, len(audio))]
            if len(block) < block_size:
                block = np.pad(block, (0, block_size - len(block)))

            changer.process_input_chunk(block)

            while changer.process_next_chunk():
                chunks_processed += 1

            changer.get_output_chunk(0)
            pos += block_size

        # 残りのチャンクを処理
        while changer.process_next_chunk():
            chunks_processed += 1
            changer.get_output_chunk(0)

        print(f"処理したチャンク数: {chunks_processed}")

        # 出力を取得
        outputs = []
        while changer.output_buffer.available > 0:
            out_block = changer.get_output_chunk(block_size)
            outputs.append(out_block)

        if outputs:
            output = np.concatenate(outputs)
            print(f"出力長: {len(output)} サンプル ({len(output) / sr:.2f}秒)")

            # クリックを検出
            threshold = np.std(audio) * 5.0  # 音声の標準偏差の5倍
            clicks = detect_clicks(output, sr, threshold)

            print(f"\nクリック検出結果:")
            print(f"  検出閾値: {threshold:.6f}")
            print(f"  クリック数: {len(clicks)}")

            if clicks:
                print(f"\n  クリック詳細 (最初の10個):")
                for i, click in enumerate(clicks[:10]):
                    print(
                        f"    {i + 1}. 位置: {click['time_ms']:.1f}ms, エネルギー: {click['energy']:.6f}"
                    )

                # クリックの位置分布を分析
                click_times = [c["time_ms"] for c in clicks]
                chunk_ms = rt_config.chunk_sec * 1000

                print(f"\n  クリックのチャンク境界分析:")
                print(f"    チャンク長: {chunk_ms}ms")

                boundary_clicks = 0
                for ct in click_times:
                    # チャンク境界の近く（±10ms）にあるクリックをカウント
                    chunk_num = int(ct / chunk_ms)
                    if (
                        abs(ct - chunk_num * chunk_ms) < 10
                        or abs(ct - (chunk_num + 1) * chunk_ms) < 10
                    ):
                        boundary_clicks += 1

                print(
                    f"    チャンク境界付近のクリック: {boundary_clicks}/{len(clicks)} ({boundary_clicks / len(clicks) * 100:.1f}%)"
                )

                if boundary_clicks / len(clicks) > 0.7:
                    print(f"    ⚠️  ほとんどのクリックがチャンク境界で発生！SOLAの問題")
            else:
                print(f"  ✓ クリックは検出されませんでした")

    except Exception as e:
        print(f"エラー: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    analyze_sola_behavior()
    analyze_real_world_sola()
