"""
チャンク連続性の詳細診断

ChunkBufferのコンテキスト重複が正しく機能しているか検証する。

問題: 第二チャンク以降のコンテキストが、前のチャンクのmain末尾と
      重複していない可能性がある。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from rcwx.audio.buffer import ChunkBuffer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def test_chunk_overlap():
    """
    チャンク間のオーバーラップを詳細に検証。
    """
    logger.info("=" * 70)
    logger.info("CHUNK OVERLAP ANALYSIS")
    logger.info("=" * 70)

    # パラメータ
    chunk_samples = 7200  # 150ms @ 48kHz
    context_samples = 4800  # 100ms @ 48kHz

    # 連番テスト信号（各サンプルがインデックスと一致）
    sample_rate = 48000
    duration = 1.0
    test_signal = np.arange(int(sample_rate * duration), dtype=np.float32)

    logger.info(f"Test signal: {len(test_signal)} samples (0 to {len(test_signal)-1})")
    logger.info(f"Chunk size: {chunk_samples} samples")
    logger.info(f"Context size: {context_samples} samples")

    # ChunkBuffer作成
    buffer = ChunkBuffer(
        chunk_samples=chunk_samples,
        crossfade_samples=0,
        context_samples=context_samples,
        lookahead_samples=0,
    )

    # 入力を追加
    buffer.add_input(test_signal)

    # チャンク抽出と分析
    chunks = []
    chunk_ranges = []

    while buffer.has_chunk():
        chunk = buffer.get_chunk()
        if chunk is not None:
            chunks.append(chunk)

            # このチャンクがカバーするオリジナル信号の範囲を特定
            # (reflection paddingがある場合は別途処理)
            if len(chunks) == 1:
                # 第一チャンク: [reflection | main]
                # mainは位置0から開始
                main_start = 0
                main_end = chunk_samples
                context_is_reflection = True
            else:
                # 第二チャンク以降: [context | main]
                # contextは入力バッファの先頭から取得される
                # 実際のサンプル値から範囲を特定
                context_start_val = int(chunk[0])
                main_start_val = int(chunk[context_samples])
                main_end_val = int(chunk[-1])

                chunk_ranges.append({
                    'chunk_idx': len(chunks) - 1,
                    'context': (context_start_val, context_start_val + context_samples - 1),
                    'main': (main_start_val, main_end_val),
                    'context_start_sample': context_start_val,
                    'main_start_sample': main_start_val,
                })

    logger.info(f"\nExtracted {len(chunks)} chunks")

    # 第一チャンクの分析
    logger.info("\n--- First Chunk ---")
    first_chunk = chunks[0]
    logger.info(f"Length: {len(first_chunk)}")
    logger.info(f"Context part (reflection): samples 0-{context_samples-1}")
    logger.info(f"  First sample value: {first_chunk[0]:.0f}")
    logger.info(f"  Last context sample value: {first_chunk[context_samples-1]:.0f}")
    logger.info(f"Main part: samples {context_samples}-{len(first_chunk)-1}")
    logger.info(f"  First main sample value: {first_chunk[context_samples]:.0f}")
    logger.info(f"  Last main sample value: {first_chunk[-1]:.0f}")

    # 第二チャンク以降の分析
    for i in range(1, len(chunks)):
        chunk = chunks[i]
        prev_chunk = chunks[i - 1]

        logger.info(f"\n--- Chunk {i} ---")
        logger.info(f"Length: {len(chunk)}")

        # コンテキスト部分
        context_first = chunk[0]
        context_last = chunk[context_samples - 1]
        logger.info(f"Context part: values {context_first:.0f} to {context_last:.0f}")

        # メイン部分
        main_first = chunk[context_samples]
        main_last = chunk[-1]
        logger.info(f"Main part: values {main_first:.0f} to {main_last:.0f}")

        # 前のチャンクのmain末尾と現在のcontext先頭の重複を確認
        prev_main_end = prev_chunk[-1]
        prev_main_overlap_start = prev_chunk[-(context_samples):]  # 前チャンクmain末尾のcontext分

        logger.info(f"\nOverlap analysis:")
        logger.info(f"  Previous chunk main ends at value: {prev_main_end:.0f}")
        logger.info(f"  Current chunk context starts at value: {context_first:.0f}")

        # 期待される重複
        # w-okadaスタイルでは、現在のcontextは前のmainの末尾context_samples分と一致すべき
        expected_context_start = prev_main_end - context_samples + 1

        if context_first == expected_context_start:
            logger.info(f"  ✓ Context correctly overlaps with previous main")
        else:
            logger.warning(f"  ✗ MISMATCH: Expected context start at {expected_context_start:.0f}, got {context_first:.0f}")
            logger.warning(f"    Gap: {context_first - expected_context_start:.0f} samples")

    # 入力バッファの進行を追跡
    logger.info("\n" + "=" * 70)
    logger.info("INPUT BUFFER PROGRESSION ANALYSIS")
    logger.info("=" * 70)

    buffer2 = ChunkBuffer(
        chunk_samples=chunk_samples,
        crossfade_samples=0,
        context_samples=context_samples,
        lookahead_samples=0,
    )
    buffer2.add_input(test_signal)

    logger.info(f"Initial buffer: {buffer2.buffered_samples} samples")

    chunk_idx = 0
    while buffer2.has_chunk():
        before = buffer2.buffered_samples
        chunk = buffer2.get_chunk()
        after = buffer2.buffered_samples

        if chunk is not None:
            consumed = before - after
            logger.info(f"Chunk {chunk_idx}: consumed {consumed} samples, remaining {after}")
            chunk_idx += 1

    # 正しい実装の説明
    logger.info("\n" + "=" * 70)
    logger.info("EXPECTED BEHAVIOR (w-okada style)")
    logger.info("=" * 70)
    logger.info("""
w-okada style chunking expects:
- First chunk: [reflection_padding | main] where main is samples 0 to chunk_samples-1
- Second chunk: [context | main] where:
  - context is samples (chunk_samples - context_samples) to (chunk_samples - 1)
  - main is samples chunk_samples to (2 * chunk_samples - 1)

This means the input buffer should advance by (chunk_samples) but the
NEXT chunk should start reading from (current_pos - context_samples) to
include overlap from the previous main.

Current implementation advances by chunk_samples, then reads from position 0
of the remaining buffer. This means:
- After first chunk: buffer position = chunk_samples
- Second chunk context reads from position chunk_samples (NOT chunk_samples - context_samples)

This creates a GAP of context_samples between consecutive chunks!
""")


def test_expected_vs_actual():
    """
    期待される動作と実際の動作を比較。
    """
    logger.info("\n" + "=" * 70)
    logger.info("EXPECTED vs ACTUAL SAMPLE RANGES")
    logger.info("=" * 70)

    chunk_samples = 7200
    context_samples = 4800

    # 期待される動作
    logger.info("\nEXPECTED (correct overlap):")
    logger.info("Chunk 0: main = [0, 7199]")
    logger.info("Chunk 1: context = [2400, 7199], main = [7200, 14399]")
    logger.info("Chunk 2: context = [9600, 14399], main = [14400, 21599]")
    logger.info("  -> Each context overlaps with previous main by context_samples")

    # 実際の動作（現在の実装）
    logger.info("\nACTUAL (current implementation - GAP):")
    logger.info("Chunk 0: main = [0, 7199] (buffer advances by 7200)")
    logger.info("Chunk 1: context = [7200, 11999], main = [12000, 19199]")
    logger.info("Chunk 2: context = [19200, 23999], main = [24000, 31199]")
    logger.info("  -> NO OVERLAP! Gap of context_samples between chunks")

    logger.info("\nThis causes:")
    logger.info("  1. Loss of audio continuity")
    logger.info("  2. Crossfade operates on non-overlapping regions")
    logger.info("  3. Artifacts at chunk boundaries")


if __name__ == "__main__":
    test_chunk_overlap()
    test_expected_vs_actual()
