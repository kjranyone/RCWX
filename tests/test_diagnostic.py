"""
Diagnostic Analysis Tool

各処理コンポーネントを個別にテストし、問題箇所を特定する。
発見した潜在的な問題点を検証するためのツール。

Usage:
    uv run python tests/test_diagnostic.py
    uv run python tests/test_diagnostic.py --component resampler
    uv run python tests/test_diagnostic.py --component feature_cache
    uv run python tests/test_diagnostic.py --component sola
    uv run python tests/test_diagnostic.py --component all
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# 1. Resampler Diagnostic
# ============================================================================

def test_stateful_resampler():
    """
    StatefulResampler vs batch resamplingの精度比較。

    問題点:
    - 第一チャンクでの過渡応答
    - チャンク境界での位相不連続性
    """
    from rcwx.audio.resample import resample, StatefulResampler

    logger.info("=" * 60)
    logger.info("TEST: StatefulResampler vs Batch Resampling")
    logger.info("=" * 60)

    # テスト信号: 440Hz + 880Hz の正弦波 (1秒)
    duration = 1.0
    orig_sr = 48000
    target_sr = 16000

    t = np.linspace(0, duration, int(orig_sr * duration), dtype=np.float32)
    test_signal = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)

    # バッチリサンプリング（参照）
    batch_result = resample(test_signal, orig_sr, target_sr, method="poly")
    logger.info(f"Batch result: {len(batch_result)} samples")

    # チャンク単位でStatefulResamplerを使用
    chunk_sizes = [0.02, 0.05, 0.1, 0.15]  # 20ms, 50ms, 100ms, 150ms

    results = {}

    for chunk_sec in chunk_sizes:
        resampler = StatefulResampler(orig_sr, target_sr)

        chunk_samples = int(orig_sr * chunk_sec)
        chunks = []
        pos = 0

        while pos < len(test_signal):
            chunk = test_signal[pos : pos + chunk_samples]
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

            resampled = resampler.resample_chunk(chunk)
            chunks.append(resampled)
            pos += chunk_samples

        streaming_result = np.concatenate(chunks)

        # バッチと比較
        min_len = min(len(batch_result), len(streaming_result))
        batch_trim = batch_result[:min_len]
        streaming_trim = streaming_result[:min_len]

        # メトリクス
        diff = batch_trim - streaming_trim
        mae = np.mean(np.abs(diff))
        rmse = np.sqrt(np.mean(diff**2))
        correlation = np.corrcoef(batch_trim, streaming_trim)[0, 1]

        # 境界アーティファクト
        expected_chunk_out = int(target_sr * chunk_sec)
        boundary_jumps = []
        for i in range(1, len(chunks)):
            if len(chunks[i-1]) > 0 and len(chunks[i]) > 0:
                jump = abs(float(chunks[i-1][-1]) - float(chunks[i][0]))
                boundary_jumps.append(jump)

        max_boundary_jump = max(boundary_jumps) if boundary_jumps else 0

        results[chunk_sec] = {
            'mae': mae,
            'rmse': rmse,
            'correlation': correlation,
            'max_boundary_jump': max_boundary_jump,
            'length_diff': len(batch_result) - len(streaming_result),
        }

        logger.info(
            f"chunk_sec={chunk_sec:.2f}s: corr={correlation:.6f}, "
            f"rmse={rmse:.6f}, max_jump={max_boundary_jump:.6f}"
        )

    # 判定
    best_chunk = max(results.keys(), key=lambda k: results[k]['correlation'])
    worst_chunk = min(results.keys(), key=lambda k: results[k]['correlation'])

    logger.info(f"\nBest chunk size: {best_chunk}s (corr={results[best_chunk]['correlation']:.6f})")
    logger.info(f"Worst chunk size: {worst_chunk}s (corr={results[worst_chunk]['correlation']:.6f})")

    if results[worst_chunk]['correlation'] < 0.99:
        logger.warning("ISSUE DETECTED: Resampler correlation < 0.99 for some chunk sizes")

    return results


# ============================================================================
# 2. Feature Cache Diagnostic
# ============================================================================

def test_feature_cache(model_path: Optional[str] = None):
    """
    特徴キャッシュの効果を検証。

    問題点:
    - キャッシュサイズがチャンクサイズを超える場合の品質劣化
    - キャッシュブレンディングによるぼやけ
    """
    from rcwx.audio.resample import resample
    from rcwx.config import RCWXConfig
    from rcwx.pipeline.inference import RVCPipeline

    logger.info("=" * 60)
    logger.info("TEST: Feature Cache Effect")
    logger.info("=" * 60)

    config = RCWXConfig.load()
    if not config.last_model_path:
        logger.error("No model configured. Run GUI first.")
        return None

    pipeline = RVCPipeline(config.last_model_path, device=config.device, use_compile=False)
    pipeline.load()

    # テスト音声生成（440Hz正弦波）
    duration = 0.5  # 500ms
    sr = 16000
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    test_audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    # チャンクサイズ別テスト
    chunk_secs = [0.10, 0.15, 0.20, 0.35]
    results = {}

    for chunk_sec in chunk_secs:
        chunk_samples = int(sr * chunk_sec)

        # キャッシュありでチャンク処理
        pipeline.clear_cache()
        outputs_with_cache = []
        pos = 0
        while pos < len(test_audio):
            chunk = test_audio[pos : pos + chunk_samples]
            if len(chunk) < chunk_samples // 2:  # 短すぎるチャンクはスキップ
                break

            output = pipeline.infer(
                chunk,
                input_sr=sr,
                pitch_shift=0,
                f0_method="fcpe",
                use_feature_cache=True,
                allow_short_input=True,
            )
            outputs_with_cache.append(output)
            pos += chunk_samples

        result_with_cache = np.concatenate(outputs_with_cache) if outputs_with_cache else np.array([])

        # キャッシュなしでチャンク処理
        pipeline.clear_cache()
        outputs_no_cache = []
        pos = 0
        while pos < len(test_audio):
            chunk = test_audio[pos : pos + chunk_samples]
            if len(chunk) < chunk_samples // 2:
                break

            output = pipeline.infer(
                chunk,
                input_sr=sr,
                pitch_shift=0,
                f0_method="fcpe",
                use_feature_cache=False,
                allow_short_input=True,
            )
            outputs_no_cache.append(output)
            pos += chunk_samples

        result_no_cache = np.concatenate(outputs_no_cache) if outputs_no_cache else np.array([])

        # 比較
        min_len = min(len(result_with_cache), len(result_no_cache))
        if min_len > 0:
            diff = result_with_cache[:min_len] - result_no_cache[:min_len]
            mae = np.mean(np.abs(diff))
            rmse = np.sqrt(np.mean(diff**2))

            # 相関
            if np.std(result_with_cache[:min_len]) > 1e-6 and np.std(result_no_cache[:min_len]) > 1e-6:
                correlation = np.corrcoef(result_with_cache[:min_len], result_no_cache[:min_len])[0, 1]
            else:
                correlation = 0.0
        else:
            mae, rmse, correlation = 0.0, 0.0, 0.0

        # キャッシュ/チャンク比率
        cache_frames = 10  # pipeline._feature_cache_framesのデフォルト
        chunk_frames = (chunk_samples / 320)  # HuBERT hop @ 16kHz
        cache_ratio = cache_frames / chunk_frames if chunk_frames > 0 else 0

        results[chunk_sec] = {
            'mae': mae,
            'rmse': rmse,
            'correlation': correlation,
            'cache_ratio': cache_ratio,
            'len_with_cache': len(result_with_cache),
            'len_no_cache': len(result_no_cache),
        }

        logger.info(
            f"chunk_sec={chunk_sec:.2f}s: cache_ratio={cache_ratio:.2f}, "
            f"corr(cache vs no_cache)={correlation:.4f}, rmse={rmse:.6f}"
        )

        if cache_ratio > 1.0:
            logger.warning(f"  WARNING: cache_ratio > 1.0 may cause temporal blur")

    return results


# ============================================================================
# 3. SOLA Crossfade Diagnostic
# ============================================================================

def test_sola_crossfade():
    """
    SOLA crossfadeの品質検証。

    問題点:
    - RMS matchingによる歪み
    - 位相不連続性
    - declick処理の効果
    """
    from rcwx.audio.crossfade import (
        SOLAState,
        apply_sola_crossfade,
        _match_rms,
        _declick_head,
    )

    logger.info("=" * 60)
    logger.info("TEST: SOLA Crossfade Quality")
    logger.info("=" * 60)

    # テスト信号
    sample_rate = 48000
    chunk_sec = 0.15
    crossfade_sec = 0.05
    context_sec = 0.10

    chunk_samples = int(sample_rate * chunk_sec)
    crossfade_samples = int(sample_rate * crossfade_sec)
    context_samples = int(sample_rate * context_sec)

    # 連続した正弦波（理想的なケース）
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    continuous_signal = 0.5 * np.sin(2 * np.pi * 440 * t)

    # チャンクに分割
    total_chunk = context_samples + chunk_samples
    chunks = []
    pos = 0
    while pos < len(continuous_signal):
        if pos == 0:
            # 最初のチャンク: context + main
            chunk = continuous_signal[:chunk_samples + context_samples]
        else:
            start = max(0, pos - context_samples)
            chunk = continuous_signal[start : pos + chunk_samples]

        if len(chunk) < context_samples + chunk_samples // 2:
            break

        chunks.append(chunk)
        pos += chunk_samples

    logger.info(f"Created {len(chunks)} chunks")

    # SOLA処理
    sola_state = SOLAState.create(crossfade_samples, sample_rate)

    outputs = []
    sola_offsets = []
    correlations = []

    for i, chunk in enumerate(chunks):
        ctx_samples = context_samples if i > 0 else 0

        result = apply_sola_crossfade(
            chunk,
            sola_state,
            wokada_mode=True,
            context_samples=ctx_samples,
        )

        outputs.append(result.audio)
        sola_offsets.append(result.sola_offset)
        correlations.append(result.correlation)

    # 連結
    sola_output = np.concatenate(outputs)

    # 元信号と比較
    min_len = min(len(continuous_signal), len(sola_output))
    original_trim = continuous_signal[:min_len]
    sola_trim = sola_output[:min_len]

    # 不連続性検出
    diff = np.abs(np.diff(sola_trim))
    discontinuities = np.where(diff > 0.1)[0]

    # メトリクス
    mae = np.mean(np.abs(original_trim - sola_trim))
    rmse = np.sqrt(np.mean((original_trim - sola_trim)**2))
    correlation = np.corrcoef(original_trim, sola_trim)[0, 1]

    logger.info(f"SOLA output length: {len(sola_output)}")
    logger.info(f"Original vs SOLA: corr={correlation:.6f}, rmse={rmse:.6f}")
    logger.info(f"Discontinuities (>0.1): {len(discontinuities)}")
    logger.info(f"SOLA offsets: {sola_offsets}")

    # RMS matching テスト
    logger.info("\n--- RMS Matching Test ---")
    test_target = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    test_ref = np.array([0.5, 0.6, 0.7], dtype=np.float32)

    matched = _match_rms(test_target, test_ref)
    target_rms = np.sqrt(np.mean(test_target**2))
    ref_rms = np.sqrt(np.mean(test_ref**2))
    matched_rms = np.sqrt(np.mean(matched**2))
    gain = matched_rms / target_rms

    logger.info(f"Target RMS: {target_rms:.4f}, Ref RMS: {ref_rms:.4f}, Matched RMS: {matched_rms:.4f}")
    logger.info(f"Applied gain: {gain:.4f}")

    if gain > 2.0:
        logger.warning("WARNING: RMS matching applied gain > 2.0, may cause distortion")

    # Declick テスト
    logger.info("\n--- Declick Test ---")
    prev_tail = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    curr_audio = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    declicked = _declick_head(curr_audio, prev_tail, samples=3)
    logger.info(f"Original: {curr_audio[:3]}")
    logger.info(f"Declicked: {declicked[:3]}")
    logger.info(f"Jump before: {abs(prev_tail[-1] - curr_audio[0]):.4f}")
    logger.info(f"Jump after: {abs(prev_tail[-1] - declicked[0]):.4f}")

    return {
        'correlation': correlation,
        'rmse': rmse,
        'discontinuities': len(discontinuities),
        'sola_offsets': sola_offsets,
    }


# ============================================================================
# 4. Chunk Boundary Diagnostic
# ============================================================================

def test_chunk_boundary():
    """
    チャンク境界での処理を検証。

    問題点:
    - 第一チャンクのreflection padding
    - コンテキストトリミングの精度
    """
    from rcwx.audio.buffer import ChunkBuffer

    logger.info("=" * 60)
    logger.info("TEST: Chunk Boundary Processing")
    logger.info("=" * 60)

    # パラメータ
    chunk_samples = 7200  # 150ms @ 48kHz
    context_samples = 4800  # 100ms @ 48kHz
    crossfade_samples = 2400  # 50ms @ 48kHz

    # テスト信号
    sample_rate = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    test_signal = 0.5 * np.sin(2 * np.pi * 440 * t)

    # ChunkBuffer作成
    buffer = ChunkBuffer(
        chunk_samples=chunk_samples,
        crossfade_samples=0,
        context_samples=context_samples,
        lookahead_samples=0,
    )

    # 入力を追加
    buffer.add_input(test_signal)

    # チャンク抽出
    chunks = []
    chunk_info = []

    while buffer.has_chunk():
        chunk = buffer.get_chunk()
        if chunk is not None:
            chunks.append(chunk)
            chunk_info.append({
                'len': len(chunk),
                'expected_len': chunk_samples + context_samples,
                'start_sample': chunk[0],
                'end_sample': chunk[-1],
            })

    logger.info(f"Extracted {len(chunks)} chunks")

    # 第一チャンクの検証
    if len(chunks) > 0:
        first_chunk = chunks[0]
        logger.info(f"\nFirst chunk analysis:")
        logger.info(f"  Length: {len(first_chunk)}")
        logger.info(f"  Expected: {chunk_samples + context_samples}")

        # Reflection paddingの検証
        if context_samples > 0:
            reflection_part = first_chunk[:context_samples]
            main_start = first_chunk[context_samples:context_samples + context_samples]

            # Reflectionは元の先頭のリバース
            expected_reflection = test_signal[:context_samples][::-1]

            reflection_match = np.allclose(reflection_part, expected_reflection, atol=1e-6)
            logger.info(f"  Reflection padding correct: {reflection_match}")

            if not reflection_match:
                logger.warning("  WARNING: Reflection padding mismatch")

    # チャンク間の連続性検証
    if len(chunks) > 1:
        logger.info(f"\nChunk continuity analysis:")
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            curr_chunk = chunks[i]

            # コンテキスト部分は前のチャンクの一部と重複するはず
            if context_samples > 0:
                prev_main_end = prev_chunk[-(chunk_samples):]  # 前チャンクのmain部分の末尾
                curr_context = curr_chunk[:context_samples]  # 現チャンクのcontext部分

                # 重複部分の確認
                overlap_len = min(len(prev_main_end), len(curr_context))
                if overlap_len > 0:
                    expected_overlap = prev_main_end[-overlap_len:]
                    actual_overlap = curr_context[:overlap_len]

                    overlap_match = np.allclose(expected_overlap, actual_overlap, atol=1e-6)
                    if not overlap_match:
                        logger.warning(f"  Chunk {i}: Context does not match previous main!")

    return {
        'num_chunks': len(chunks),
        'chunk_lengths': [len(c) for c in chunks],
    }


# ============================================================================
# 5. End-to-End Latency Analysis
# ============================================================================

def test_latency_accumulation():
    """
    レイテンシの蓄積を検証。

    問題点:
    - 出力サンプル数の累積誤差
    - バッファリングによる遅延
    """
    from rcwx.audio.resample import resample, StatefulResampler

    logger.info("=" * 60)
    logger.info("TEST: Latency Accumulation")
    logger.info("=" * 60)

    # シミュレーション設定
    mic_sr = 48000
    process_sr = 16000
    output_sr = 48000
    model_sr = 40000

    chunk_sec = 0.15
    duration = 5.0  # 5秒

    # 入力サンプル計算
    mic_samples = int(mic_sr * duration)
    chunk_samples_mic = int(mic_sr * chunk_sec)
    num_chunks = mic_samples // chunk_samples_mic

    logger.info(f"Input: {mic_samples} samples @ {mic_sr}Hz")
    logger.info(f"Chunk size: {chunk_samples_mic} samples ({chunk_sec}s)")
    logger.info(f"Number of chunks: {num_chunks}")

    # 各ステップでのサンプル数を追跡
    input_resampler = StatefulResampler(mic_sr, process_sr)
    output_resampler = StatefulResampler(model_sr, output_sr)

    total_input = 0
    total_after_input_resample = 0
    total_model_output = 0
    total_after_output_resample = 0

    for i in range(num_chunks):
        # Input chunk
        input_samples = chunk_samples_mic
        total_input += input_samples

        # Resample to 16kHz
        dummy_input = np.zeros(input_samples, dtype=np.float32)
        resampled_16k = input_resampler.resample_chunk(dummy_input)
        total_after_input_resample += len(resampled_16k)

        # Model output (40kHz, roughly 2.5x of 16kHz)
        expected_model_output = int(len(resampled_16k) * model_sr / process_sr)
        total_model_output += expected_model_output

        # Resample to 48kHz
        dummy_model_output = np.zeros(expected_model_output, dtype=np.float32)
        resampled_48k = output_resampler.resample_chunk(dummy_model_output)
        total_after_output_resample += len(resampled_48k)

    # 理論値との比較
    expected_final = int(mic_samples * (model_sr / process_sr) * (output_sr / model_sr))

    logger.info(f"\nSample counts through pipeline:")
    logger.info(f"  Input (mic): {total_input}")
    logger.info(f"  After input resample (16k): {total_after_input_resample}")
    logger.info(f"  Model output (40k): {total_model_output}")
    logger.info(f"  After output resample (48k): {total_after_output_resample}")
    logger.info(f"  Expected final: {expected_final}")
    logger.info(f"  Difference: {total_after_output_resample - expected_final} samples")

    # サンプル誤差の時間換算
    sample_error = abs(total_after_output_resample - expected_final)
    time_error_ms = (sample_error / output_sr) * 1000
    logger.info(f"  Time error: {time_error_ms:.2f}ms")

    if time_error_ms > 10:
        logger.warning("WARNING: Cumulative sample error > 10ms")

    return {
        'input_samples': total_input,
        'output_samples': total_after_output_resample,
        'expected_samples': expected_final,
        'sample_error': sample_error,
        'time_error_ms': time_error_ms,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Diagnostic Analysis Tool")
    parser.add_argument(
        "--component",
        choices=["resampler", "feature_cache", "sola", "chunk_boundary", "latency", "all"],
        default="all",
        help="Component to test",
    )

    args = parser.parse_args()

    output_dir = Path("test_output/diagnostic")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    if args.component in ["resampler", "all"]:
        results["resampler"] = test_stateful_resampler()

    if args.component in ["sola", "all"]:
        results["sola"] = test_sola_crossfade()

    if args.component in ["chunk_boundary", "all"]:
        results["chunk_boundary"] = test_chunk_boundary()

    if args.component in ["latency", "all"]:
        results["latency"] = test_latency_accumulation()

    if args.component in ["feature_cache", "all"]:
        results["feature_cache"] = test_feature_cache()

    # サマリー
    logger.info("\n" + "=" * 60)
    logger.info("DIAGNOSTIC SUMMARY")
    logger.info("=" * 60)

    issues_found = []

    if "resampler" in results and results["resampler"]:
        for chunk_sec, data in results["resampler"].items():
            if data["correlation"] < 0.99:
                issues_found.append(f"Resampler: Low correlation ({data['correlation']:.4f}) at chunk_sec={chunk_sec}")

    if "sola" in results and results["sola"]:
        if results["sola"]["discontinuities"] > 0:
            issues_found.append(f"SOLA: {results['sola']['discontinuities']} discontinuities detected")

    if "latency" in results and results["latency"]:
        if results["latency"]["time_error_ms"] > 10:
            issues_found.append(f"Latency: Cumulative error {results['latency']['time_error_ms']:.2f}ms")

    if issues_found:
        logger.warning("Issues found:")
        for issue in issues_found:
            logger.warning(f"  - {issue}")
    else:
        logger.info("No major issues detected")

    logger.info(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
