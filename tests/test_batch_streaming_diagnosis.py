"""Batch vs Streaming 不一致の根本原因を切り分ける診断テスト.

各処理段階を個別に比較し、どこで乖離が発生するかを特定する。
モデルが必要。
"""
from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_pipeline():
    from rcwx.config import RCWXConfig
    from rcwx.pipeline.inference import RVCPipeline

    config = RCWXConfig.load()
    if not config.last_model_path:
        raise RuntimeError("No model configured. Run GUI first.")
    pipeline = RVCPipeline(config.last_model_path, device=config.device, use_compile=False)
    pipeline.load()
    return pipeline


def load_test_audio(sr: int = 16000, duration_sec: float = 1.0) -> np.ndarray:
    """テスト音声をロード（なければサイン波を生成）."""
    test_file = Path(__file__).parent.parent / "sample_data" / "seki.wav"
    if not test_file.exists():
        test_file = Path(__file__).parent.parent / "sample_data" / "sustained_voice.wav"
    if test_file.exists():
        from scipy.io import wavfile
        from rcwx.audio.resample import resample

        file_sr, data = wavfile.read(str(test_file))
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        if data.ndim > 1:
            data = data.mean(axis=1)
        if file_sr != sr:
            data = resample(data, file_sr, sr)
        n = int(sr * duration_sec)
        return data[:n].astype(np.float32)
    else:
        # フォールバック: サイン波
        t = np.linspace(0, duration_sec, int(sr * duration_sec), dtype=np.float32)
        return 0.5 * np.sin(2 * np.pi * 220 * t)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return float(np.corrcoef(a[:n], b[:n])[0, 1])


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return float("inf")
    return float(np.sqrt(np.mean((a[:n] - b[:n]) ** 2)))


# =====================================================================
# 診断 1: HuBERT の入力長依存性（パディング量の違い）
# =====================================================================
def diagnose_1_hubert_padding_sensitivity(pipeline):
    """同一音声でパディング量を変えたときのHuBERT出力の差異を測定."""
    print("\n" + "=" * 70)
    print("診断 1: HuBERT パディング感度")
    print("  infer()は t_pad=800, infer_streaming()も t_pad=800")
    print("  → パディング量が同じなら HuBERT 出力も同一のはず")
    print("=" * 70)

    from rcwx.pipeline.inference import highpass_filter

    audio_16k = load_test_audio(16000, 0.5)
    hubert_hop = 320
    aligned = (len(audio_16k) // hubert_hop) * hubert_hop
    audio = audio_16k[:aligned]
    audio_hpf = highpass_filter(audio, sr=16000, cutoff=48)

    results = {}
    for label, t_pad in [("pad_800", 800), ("pad_320", 320), ("pad_0", 0)]:
        padded_len = t_pad + len(audio_hpf) + t_pad
        remainder = padded_len % hubert_hop
        extra = (hubert_hop - remainder) if remainder != 0 else 0
        padded = np.pad(audio_hpf, (t_pad, t_pad + extra), mode="reflect")

        audio_t = torch.from_numpy(padded).float().unsqueeze(0).to(pipeline.device)
        with torch.no_grad(), torch.autocast(device_type=pipeline.device, dtype=pipeline.dtype):
            feats = pipeline.hubert.extract(audio_t, output_layer=12, output_dim=768)

        # パディングフレームを除去して中身だけ比較
        pad_frames = t_pad // hubert_hop
        extra_frames = extra // hubert_hop
        if pad_frames > 0:
            content = feats[:, pad_frames:feats.shape[1] - pad_frames - extra_frames, :]
        else:
            content = feats[:, :feats.shape[1] - extra_frames, :]
        results[label] = content[0].cpu().float().numpy()
        print(f"  {label}: padded={len(padded)}, feats={feats.shape[1]}, content={content.shape[1]} frames")

    # 比較
    for a, b in [("pad_800", "pad_320"), ("pad_800", "pad_0")]:
        n = min(results[a].shape[0], results[b].shape[0])
        cos_sims = []
        for i in range(n):
            dot = np.dot(results[a][i], results[b][i])
            na = np.linalg.norm(results[a][i])
            nb = np.linalg.norm(results[b][i])
            cos_sims.append(dot / (na * nb + 1e-8))
        mean_cos = np.mean(cos_sims)
        min_cos = np.min(cos_sims)
        print(f"  {a} vs {b}: mean_cosine={mean_cos:.6f}, min_cosine={min_cos:.6f}, frames={n}")


# =====================================================================
# 診断 2: TextEncoder の受容野（ウィンドウアテンション）
# =====================================================================
def diagnose_2_text_encoder_context(pipeline):
    """同一特徴量の部分列と全体でシンセサイザー出力がどう変わるかを測定.

    TextEncoder は window_size=10 のアテンションを使っている。
    特徴量の前後に異なるコンテキストがあると出力が変わるはず。
    """
    print("\n" + "=" * 70)
    print("診断 2: TextEncoder ウィンドウアテンション")
    print("  シンセサイザーに同一特徴量を渡すが、前後コンテキスト長を変える")
    print("  → コンテキスト依存がなければ同一出力になるはず")
    print("=" * 70)

    from rcwx.pipeline.inference import highpass_filter

    audio_16k = load_test_audio(16000, 2.0)
    hubert_hop = 320
    aligned = (len(audio_16k) // hubert_hop) * hubert_hop
    audio = audio_16k[:aligned]
    audio_hpf = highpass_filter(audio, sr=16000, cutoff=48)

    # 全体で HuBERT
    t_pad = 800
    padded_len = t_pad + len(audio_hpf) + t_pad
    remainder = padded_len % hubert_hop
    extra = (hubert_hop - remainder) if remainder != 0 else 0
    padded = np.pad(audio_hpf, (t_pad, t_pad + extra), mode="reflect")

    audio_t = torch.from_numpy(padded).float().unsqueeze(0).to(pipeline.device)
    with torch.no_grad(), torch.autocast(device_type=pipeline.device, dtype=pipeline.dtype):
        all_feats = pipeline.hubert.extract(audio_t, output_layer=12, output_dim=768)

    pad_frames = t_pad // hubert_hop
    extra_frames = extra // hubert_hop
    content_feats = all_feats[:, pad_frames:all_feats.shape[1] - pad_frames - extra_frames, :]
    total_frames = content_feats.shape[1]
    print(f"  全体: {total_frames} content frames")

    # テスト区間: 中央 20 フレーム
    center = total_frames // 2
    target_start = center - 10
    target_end = center + 10
    target_feats = content_feats[:, target_start:target_end, :]
    print(f"  ターゲット: frames [{target_start}:{target_end}] (20 frames)")

    # コンテキスト量を変えてシンセサイザーを通す
    model_sr = pipeline.sample_rate
    samples_per_frame = model_sr // 100  # 100fps features

    outputs = {}
    for label, ctx_frames in [("ctx_0", 0), ("ctx_5", 5), ("ctx_10", 10), ("ctx_20", 20), ("ctx_all", target_start)]:
        start = max(0, target_start - ctx_frames)
        end = min(total_frames, target_end + ctx_frames)
        feats_slice = content_feats[:, start:end, :]

        # 2x interpolation
        feats_interp = torch.nn.functional.interpolate(
            feats_slice.permute(0, 2, 1), scale_factor=2, mode="linear", align_corners=False
        ).permute(0, 2, 1)

        feat_len = torch.tensor([feats_interp.shape[1]], dtype=torch.long, device=pipeline.device)
        pitch = torch.ones(1, feats_interp.shape[1], dtype=torch.long, device=pipeline.device)
        pitchf = torch.zeros(1, feats_interp.shape[1], dtype=pipeline.dtype, device=pipeline.device)

        with torch.no_grad(), torch.autocast(device_type=pipeline.device, dtype=pipeline.dtype):
            out = pipeline.synthesizer.infer(feats_interp, feat_len, pitch=pitch, pitchf=pitchf)

        out_np = out.cpu().float().numpy().squeeze()

        # ターゲット区間だけ切り出す（前後コンテキスト分をトリム）
        pre_ctx = target_start - start
        pre_samples = pre_ctx * 2 * samples_per_frame  # 2x interpolation
        target_samples = 20 * 2 * samples_per_frame
        out_target = out_np[pre_samples:pre_samples + target_samples]

        outputs[label] = out_target
        print(f"  {label}: feats={feats_interp.shape[1]}, out={len(out_np)}, target_slice={len(out_target)}")

    # ctx_all を基準に比較
    ref = outputs["ctx_all"]
    for label in ["ctx_0", "ctx_5", "ctx_10", "ctx_20"]:
        c = corr(ref, outputs[label])
        r = rmse(ref, outputs[label])
        print(f"  ctx_all vs {label}: corr={c:.6f}, rmse={r:.6f}")


# =====================================================================
# 診断 3: infer() vs infer_streaming() 同一音声・全体（チャンクなし）
# =====================================================================
def diagnose_3_single_chunk_comparison(pipeline):
    """全体を一括で処理した場合の infer() vs infer_streaming() 比較.

    チャンク分割なしで全体を渡す → SOLA, リサンプリング, チャンク境界の影響を排除。
    """
    print("\n" + "=" * 70)
    print("診断 3: 単一チャンク比較 (infer vs infer_streaming)")
    print("  同一音声を全体として渡し、パイプラインの内部処理の違いだけを測定")
    print("=" * 70)

    audio_16k = load_test_audio(16000, 1.0)
    hubert_hop = 320
    aligned = (len(audio_16k) // hubert_hop) * hubert_hop
    audio = audio_16k[:aligned]
    model_sr = pipeline.sample_rate

    # infer() — バッチ
    pipeline.clear_cache()
    batch_out = pipeline.infer(
        audio,
        input_sr=16000,
        pitch_shift=0,
        f0_method="fcpe",
        use_feature_cache=False,
        voice_gate_mode="off",
    )

    # infer_streaming() — overlap=0 の全体
    pipeline.clear_cache()
    stream_out = pipeline.infer_streaming(
        audio_16k=audio,
        overlap_samples=0,
        pitch_shift=0,
        f0_method="fcpe",
        voice_gate_mode="off",
    )

    c = corr(batch_out, stream_out)
    r = rmse(batch_out, stream_out)
    print(f"  batch: {len(batch_out)} samples, rms={np.sqrt(np.mean(batch_out**2)):.6f}")
    print(f"  stream: {len(stream_out)} samples, rms={np.sqrt(np.mean(stream_out**2)):.6f}")
    print(f"  corr={c:.6f}, rmse={r:.6f}")
    print(f"  len diff: {len(batch_out) - len(stream_out)} samples")

    # 差異のある箇所を分析
    n = min(len(batch_out), len(stream_out))
    diff = np.abs(batch_out[:n] - stream_out[:n])
    # 差異が大きい区間を特定
    frame_size = model_sr // 20  # 50ms frames
    n_frames = n // frame_size
    frame_diffs = []
    for i in range(n_frames):
        seg_diff = diff[i * frame_size:(i + 1) * frame_size]
        frame_diffs.append(np.mean(seg_diff))
    frame_diffs = np.array(frame_diffs)
    worst_frames = np.argsort(frame_diffs)[-5:][::-1]
    print(f"  差異が大きいフレーム (50ms単位): {worst_frames.tolist()}")
    print(f"  差異: {[f'{frame_diffs[f]:.6f}' for f in worst_frames]}")
    print(f"  平均差異: {np.mean(frame_diffs):.6f}, 最大差異: {np.max(frame_diffs):.6f}")

    return batch_out, stream_out


# =====================================================================
# 診断 4: 出力トリミングの差異
# =====================================================================
def diagnose_4_trimming_difference(pipeline):
    """infer() と infer_streaming() の出力トリミング計算の差異を検証."""
    print("\n" + "=" * 70)
    print("診断 4: 出力トリミング差異")
    print("  同一パディング済み音声→同一特徴量→同一シンセサイザー出力に対し")
    print("  それぞれのトリミングロジックを適用して比較")
    print("=" * 70)

    from rcwx.pipeline.inference import highpass_filter

    audio_16k = load_test_audio(16000, 0.5)
    hubert_hop = 320
    aligned = (len(audio_16k) // hubert_hop) * hubert_hop
    audio = audio_16k[:aligned]
    model_sr = pipeline.sample_rate
    input_length = len(audio)

    audio_hpf = highpass_filter(audio, sr=16000, cutoff=48)

    # 共通パディング
    t_pad = 800
    padded_len = t_pad + len(audio_hpf) + t_pad
    remainder = padded_len % hubert_hop
    extra_pad = (hubert_hop - remainder) if remainder != 0 else 0
    padded = np.pad(audio_hpf, (t_pad, t_pad + extra_pad), mode="reflect")
    print(f"  input={input_length}, t_pad={t_pad}, extra_pad={extra_pad}, padded={len(padded)}")

    # HuBERT + synth
    audio_t = torch.from_numpy(padded).float().unsqueeze(0).to(pipeline.device)
    with torch.no_grad(), torch.autocast(device_type=pipeline.device, dtype=pipeline.dtype):
        feats = pipeline.hubert.extract(audio_t, output_layer=12, output_dim=768)
    feats_interp = torch.nn.functional.interpolate(
        feats.permute(0, 2, 1), scale_factor=2, mode="linear", align_corners=False
    ).permute(0, 2, 1)
    feat_len = torch.tensor([feats_interp.shape[1]], dtype=torch.long, device=pipeline.device)
    pitch = torch.ones(1, feats_interp.shape[1], dtype=torch.long, device=pipeline.device)
    pitchf = torch.zeros(1, feats_interp.shape[1], dtype=pipeline.dtype, device=pipeline.device)
    with torch.no_grad(), torch.autocast(device_type=pipeline.device, dtype=pipeline.dtype):
        synth_out = pipeline.synthesizer.infer(feats_interp, feat_len, pitch=pitch, pitchf=pitchf)
    synth_np = synth_out.cpu().float().numpy().squeeze()
    print(f"  synth output: {len(synth_np)} samples")

    # infer() スタイルのトリミング
    t_pad_tgt = int(t_pad * model_sr / 16000)
    extra_pad_tgt = int(extra_pad * model_sr / 16000)
    infer_trim_start = t_pad_tgt
    infer_trim_end = t_pad_tgt + extra_pad_tgt
    if infer_trim_end > 0:
        trimmed_infer = synth_np[infer_trim_start:-infer_trim_end]
    else:
        trimmed_infer = synth_np[infer_trim_start:]

    # infer_streaming() スタイルのトリミング（overlap=0 の場合）
    stream_trim_left = t_pad_tgt  # + overlap_tgt (0 here)
    stream_trim_right = t_pad_tgt  # ← extra_pad_tgt がない!
    if stream_trim_right > 0:
        trimmed_stream = synth_np[stream_trim_left:-stream_trim_right]
    else:
        trimmed_stream = synth_np[stream_trim_left:]

    print(f"  infer  trim: [{infer_trim_start}:-{infer_trim_end}] → {len(trimmed_infer)} samples")
    print(f"  stream trim: [{stream_trim_left}:-{stream_trim_right}] → {len(trimmed_stream)} samples")
    print(f"  長さ差: {len(trimmed_stream) - len(trimmed_infer)} samples (= extra_pad_tgt={extra_pad_tgt})")

    if extra_pad_tgt != 0:
        print(f"  ★ BUG: infer_streaming() は extra_pad をトリムしていない!")
        print(f"    extra_pad={extra_pad} samples @16kHz → {extra_pad_tgt} samples @{model_sr}Hz")
        print(f"    streaming の末尾 {extra_pad_tgt} サンプルは反射パディングの残り")
    else:
        print(f"  OK: extra_pad=0 のためトリミングは一致")

    # 共通部分の比較
    n = min(len(trimmed_infer), len(trimmed_stream))
    if n > 0:
        c = corr(trimmed_infer, trimmed_stream)
        max_diff = np.max(np.abs(trimmed_infer[:n] - trimmed_stream[:n]))
        print(f"  共通部分 ({n} samples): corr={c:.6f}, max_diff={max_diff:.8f}")


# =====================================================================
# 診断 5: チャンク分割時の F0 連続性
# =====================================================================
def diagnose_5_f0_continuity(pipeline):
    """F0 を全体/チャンク単位で抽出し、境界の不連続性を測定."""
    print("\n" + "=" * 70)
    print("診断 5: F0 連続性 (バッチ vs チャンク)")
    print("  同一音声の F0 を全体で抽出 vs チャンク単位で抽出")
    print("=" * 70)

    from rcwx.pipeline.inference import highpass_filter

    audio_16k = load_test_audio(16000, 2.0)
    hubert_hop = 320
    aligned = (len(audio_16k) // hubert_hop) * hubert_hop
    audio = audio_16k[:aligned]
    audio_hpf = highpass_filter(audio, sr=16000, cutoff=48)

    # 全体で F0 抽出
    t_pad = 800
    padded_len = t_pad + len(audio_hpf) + t_pad
    remainder = padded_len % hubert_hop
    extra = (hubert_hop - remainder) if remainder != 0 else 0
    padded = np.pad(audio_hpf, (t_pad, t_pad + extra), mode="reflect")

    audio_t = torch.from_numpy(padded).float().unsqueeze(0).to(pipeline.device)
    with torch.no_grad(), torch.autocast(device_type=pipeline.device, dtype=pipeline.dtype):
        f0_batch = pipeline.fcpe.infer(audio_t, threshold=0.006)
    f0_batch_np = f0_batch[0].cpu().float().numpy()
    # パディングフレームを除去 (F0 hop=160, so pad_frames = t_pad//160)
    f0_pad_frames = t_pad // 160
    f0_extra_frames = extra // 160
    f0_batch_content = f0_batch_np[f0_pad_frames:len(f0_batch_np) - f0_pad_frames - f0_extra_frames]
    print(f"  バッチ F0: {len(f0_batch_content)} frames")

    # チャンク単位で F0 抽出
    chunk_sec = 0.15
    overlap_sec = 0.10
    chunk_samples = (int(16000 * chunk_sec) // hubert_hop) * hubert_hop
    overlap_samples = (int(16000 * overlap_sec) // hubert_hop) * hubert_hop

    f0_chunks = []
    pos = 0
    chunk_idx = 0
    while pos < len(audio_hpf):
        new_hop = audio_hpf[pos:pos + chunk_samples]
        if len(new_hop) < hubert_hop:
            break
        aligned_hop = (len(new_hop) // hubert_hop) * hubert_hop
        new_hop = new_hop[:aligned_hop]
        if len(new_hop) == 0:
            break

        # overlap 組み立て
        if chunk_idx == 0:
            if overlap_samples > 0 and len(new_hop) > overlap_samples:
                reflection = new_hop[:overlap_samples][::-1].copy()
                chunk = np.concatenate([reflection, new_hop])
                ovl = overlap_samples
            else:
                chunk = new_hop
                ovl = 0
        else:
            ovl_start = max(0, pos - overlap_samples)
            overlap_audio = audio_hpf[ovl_start:pos]
            aligned_ovl = (len(overlap_audio) // hubert_hop) * hubert_hop
            if aligned_ovl > 0:
                overlap_audio = overlap_audio[-aligned_ovl:]
                chunk = np.concatenate([overlap_audio, new_hop])
                ovl = len(overlap_audio)
            else:
                chunk = new_hop
                ovl = 0

        # パディング
        chunk_padded_len = t_pad + len(chunk) + t_pad
        r = chunk_padded_len % hubert_hop
        ep = (hubert_hop - r) if r != 0 else 0
        chunk_padded = np.pad(chunk, (t_pad, t_pad + ep), mode="reflect")

        chunk_t = torch.from_numpy(chunk_padded).float().unsqueeze(0).to(pipeline.device)
        with torch.no_grad(), torch.autocast(device_type=pipeline.device, dtype=pipeline.dtype):
            f0_chunk = pipeline.fcpe.infer(chunk_t, threshold=0.006)
        f0_chunk_np = f0_chunk[0].cpu().float().numpy()

        # トリム: パディング + overlap
        f0_pad = t_pad // 160
        f0_ep = ep // 160
        f0_ovl = ovl // 160
        f0_trimmed = f0_chunk_np[f0_pad + f0_ovl:len(f0_chunk_np) - f0_pad - f0_ep]
        f0_chunks.append(f0_trimmed)

        pos += chunk_samples
        chunk_idx += 1

    f0_chunked = np.concatenate(f0_chunks)
    print(f"  チャンク F0: {len(f0_chunked)} frames ({chunk_idx} chunks)")

    # 比較
    n = min(len(f0_batch_content), len(f0_chunked))
    voiced_batch = f0_batch_content[:n] > 0
    voiced_chunk = f0_chunked[:n] > 0
    both_voiced = voiced_batch & voiced_chunk

    if both_voiced.sum() > 0:
        diff = np.abs(f0_batch_content[:n][both_voiced] - f0_chunked[:n][both_voiced])
        mean_diff = np.mean(diff)
        max_diff = np.max(diff)
        p95_diff = np.percentile(diff, 95)
        c = np.corrcoef(f0_batch_content[:n][both_voiced], f0_chunked[:n][both_voiced])[0, 1]
        print(f"  有声区間 F0 比較 ({both_voiced.sum()}/{n} frames):")
        print(f"    corr={c:.6f}, mean_diff={mean_diff:.2f}Hz, p95={p95_diff:.2f}Hz, max={max_diff:.2f}Hz")
    else:
        print(f"  有声フレームなし")

    # 境界での不連続性
    f0_chunk_diffs = []
    boundary_pos = 0
    for i, fc in enumerate(f0_chunks[:-1]):
        boundary_pos += len(fc)
        if boundary_pos < len(f0_chunked) - 1:
            left = f0_chunked[boundary_pos - 1]
            right = f0_chunked[boundary_pos]
            if left > 0 and right > 0:
                jump = abs(right - left)
                f0_chunk_diffs.append((boundary_pos, jump))
    if f0_chunk_diffs:
        jumps = [d[1] for d in f0_chunk_diffs]
        print(f"  チャンク境界 F0 ジャンプ: mean={np.mean(jumps):.2f}Hz, max={np.max(jumps):.2f}Hz")
    else:
        print(f"  チャンク境界でのジャンプなし（無声区間のみ）")


# =====================================================================
# 診断 6: チャンク分割ストリーミングの累積乖離
# =====================================================================
def diagnose_6_chunked_streaming(pipeline):
    """チャンク分割してinfer_streaming()を呼び、バッチと比較."""
    print("\n" + "=" * 70)
    print("診断 6: チャンク分割ストリーミング vs バッチ")
    print("  実際の使用パターンを再現")
    print("=" * 70)

    audio_16k = load_test_audio(16000, 2.0)
    hubert_hop = 320
    aligned = (len(audio_16k) // hubert_hop) * hubert_hop
    audio = audio_16k[:aligned]
    model_sr = pipeline.sample_rate

    # バッチ
    pipeline.clear_cache()
    batch_out = pipeline.infer(
        audio, input_sr=16000, pitch_shift=0, f0_method="fcpe",
        use_feature_cache=False, voice_gate_mode="off",
    )
    print(f"  バッチ: {len(batch_out)} samples")

    # チャンク分割ストリーミング
    pipeline.clear_cache()
    chunk_sec = 0.15
    overlap_sec = 0.10
    chunk_samples = (int(16000 * chunk_sec) // hubert_hop) * hubert_hop
    overlap_samples = (int(16000 * overlap_sec) // hubert_hop) * hubert_hop

    chunks_output = []
    pos = 0
    chunk_idx = 0
    while pos < len(audio):
        new_hop = audio[pos:pos + chunk_samples]
        if len(new_hop) < hubert_hop:
            break
        aligned_hop = (len(new_hop) // hubert_hop) * hubert_hop
        new_hop = new_hop[:aligned_hop]
        if len(new_hop) == 0:
            break

        if chunk_idx == 0:
            if overlap_samples > 0 and len(new_hop) > overlap_samples:
                reflection = new_hop[:overlap_samples][::-1].copy()
                chunk_16k = np.concatenate([reflection, new_hop])
                ovl = overlap_samples
            else:
                chunk_16k = new_hop
                ovl = 0
        else:
            ovl_start = max(0, pos - overlap_samples)
            overlap_audio = audio[ovl_start:pos]
            aligned_ovl = (len(overlap_audio) // hubert_hop) * hubert_hop
            if aligned_ovl > 0:
                overlap_audio = overlap_audio[-aligned_ovl:]
                chunk_16k = np.concatenate([overlap_audio, new_hop])
                ovl = len(overlap_audio)
            else:
                chunk_16k = new_hop
                ovl = 0

        out = pipeline.infer_streaming(
            audio_16k=chunk_16k,
            overlap_samples=ovl,
            pitch_shift=0,
            f0_method="fcpe",
            voice_gate_mode="off",
        )
        chunks_output.append(out)
        pos += chunk_samples
        chunk_idx += 1

    streamed = np.concatenate(chunks_output)
    print(f"  ストリーミング: {len(streamed)} samples ({chunk_idx} chunks)")

    # 全体比較
    c = corr(batch_out, streamed)
    r = rmse(batch_out, streamed)
    print(f"  全体: corr={c:.6f}, rmse={r:.6f}")

    # チャンクごとの比較
    n = min(len(batch_out), len(streamed))
    frame_size = int(model_sr * chunk_sec)
    n_frames = n // frame_size
    print(f"  チャンクごとの相関:")
    for i in range(min(n_frames, 10)):
        seg_batch = batch_out[i * frame_size:(i + 1) * frame_size]
        seg_stream = streamed[i * frame_size:(i + 1) * frame_size]
        seg_len = min(len(seg_batch), len(seg_stream))
        if seg_len > 100:
            seg_c = np.corrcoef(seg_batch[:seg_len], seg_stream[:seg_len])[0, 1]
            print(f"    chunk {i}: corr={seg_c:.4f}")


# =====================================================================
# 診断 7: lowpass_f0 のエッジ効果
# =====================================================================
def diagnose_7_f0_filter_edge_effects(pipeline):
    """短い F0 系列に lowpass_f0 を適用した時のエッジ効果を測定."""
    print("\n" + "=" * 70)
    print("診断 7: F0 フィルタのエッジ効果")
    print("  同一 F0 系列を全体/チャンクで lowpass_f0 した結果を比較")
    print("=" * 70)

    from rcwx.pipeline.inference import lowpass_f0, smooth_f0_spikes

    # 100fps の F0 系列を作成 (2 秒 = 200 フレーム)
    n_frames = 200
    f0 = torch.zeros(1, n_frames)
    # 有声区間: 220Hz にゆるやかなビブラート
    for i in range(n_frames):
        f0[0, i] = 220.0 + 5.0 * np.sin(2 * np.pi * 5.0 * i / 100)

    # 全体で処理
    f0_full = lowpass_f0(smooth_f0_spikes(f0.clone(), window=3), cutoff_hz=8.0)

    # チャンク単位で処理 (30 フレーム = 300ms)
    chunk_frames = 30
    f0_chunks = []
    for start in range(0, n_frames, chunk_frames):
        end = min(start + chunk_frames, n_frames)
        chunk = f0[:, start:end].clone()
        chunk = smooth_f0_spikes(chunk, window=3)
        chunk = lowpass_f0(chunk, cutoff_hz=8.0)
        f0_chunks.append(chunk)

    f0_chunked = torch.cat(f0_chunks, dim=1)

    # 比較
    f0_full_np = f0_full[0].numpy()
    f0_chunked_np = f0_chunked[0].numpy()
    n = min(len(f0_full_np), len(f0_chunked_np))
    diff = np.abs(f0_full_np[:n] - f0_chunked_np[:n])
    print(f"  全体: {len(f0_full_np)} frames, チャンク: {len(f0_chunked_np)} frames")
    print(f"  mean_diff={np.mean(diff):.4f}Hz, max_diff={np.max(diff):.4f}Hz, p95={np.percentile(diff, 95):.4f}Hz")

    # 境界付近の差異
    for i in range(1, n_frames // chunk_frames):
        boundary = i * chunk_frames
        if boundary < n:
            region = diff[max(0, boundary - 3):boundary + 3]
            print(f"  境界 {boundary}: diff={region.tolist()}")


# =====================================================================
# メイン
# =====================================================================
def main():
    print("=" * 70)
    print("バッチ vs ストリーミング 不一致診断テスト")
    print("=" * 70)

    pipeline = load_pipeline()
    model_sr = pipeline.sample_rate
    print(f"Model: {pipeline.model_path.name}, SR={model_sr}")

    diagnose_1_hubert_padding_sensitivity(pipeline)
    diagnose_2_text_encoder_context(pipeline)
    diagnose_3_single_chunk_comparison(pipeline)
    diagnose_4_trimming_difference(pipeline)
    diagnose_5_f0_continuity(pipeline)
    diagnose_6_chunked_streaming(pipeline)
    diagnose_7_f0_filter_edge_effects(pipeline)

    print("\n" + "=" * 70)
    print("診断完了")
    print("=" * 70)


if __name__ == "__main__":
    main()
