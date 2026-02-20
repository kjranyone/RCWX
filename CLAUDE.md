# RCWX Development Guide

## Project Overview

**RCWX** = RVC Real-time Voice Changer on Intel Arc (XPU)

RVC v2モデルを使ったリアルタイムボイスチェンジャー。現行のリアルタイム処理は
`RealtimeVoiceChangerUnified` を中心とした単一パス実装です。

## Quick Start

```powershell
# 1) 依存関係インストール（PyTorch XPU版を使用）
uv sync

# 2) XPU確認
uv run python -c "import torch; print(torch.__version__, torch.xpu.is_available())"

# 3) 必須モデルをダウンロード（HuBERT / RMVPE）
uv run rcwx download

# 4) (オプション) ML Denoiser利用可否を確認
uv run python -c "from rcwx.audio.denoise import is_ml_denoiser_available; print(is_ml_denoiser_available())"

# 5) (推奨) フィードバック診断
uv run rcwx diagnose

# 6) GUI起動
uv run rcwx
```

## Current Architecture

```text
AudioInput (mic rate)
  -> Input accumulator (hop単位)
  -> [Input Queue]
  -> Inference Thread
       1) input gain
       2) StatefulResampler (mic -> 16k)
       3) optional denoise (auto/ml/spectral)
       4) [overlap | new_hop] を組み立て
       5) RVCPipeline.infer_streaming()
       6) StatefulResampler (model_sr -> output_sr)
       7) soft clip + SOLA crossfade
       8) feedback detection
       9) [Output Queue]
  -> RingOutputBuffer
  -> AudioOutput (48kHz)
```

実装上の要点:

- リアルタイム経路は `pipeline/realtime_unified.py` のみを使用
- チャンク境界連続性は **audio-level overlap** + `infer_streaming()` + SOLA で処理
- 旧互換の `context_sec` / `lookahead_sec` / `set_context()` / `set_lookahead()` は廃止
- 旧GUIトグルの `use_feature_cache` は廃止
- 過負荷時は一時的に `f0_method="none"` と `index_rate=0.0` に自動退避

## Directory Structure

```text
rcwx/
├── cli.py
├── config.py
├── device.py
├── diagnose.py
├── downloader.py
├── audio/
│   ├── input.py           # AudioInput (マイク入力)
│   ├── output.py          # AudioOutput
│   ├── buffer.py          # RingOutputBuffer
│   ├── resample.py        # StatefulResampler
│   ├── sola.py            # simple SOLA
│   ├── denoise.py         # auto/ml/spectral
│   ├── wav_input.py       # WAVファイルループ入力
│   └── stream_base.py     # ストリーム基底
├── models/
│   ├── hubert_loader.py
│   ├── rmvpe.py
│   ├── fcpe.py
│   ├── swiftf0.py         # SwiftF0 (ONNX/CPU)
│   ├── synthesizer.py
│   └── infer_pack/
├── pipeline/
│   ├── inference.py       # infer / infer_streaming
│   └── realtime_unified.py
└── gui/
    ├── app.py
    ├── realtime_controller.py
    ├── model_loader.py    # 非同期モデルロード
    ├── audio_test.py      # オーディオテスト
    ├── file_converter.py  # ファイル変換
    └── widgets/
        ├── audio_settings.py
        ├── latency_settings.py
        ├── latency_monitor.py
        ├── model_selector.py
        └── pitch_control.py
```

## Configuration (Current)

### `AudioConfig` (`rcwx/config.py`)

| Key                       |  Default | Notes                   |
| ------------------------- | -------: | ----------------------- |
| `sample_rate`             |  `16000` | 内部処理入力レート      |
| `output_sample_rate`      |  `48000` | 出力レート              |
| `chunk_sec`               |    `0.3` | 保存設定上のチャンク長  |
| `prebuffer_chunks`        |      `1` | 出力プリバッファ        |
| `buffer_margin`           |    `0.5` | バッファ余裕            |
| `input_gain_db`           |    `0.0` | 入力ゲイン              |
| `input_channel_selection` |   `auto` | left/right/average/auto |
| `input_hostapi_filter`    | `WASAPI` | Windows向け             |
| `output_hostapi_filter`   | `WASAPI` | Windows向け             |

### `InferenceConfig` (`rcwx/config.py`)

| Key                           |  Default | Notes                           |
| ----------------------------- | -------: | ------------------------------- |
| `f0_method`                   |  `rmvpe` | `rmvpe` / `fcpe` / `swiftf0`    |
| `use_f0`                      |   `true` | F0有効化                        |
| `use_index`                   |   `true` | FAISS有効化                     |
| `index_ratio`                 |   `0.15` | FAISS混合率                     |
| `index_k`                     |      `4` | 近傍数                          |
| `use_compile`                 |  `false` | 既定OFF                         |
| `resample_method`             | `linear` | `linear` / `poly`               |
| `use_parallel_extraction`     |   `true` | HuBERT+F0並列                   |
| `voice_gate_mode`             |    `off` | off/strict/expand/energy        |
| `energy_threshold`            |    `0.2` | energyモード閾値                |
| `overlap_sec`                 |   `0.20` | 音声オーバーラップ              |
| `crossfade_sec`               |   `0.08` | SOLAクロスフェード長            |
| `use_sola`                    |   `true` | SOLA有効化                      |
| `sola_search_ms`              |   `10.0` | SOLA探索窓                      |
| `hubert_context_sec`          |    `1.0` | HuBERTコンテキスト窓 (秒)       |
| `pre_hubert_pitch_ratio`      |   `0.08` | プレHuBERTシフト比率 (0.0-1.0)  |
| `moe_boost`                   |   `0.45` | Moeボイススタイル強度 (0.0-1.0) |
| `noise_scale`                 |   `0.45` | 合成ノイズスケール (0.0-1.0)    |
| `f0_lowpass_cutoff_hz`        |   `16.0` | F0ローパスカットオフ (Hz)       |
| `enable_octave_flip_suppress` |   `true` | 1オクターブF0飛び補正           |
| `enable_f0_slew_limit`        |   `true` | フレーム間F0変化量制限          |
| `f0_slew_max_step_st`         |    `3.6` | 最大F0ステップ (semitones)      |
| `denoise.enabled`             |   `true` | ノイズ除去                      |
| `denoise.method`              |     `ml` | `auto` / `ml` / `spectral`      |

### `RCWXConfig` (`rcwx/config.py`) トップレベル

| Key               |                Default | Notes                                             |
| ----------------- | ---------------------: | ------------------------------------------------- |
| `models_dir`      | `~/.cache/rcwx/models` | HuBERT・RMVPEモデルディレクトリ                   |
| `rvc_models_dir`  |                 `None` | RVCモデルディレクトリ（ドロップダウンスキャン用） |
| `last_model_path` |                 `None` | 最後に使用したモデルパス                          |
| `device`          |                 `auto` | auto/xpu/cuda/cpu                                 |
| `dtype`           |              `float16` | float16/float32/bfloat16                          |

### `RealtimeConfig` (`rcwx/pipeline/realtime_unified.py`)

実行時にGUI設定から生成される主要値:

- `chunk_sec` (既定 0.30、20ms境界に丸め)
- `overlap_sec` (既定 0.20、20ms境界に丸め)
- `crossfade_sec` (既定 0.08)
- `sola_search_ms` (既定 10.0)
- `hubert_context_sec` (既定 1.0)
- `prebuffer_chunks` (既定 1)
- `buffer_margin` (既定 0.5)
- `f0_method` (既定 `rmvpe`)
- `pre_hubert_pitch_ratio` (既定 0.08)
- `noise_scale` (既定 0.45)
- `f0_lowpass_cutoff_hz` (既定 16.0)
- `max_queue_size` (既定 8)

## GUI Latency Model (Current)

`LatencySettingsFrame` でユーザーが直接変更できるのは `chunk_sec` のみです。
他は自動導出されます。

自動導出ルール (`rcwx/gui/widgets/latency_settings.py`):

- `overlap_sec` = chunkの100%（60-200msにクランプ、20ms刻み）
- `crossfade_sec` = chunkの25%（10-80msにクランプ、10ms刻み）
- `prebuffer_chunks` = 1
- `buffer_margin` = 0.5
- `use_sola` = true

補足:

- GUI起動時は `config.audio.chunk_sec` を復元し、上記の自動値を再計算します。
- 実行中変更時は `set_overlap()` / `set_crossfade()` / `set_chunk_sec()` を使用します。

## CLI Commands

```powershell
uv run rcwx                  # GUI起動（command省略時デフォルト）
uv run rcwx gui              # GUI起動
uv run rcwx devices          # デバイス一覧
uv run rcwx download         # 必須モデルダウンロード
uv run rcwx run in.wav model.pth -o out.wav --pitch 5 --f0-method rmvpe
uv run rcwx info model.pth   # モデル情報
uv run rcwx diagnose         # フィードバック診断
uv run rcwx logs             # ログ一覧
uv run rcwx logs --tail 50   # 最新ログ末尾
uv run rcwx logs --open      # 最新ログを開く
```

## Logs & Diagnostics

ログ保存先:

- `~/.config/rcwx/logs/rcwx_YYYYMMDD_HHMMSS.log`

よく出るログ:

- `[INPUT] Queue full, dropping chunk`
- `[INFER] Chunk #...`
- `[PERF] Inference slow ...`
- `[FEEDBACK] Detected feedback (corr=...)`
- `[WARMUP] ...`

トラブルシュート:

- Queue full が頻発: `chunk_sec` 増、デノイズOFF、F0方式見直し
- 遅延増加: `chunk_sec`/`buffer_margin` 見直し、出力デバイス設定確認
- フィードバック警告: `uv run rcwx diagnose` 実施、入出力ループ回避
- XPU未認識: `uv sync` 後に `torch.__version__` が `+xpu` か確認

## Models

既定配置:

- `~/.cache/rcwx/models/hubert/hubert_base.pt`
- `~/.cache/rcwx/models/rmvpe/rmvpe.pt`
- RVC本体モデル: 任意の `*.pth`
- FAISS index: モデル隣接の `*.index`（自動検出）

## Tests

主なテスト:

```powershell
uv run python tests/integration/test_diagnostic.py
uv run python tests/integration/test_infer_streaming.py
uv run python tests/integration/test_pre_hubert_pitch.py
uv run python tests/integration/test_moe_clarity_scoring.py
uv run python tests/crossfade/test_sola_compensation.py
uv run python tests/models/test_inference.py
uv run python tests/models/test_rmvpe.py
uv run python tests/models/test_cumulative_context.py
uv run python tests/models/test_pitch_clarity.py
uv run python tests/models/test_config_wiring.py
uv run python tests/models/test_hubert_weight_audit.py
```

## References

- RVC WebUI
- PyTorch XPU
- Facebook Denoiser (`denoiser`)
- torchfcpe
