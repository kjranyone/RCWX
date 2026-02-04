# CLAUDE.md - RCWX Development Guide

## Project Overview

**RCWX** = RVC Real-time Voice Changer on Intel Arc (XPU)

RVC v2モデルを使ったリアルタイムボイスチェンジャー。Intel Arc GPU (XPU)に最適化した実装で、GUIとCLIの両方を提供します。

## Quick Start

```powershell
# 1. 依存関係インストール（PyTorch XPU版が自動的にインストールされる）
uv sync

# 2. XPU確認
uv run python -c "import torch; print(f'XPU: {torch.xpu.is_available()}')"

# 3. モデルダウンロード（HuBERT, RMVPE）
uv run rcwx download

# 4. (推奨) FCPE低レイテンシF0抽出をインストール
uv sync --extra lowlatency
# または: pip install torchfcpe

# 5. (オプション) ML Denoiserの確認（Facebook Denoiser）
uv run python -c "from rcwx.audio.denoise import is_ml_denoiser_available; print(f'ML Denoiser: {is_ml_denoiser_available()}')"

# 6. オーディオデバイス診断 (推奨: フィードバック防止)
uv run rcwx diagnose

# 7. GUI起動
uv run rcwx
```

## 初回起動時のデフォルト設定

`config.py` のデフォルト値に基づきます（GUIは保存済み設定を復元）。

- F0方式: `fcpe`
- Audio `chunk_sec`: **0.15s (150ms)**
- Realtime `chunk_sec`（実行時）: **0.10s (100ms)**
- `buffer_margin`: 0.3
- `context_sec`: 0.10s
- `crossfade_sec`: 0.05s
- `lookahead_sec`: 0.0s
- `use_sola`: true
- `use_parallel_extraction`: true
- `resample_method`: `linear`
- `chunking_mode`: `wokada`

F0方式ごとの最小チャンク（自動補正あり）:
- FCPE: 0.10s
- RMVPE: 0.32s

## GUIのレイテンシプリセット

`rcwx/gui/widgets/latency_settings.py` の定義。

- low: chunk 0.20s / prebuffer 0 / margin 0.5 / context 0.03 / crossfade 0.03
- balanced: chunk 0.35s / prebuffer 1 / margin 0.5 / context 0.10 / crossfade 0.05
- quality: chunk 0.50s / prebuffer 2 / margin 1.0 / context 0.15 / crossfade 0.08 / lookahead 0.05

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     RealtimeVoiceChanger                    │
├─────────────────────────────────────────────────────────────┤
│  AudioInput (48kHz)                                          │
│       │                                                      │
│       ▼                                                      │
│  ChunkBuffer (context + lookahead)                           │
│       │                                                      │
│       ▼                                                      │
│  [Input Queue] ──► Inference Thread                          │
│                         │                                    │
│                    Input Gain (dB)                           │
│                         │                                    │
│                    Resample 48k→16k                          │
│                         │                                    │
│                    Denoise (ML/Spectral)                     │
│                         │                                    │
│                    ┌─────────────────────┐                   │
│                    │  RVCPipeline.infer() │                  │
│                    ├──────────────────────┤                  │
│                    │ HuBERT (feature)     │                  │
│                    │ └─ Feature Cache     │                  │
│                    │ F0 (RMVPE/FCPE)      │                  │
│                    │ └─ F0 Cache          │                  │
│                    │ FAISS Index Search   │                  │
│                    │ Synthesizer          │                  │
│                    │ └─ Voice Gate        │                  │
│                    └──────────────────────┘                  │
│                         │                                    │
│                    Resample 40k→48k                          │
│                         │                                    │
│                    SOLA Crossfade                            │
│                         │                                    │
│                    Feedback Detection                         │
│                         │                                    │
│  [Output Queue] ◄───────┘                                    │
│       │                                                      │
│       ▼                                                      │
│  OutputBuffer (dynamic latency)                              │
│       │                                                      │
│       ▼                                                      │
│  AudioOutput (48kHz)                                         │
└─────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
rcwx/
├── cli.py                 # CLIエントリポイント、ログ設定
├── config.py              # RCWXConfig (JSON永続化)
├── device.py              # XPU/CUDA/CPU選択
├── diagnose.py            # オーディオフィードバック診断ツール
├── downloader.py          # モデルダウンロード
├── audio/
│   ├── input.py           # AudioInput (sounddevice)
│   ├── output.py          # AudioOutput (sounddevice)
│   ├── buffer.py          # ChunkBuffer, OutputBuffer
│   ├── crossfade.py       # SOLA処理
│   ├── crossfade_strategy.py # クロスフェード戦略
│   ├── analysis.py        # Adaptive parameter計算
│   ├── resample.py        # resample_poly / stateful resampler
│   └── denoise.py         # ML/Spectral denoise
├── audio/chunking/        # chunking mode実装
├── models/
│   ├── hubert.py          # HuBERTFeatureExtractor (transformers)
│   ├── hubert_fairseq.py  # Fairseq形式HuBERT
│   ├── hubert_loader.py   # HuBERT統合ローダー
│   ├── rmvpe.py           # RMVPE F0抽出
│   ├── fcpe.py            # FCPE F0抽出
│   ├── synthesizer.py     # SynthesizerLoader
│   └── infer_pack/        # RVC WebUIから移植したコアモジュール
├── pipeline/
│   ├── inference.py       # RVCPipeline (単発推論、特徴キャッシング)
│   └── realtime.py        # RealtimeVoiceChanger
└── gui/
    ├── app.py             # RCWXApp (CustomTkinter)
    └── widgets/           # UIコンポーネント
```

## Key Configuration

### AudioConfig (config.py)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `sample_rate` | 16000 | 入力処理レート |
| `output_sample_rate` | 48000 | 出力レート |
| `chunk_sec` | 0.15 | チャンクサイズ（GUIの初期値） |
| `crossfade_sec` | 0.05 | クロスフェード長 |
| `prebuffer_chunks` | 1 | 出力開始前のプリバッファ |
| `buffer_margin` | 0.3 | バッファマージン |
| `input_gain_db` | 0.0 | 入力ゲイン |

### InferenceConfig (config.py)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `f0_method` | `fcpe` | F0方式 (`fcpe` / `rmvpe`) |
| `use_parallel_extraction` | true | HuBERT+F0並列抽出 |
| `resample_method` | `linear` | `linear` / `poly` |
| `index_k` | 4 | FAISS近傍数 |
| `voice_gate_mode` | `expand` | off/strict/expand/energy |
| `energy_threshold` | 0.05 | energyモード閾値 |
| `use_feature_cache` | true | チャンク連続性キャッシュ |
| `context_sec` | 0.10 | 左コンテキスト |
| `lookahead_sec` | 0.0 | 先読み（レイテンシ増） |
| `crossfade_sec` | 0.05 | クロスフェード |
| `use_sola` | true | SOLA有効化 |
| `chunking_mode` | `wokada` | `wokada` / `rvc_webui` / `hybrid` |
| `denoise.enabled` | false | ノイズ除去有効化 |
| `denoise.method` | `auto` | `auto` / `ml` / `spectral` |

### RealtimeConfig (pipeline/realtime.py)

実行時にGUIの設定を反映して生成されます。追加パラメータ:
- `chunking_mode`: `wokada` / `rvc_webui` / `hybrid`
- `rvc_overlap_sec`: 0.22（rvc_webui専用）
- `use_adaptive_parameters`: 既定 false（クロスフェード/コンテキストを自動調整）
- `use_energy_normalization`: 既定 false（出力エネルギー正規化）

## Chunking Modes

- `wokada`: コンテキスト付きチャンク処理。低レイテンシ。
- `rvc_webui`: オーバーラップ基準のチャンク（連続性重視）。
- `hybrid`: RVC hop + w-okada context + 最適化SOLA。

## CLI Commands

```powershell
uv run rcwx              # GUI起動
uv run rcwx devices      # デバイス一覧
uv run rcwx download     # モデルダウンロード
uv run rcwx run in.wav model.pth -o out.wav --pitch 5
uv run rcwx info model.pth
uv run rcwx diagnose     # フィードバック診断
uv run rcwx logs         # ログファイル一覧
uv run rcwx logs --tail 50   # 最新ログの末尾50行
uv run rcwx logs --open      # 最新ログを開く
```

## Log Investigation

### ログファイル

```
~/.config/rcwx/logs/rcwx_YYYYMMDD_HHMMSS.log
```

### ログタグ

| タグ | 説明 |
|------|------|
| `[INPUT]` | 入力コールバック / 入力キュー監視 |
| `[OUTPUT]` | 出力コールバック / バッファ監視 |
| `[INFER]` | 推論スレッド / 処理時間 / レイテンシ |
| `[SOLA]` | SOLA処理オフセット |
| `[FEEDBACK]` | フィードバック検出 |

## Troubleshooting

### バッファアンダーラン (音切れ)

```
[OUTPUT] Buffer underrun #1
```

- `chunk_sec` を増やす
- `buffer_margin` を増やす
- `f0_method` を `fcpe` または `none` にする

### バッファオーバーラン (遅延増加)

```
[INPUT] Queue full, dropping chunk
```

- 出力デバイスのサンプルレート確認
- デノイズを無効化
- `max_queue_size` を確認

### フィードバック検出

```
[FEEDBACK] 音声フィードバックを検出しました (相関係数=0.45)
```

- `rcwx diagnose` で設定確認
- 入出力デバイスを別インターフェースにする
- Windowsの「このデバイスを聴く」を無効化

### XPUが認識されない

```powershell
uv run python -c "import torch; print(torch.__version__, torch.xpu.is_available())"
```

- `+xpu` でない場合: `uv sync --reinstall` を実行

## Tests

### チャンク処理統合テスト

```powershell
uv run python tests/test_realtime_chunk_processing.py
```

## References

- RVC WebUI
- PyTorch XPU
- Facebook Denoiser (denoiser)
- FCPE (torchfcpe)
- w-okada Voice Changer
