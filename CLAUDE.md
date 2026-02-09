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
- Audio `chunk_sec`: **0.15s (150ms)** (GUIで唯一のユーザー制御パラメータ)
- 以下はchunk_secから自動導出 (GUI上は読み取り専用):
  - `overlap_sec`: 80ms (chunk_secの50%, 60-200ms, 20ms刻み)
  - `crossfade_sec`: 40ms (chunk_secの25%, 10-80ms, 10ms刻み)
  - `prebuffer_chunks`: 1 (固定)
  - `buffer_margin`: 0.5 (固定)
  - `lookahead_sec`: 0.0 (固定)
  - `use_sola`: true (固定)
- `sola_search_ms`: 10.0
- `use_parallel_extraction`: true
- `resample_method`: `linear`

F0方式ごとの最小チャンク（自動補正あり）:
- FCPE: 0.10s
- RMVPE: 0.32s

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│               RealtimeVoiceChangerUnified                     │
├─────────────────────────────────────────────────────────────┤
│  AudioInput (48kHz)                                          │
│       │                                                      │
│       ▼                                                      │
│  InputAccumulator (hop蓄積)                                   │
│       │                                                      │
│       ▼                                                      │
│  [Input Queue] ──► Inference Thread                          │
│                         │                                    │
│                    Input Gain (dB)                           │
│                         │                                    │
│                    Resample 48k→16k (StatefulResampler)      │
│                         │                                    │
│                    Denoise (ML/Spectral, optional)           │
│                         │                                    │
│                    ChunkAssembly                              │
│                    [overlap | new_hop] (HuBERT整列)           │
│                         │                                    │
│                    ┌──────────────────────────┐              │
│                    │ RVCPipeline.infer_streaming() │          │
│                    ├──────────────────────────┤              │
│                    │ HuBERT (全音声)           │              │
│                    │ F0 (全音声) [並列抽出]     │              │
│                    │ overlapフレーム除去        │              │
│                    │ FAISS Index Search        │              │
│                    │ Synthesizer               │              │
│                    │ Voice Gate                │              │
│                    └──────────────────────────┘              │
│                         │                                    │
│                    Resample model_sr→48k (StatefulResampler) │
│                         │                                    │
│                    SimpleSola (conv1d相関 + Hann窓)          │
│                         │                                    │
│                    Feedback Detection                         │
│                         │                                    │
│  [Output Queue] ◄───────┘                                    │
│       │                                                      │
│       ▼                                                      │
│  RingOutputBuffer (事前確保、3×chunk容量)                     │
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
│   ├── buffer.py          # ChunkBuffer, OutputBuffer, RingOutputBuffer
│   ├── sola.py            # シンプルSOLA (相関+Hann窓クロスフェード)
│   ├── resample.py        # resample_poly / stateful resampler
│   └── denoise.py         # ML/Spectral denoise
├── models/
│   ├── hubert.py          # HuBERTFeatureExtractor (transformers)
│   ├── hubert_fairseq.py  # Fairseq形式HuBERT
│   ├── hubert_loader.py   # HuBERT統合ローダー
│   ├── rmvpe.py           # RMVPE F0抽出
│   ├── fcpe.py            # FCPE F0抽出
│   ├── synthesizer.py     # SynthesizerLoader
│   └── infer_pack/        # RVC WebUIから移植したコアモジュール
├── pipeline/
│   ├── inference.py       # RVCPipeline (バッチ推論 + infer_streaming)
│   └── realtime_unified.py # RealtimeVoiceChangerUnified (統合パイプライン)
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
| `overlap_sec` | 0.10 | 音声レベルオーバーラップ（HuBERT連続性） |
| `lookahead_sec` | 0.0 | 先読み（レイテンシ増） |
| `crossfade_sec` | 0.05 | SOLAクロスフェード長 |
| `use_sola` | true | SOLA有効化 |
| `sola_search_ms` | 10.0 | SOLA探索窓（ms） |
| `denoise.enabled` | false | ノイズ除去有効化 |
| `denoise.method` | `auto` | `auto` / `ml` / `spectral` |

### RealtimeConfig (pipeline/realtime_unified.py)

実行時にGUIの設定を反映して生成されます。主要パラメータ:
- `chunk_sec`: チャンクサイズ（HuBERTフレーム境界20msに自動整列）
- `overlap_sec`: 音声オーバーラップ（デフォルト0.10s）
- `crossfade_sec`: SOLAクロスフェード長
- `sola_search_ms`: SOLA探索窓（デフォルト10ms）
- `prebuffer_chunks`: プリバッファチャンク数
- `buffer_margin`: バッファマージン

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

リアルタイム音声変換の品質を検証するためのテストフレームワーク。

### クイックスタート

```powershell
# コンポーネント診断テスト（モデル必要）
uv run python tests/integration/test_diagnostic.py

# infer_streaming() API検証（モデル必要）
uv run python tests/integration/test_infer_streaming.py
```

### ディレクトリ構成

```
tests/
├── integration/                   # 統合テスト（モデル必要）
│   ├── test_diagnostic.py             # コンポーネント別診断
│   ├── test_realtime_integration.py   # マルチスレッド統合テスト
│   ├── test_infer_streaming.py        # infer_streaming() API検証
│   └── test_chunking_modes_comparison.py  # チャンキング比較
│
├── crossfade/                     # SOLA・クロスフェード
│   └── test_sola_compensation.py      # SOLAタイミングドリフト検証
│
├── models/                        # モデル固有テスト
│   ├── test_inference.py              # RVCパイプライン推論
│   ├── test_rmvpe.py                  # RMVPE F0抽出
│   └── test_cumulative_context.py     # HuBERT累積コンテキスト
│
└── test_output/                   # テスト出力
```

### test_diagnostic.py

各処理コンポーネントを個別にテスト。

```powershell
uv run python tests/integration/test_diagnostic.py --component all
uv run python tests/integration/test_diagnostic.py --component sola      # SOLA単体
uv run python tests/integration/test_diagnostic.py --component resampler # リサンプラ単体
```

**テスト項目**:
- `resampler`: StatefulResampler vs バッチ（相関 > 0.99）
- `sola`: クロスフェード品質（不連続性 = 0）
- `latency`: サンプル数の累積誤差（< 10ms）

### test_realtime_integration.py

GUIと同等のスレッド構成（入力/推論/出力）でテスト。
SimulatedAudioDeviceで実デバイスのタイミング（ジッター含む）をエミュレート。

```powershell
uv run python tests/integration/test_realtime_integration.py                  # 10秒テスト
uv run python tests/integration/test_realtime_integration.py --duration 60    # 長時間テスト
uv run python tests/integration/test_realtime_integration.py --stress         # CPU負荷テスト
```

**評価項目**:
- `underruns`: 出力バッファ枯渇（音切れ）
- `overruns`: 入力キュー溢れ
- `latency`: チャンク処理時間

### 品質基準

| 指標 | 合格基準 |
|------|----------|
| SOLA不連続性 | 0 件 |
| Resampler相関 | > 0.99 |
| 累積時間誤差 | < 10ms |
| Underruns | 0 件 |
| Overruns | < 5 件 |

### テストデータ

`sample_data/` に含まれるテスト用音声ファイル:

| ファイル | 内容 | 用途 |
|----------|------|------|
| `sustained_voice.wav` | 持続母音（デフォルト） | 基本テスト |
| `sustained_tone.wav` | 持続音 | 音程安定性テスト |
| `pure_sine.wav` | 純正弦波 | 信号処理検証 |
| `nc283304.mp3` | 音声サンプル | 実音声テスト |

### テストの限界

現在のテストでカバーしていない項目:

| 項目 | 説明 |
|------|------|
| 実デバイスレイテンシ | sounddeviceドライバの遅延 |
| 自然音声 | 子音・無音区間・ビブラート |
| パラメータ変更 | 変換中のpitch/index_rate変更 |
| 長時間安定性 | 数時間の連続動作 |

## References

- RVC WebUI
- PyTorch XPU
- Facebook Denoiser (denoiser)
- FCPE (torchfcpe)
- w-okada Voice Changer
