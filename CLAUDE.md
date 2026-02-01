# CLAUDE.md - RCWX Development Guide

## Project Overview

**RCWX** = RVC Real-time Voice Changer on Intel Arc (XPU)

RVCv2モデルを使用したリアルタイムボイスチェンジャー。Intel Arc GPU (XPU) に最適化されたフルスクラッチ実装。

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

# 5. (オプション) DeepFilterNetノイズキャンセリングの確認
uv run python -c "from rcwx.audio.denoise import is_ml_denoiser_available; print(f'DeepFilterNet: {is_ml_denoiser_available()}')"

# 6. オーディオデバイス診断 (推奨: フィードバック防止)
uv run rcwx diagnose

# 7. GUI起動
uv run rcwx
```

**初回起動時のデフォルト設定** (超低レイテンシ最適化):
- F0方式: **FCPE** (低レイテンシ、100ms min) ← デフォルト
  - RMVPEに変更可能 (高品質、320ms min)
- Voice Gate: expand (推奨)
- Feature Cache: 有効 (デフォルト)
- SOLA: 有効 (デフォルト)
- **チャンクサイズ: 100ms** (超低レイテンシ、FCPE公式最小値) ← デフォルト
  - RMVPEの場合は350msに変更推奨
- **Buffer Margin: 0.3** (タイトバッファ) ← デフォルト
- **Parallel Extraction: 有効** (10-20%高速化) ← デフォルト
- **torch.compile: 無効** (Windows XPU安定性優先) ← デフォルト

**レイテンシ目標**:
- FCPE: **200-250ms** (デフォルト、実測値)
- RMVPE: 400-500ms (高品質モード)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     RealtimeVoiceChanger                     │
├─────────────────────────────────────────────────────────────┤
│  AudioInput (48kHz)                                          │
│       │                                                      │
│       ▼                                                      │
│  ChunkBuffer (w-okada style: context + lookahead)            │
│       │                                                      │
│       ▼                                                      │
│  [Input Queue] ──► Inference Thread                          │
│                         │                                    │
│                    Input Gain (dB調整)                       │
│                         │                                    │
│                    Resample 48k→16k                          │
│                         │                                    │
│                    Denoise (DeepFilterNet/Spectral)          │
│                         │                                    │
│                    ┌─────────────────────┐                   │
│                    │  RVCPipeline.infer() │                  │
│                    ├──────────────────────┤                  │
│                    │ HuBERT (特徴抽出)    │                  │
│                    │ └─ Feature Cache     │ ◄──┐            │
│                    │                      │    │            │
│                    │ F0 (RMVPE/FCPE)      │    │ チャンク   │
│                    │ └─ F0 Cache          │ ◄──┤ 連続性     │
│                    │                      │    │            │
│                    │ FAISS Index Search   │    │            │
│                    │                      │    │            │
│                    │ Synthesizer          │    │            │
│                    │ └─ Voice Gate        │    │            │
│                    └──────────────────────┘    │            │
│                         │                      │            │
│                    Resample 40k→48k            │            │
│                         │                      │            │
│                    SOLA Crossfade ─────────────┘            │
│                         │                                    │
│                    Feedback Detection (相関係数監視)         │
│                         │                                    │
│  [Output Queue] ◄───────┘                                    │
│       │                                                      │
│       ▼                                                      │
│  OutputBuffer (動的レイテンシ制御)                           │
│       │                                                      │
│       ▼                                                      │
│  AudioOutput (48kHz)                                         │
└─────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
rcwx/
├── cli.py                 # CLIエントリポイント、ログ設定
├── config.py              # RCWXConfig (JSON永続化、AudioConfig, InferenceConfig)
├── device.py              # XPU/CUDA/CPU選択
├── diagnose.py            # オーディオフィードバック診断ツール
├── downloader.py          # HuggingFaceモデルダウンロード
├── audio/
│   ├── input.py           # AudioInput (sounddevice)
│   ├── output.py          # AudioOutput (sounddevice)
│   ├── buffer.py          # ChunkBuffer, OutputBuffer
│   ├── crossfade.py       # SOLA処理 (RVC WebUI方式)
│   ├── resample.py        # scipy.signal.resample_poly
│   └── denoise.py         # DeepFilterNet, SpectralGate
├── models/
│   ├── hubert.py          # HuBERTFeatureExtractor (transformers)
│   ├── hubert_fairseq.py  # Fairseq形式HuBERT (RVC WebUI互換)
│   ├── hubert_loader.py   # 統合HuBERTローダー
│   ├── rmvpe.py           # RMVPE F0抽出
│   ├── fcpe.py            # FCPE F0抽出 (低レイテンシ)
│   ├── synthesizer.py     # SynthesizerLoader
│   └── infer_pack/        # RVC WebUIから移植したコアモジュール
├── pipeline/
│   ├── inference.py       # RVCPipeline (単発推論、特徴キャッシング)
│   └── realtime.py        # RealtimeVoiceChanger (リアルタイム処理、フィードバック検出)
└── gui/
    ├── app.py             # RCWXApp (CustomTkinter)
    └── widgets/           # UIコンポーネント
        ├── audio_settings.py      # オーディオデバイス設定
        ├── latency_settings.py    # レイテンシ設定
        ├── latency_monitor.py     # ステータスバー
        ├── model_selector.py      # モデル選択
        └── pitch_control.py       # ピッチ・F0設定
```

## Key Configuration

### pyproject.toml - PyTorch XPU設定

```toml
[tool.uv]
environments = ["sys_platform == 'win32'"]
override-dependencies = [
    "triton-xpu; sys_platform == 'linux'",
    "pytorch-triton-xpu; sys_platform == 'linux'",
]

[tool.uv.sources]
torch = { index = "pytorch-xpu" }
torchaudio = { index = "pytorch-xpu" }

[[tool.uv.index]]
name = "pytorch-xpu"
url = "https://download.pytorch.org/whl/xpu"
explicit = true
```

### RealtimeConfig (realtime.py)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `mic_sample_rate` | 48000 | マイク入力レート |
| `input_sample_rate` | 16000 | 処理レート (HuBERT/RMVPE用) |
| `output_sample_rate` | 48000 | 出力レート |
| `chunk_sec` | 0.10 | チャンクサイズ (FCPE公式最小値: 0.10, RMVPE: >=0.32) |
| `f0_method` | fcpe | F0抽出方法 (fcpe=低レイテンシ, rmvpe=高品質) |
| `max_queue_size` | 8 | キュー最大サイズ |
| `prebuffer_chunks` | 1 | 出力開始前のプリバッファ |
| `buffer_margin` | 0.3 | バッファマージン (0.3=tight, 0.5=balanced, 1.0=relaxed) |
| `use_parallel_extraction` | true | 並列HuBERT+F0抽出 (10-20%高速化) |
| `input_gain_db` | 0.0 | 入力ゲイン (dB) |
| `index_rate` | 0.0 | FAISSインデックス率 (0=無効, 0.5=バランス, 1=インデックスのみ) |
| `denoise_enabled` | false | ノイズキャンセリング有効化 |
| `denoise_method` | auto | ノイズ除去方式 (auto/deepfilter/spectral) |
| `voice_gate_mode` | expand | ボイスゲートモード (off/strict/expand/energy) |
| `energy_threshold` | 0.05 | エネルギー閾値 (energyモード用: 0.01-0.2) |
| `use_feature_cache` | true | HuBERT/F0特徴キャッシング (チャンク連続性) |
| `context_sec` | 0.05 | 左右コンテキスト (w-okada style) |
| `extra_sec` | 0.0 | 追加カット範囲 |
| `crossfade_sec` | 0.05 | クロスフェード長 |
| `lookahead_sec` | 0.0 | 先読み (レイテンシ増加!) |
| `use_sola` | true | SOLA (最適クロスフェード位置探索、w-okada対応) |

## CLI Commands

```powershell
uv run rcwx              # GUI起動
uv run rcwx devices      # デバイス一覧
uv run rcwx download     # モデルダウンロード
uv run rcwx run in.wav model.pth -o out.wav --pitch 5
uv run rcwx info model.pth
uv run rcwx diagnose     # オーディオフィードバック診断
uv run rcwx logs         # ログファイル一覧
uv run rcwx logs --tail 50   # 最新ログの末尾50行
uv run rcwx logs --open      # 最新ログを開く
```

## Log Investigation

### ログファイルの場所

```
~/.config/rcwx/logs/rcwx_YYYYMMDD_HHMMSS.log
```

### ログタグの意味

| タグ | 説明 |
|------|------|
| `[INPUT]` | 入力コールバック - バッファ状態、キューサイズ |
| `[INPUT-RAW]` | 入力オーディオの署名 (フィードバック検出用) |
| `[OUTPUT]` | 出力コールバック - バッファ状態、ドロップ数 |
| `[OUTPUT-PROC]` | 処理済み出力の署名 (フィードバック検出用) |
| `[INFER]` | 推論スレッド - 処理時間、レイテンシ |
| `[SOLA]` | SOLA処理 - オフセット値 |
| `[FEEDBACK]` | フィードバック検出 - 相関係数 |

### 正常動作時のログ例

```
[INPUT] received=4200, input_buffer=16800, input_queue=0
[INFER] Chunk #1: in=16800, out=14000, infer=45ms, latency=180ms, buf=14000, under=0, over=0
[SOLA] chunk=1, offset=23
[OUTPUT] frames=4200, chunks_added=1, output_buffer=14000, output_queue=0, dropped=0
```

### 問題パターン

#### 1. バッファアンダーラン (音切れ)
```
[OUTPUT] Buffer underrun #1
[OUTPUT] Buffer underrun #2
```
**原因**: 推論が間に合っていない
**対策**:
- `chunk_sec` を増やす (0.35 → 0.40)
- F0方式を変更 (rmvpe → fcpe または none)
- `buffer_margin` を増やす (0.5 → 1.0)

#### 2. バッファオーバーラン (遅延増加)
```
[INPUT] Queue full, dropping chunk
[OUTPUT] ... dropped=120
```
**原因**: 推論が遅すぎる、または出力が消費されていない
**対策**:
- 出力デバイスのサンプルレート確認
- デノイズ無効化でテスト
- `max_queue_size` 確認

#### 3. 入力遅延 (チャンク蓄積)
```
[INPUT] Falling behind: queued 3 chunks at once
```
**原因**: 入力処理が追いついていない
**対策**: チャンクサイズ・マイクサンプルレート確認

#### 4. フィードバック検出
```
[FEEDBACK] 音声フィードバックを検出しました (相関係数=0.45)
```
**原因**: 入出力がループしている
**対策**: `rcwx diagnose` 実行、デバイス設定確認

## Known Issues

### 1. フィードバックループ (解決済み - 自動検出実装)

**症状**: 音声がエコーのように繰り返される、ピッチが蓄積する

**対策**:
- **自動検出**: 相関係数監視による自動警告
- **診断ツール**: `rcwx diagnose` で設定確認
- **推奨設定**:
  - 入力・出力で異なるインターフェース使用
  - Windowsの「このデバイスを聴く」を無効化
  - Stereo Mix等のループバックデバイスを避ける

### 2. XPUが認識されない

```
XPU available: False
```

**確認**:
```powershell
uv run python -c "import torch; print(torch.__version__)"
# → 2.10.0+xpu であることを確認 (+cpu だと問題)
```

**対策**: `uv.lock` を削除して `uv sync` を再実行

### 3. uv run でパッケージが入れ替わる

`uv run` は `uv.lock` に同期するため、手動インストールしたパッケージが上書きされる。
`pyproject.toml` でXPUインデックスを設定済みなので `uv sync` で正しくインストールされる。

### 4. DeepFilterNet (ML Denoiser) のインストール

MLベースのノイズキャンセリングを使用する場合:

```powershell
# denoiserパッケージは既に依存関係に含まれている
uv sync

# DeepFilterNetが利用可能か確認
uv run python -c "from rcwx.audio.denoise import is_deepfilter_available; print(is_deepfilter_available())"
```

**注意**: DeepFilterNetはCPU専用で実行されます（XPU/CUDA非対応）。

### 5. FCPE (低レイテンシF0抽出) のインストール

低レイテンシモードでF0抽出を使用する場合:

```powershell
pip install torchfcpe

# FCPEが利用可能か確認
uv run python -c "from rcwx.models.fcpe import is_fcpe_available; print(is_fcpe_available())"
```

**利点**:
- 最小チャンクサイズ 100ms (RMVPE: 320ms)
- より低いレイテンシ (80-150ms)
- F0品質はRMVPEよりやや劣る

### 6. SOLA (解決済み - w-okada対応実装完了)

**以前の問題**: SOLA有効時に出力音声が入力の40-50%の長さになる

**原因**:
- 元のSOLA実装がRVC WebUI方式のオーバーラップチャンキングを前提
- w-okada方式のコンテキストチャンキングと非互換
- 各チャンクから可変長の音声を削除していた

**解決方法**:
- **w-okada対応モード実装** (`wokada_mode=True`)
- 左コンテキストで最適オフセットを探索
- 固定長（context_sec）のトリミング
- バッファは保存するが出力から削除しない

**現状**:
- デフォルトで有効化済み (`use_sola=true`)
- 出力長さ正常: 5.00s入力 → 5.04s出力 (100.8%)
- 位相整合されたクロスフェードで高音質

## Audio Device Setup

### 推奨構成

#### パターン1: ローカル試聴 (フィードバック回避)

```
入力: USBマイク (Fifine K420等)
出力: オンボードスピーカー/ヘッドホン

利点: 異なるインターフェースでフィードバック回避
注意: 入出力が同じ "High Definition Audio" だとフィードバックの可能性
```

#### パターン2: Discord/OBS等で使用 (VB-Cable)

```
RCWX:
  入力: 物理マイク (Fifine K420等)
  出力: CABLE Input

Discord/OBS:
  入力: CABLE Output
  出力: スピーカー/ヘッドホン

警告: CABLE OutputをRCWX入力に設定するとフィードバックループが発生!
```

#### パターン3: VoiceMeeter使用

```
VoiceMeeter:
  Hardware Input 1: 物理マイク
  Hardware Out A1: スピーカー

RCWX:
  入力: VoiceMeeter VAIO (Input)
  出力: VoiceMeeter Aux (Input)

Discord/OBS:
  入力: VoiceMeeter Output

注意: VoiceMeeter内でループバック設定を確認
```

### デバイス診断

```powershell
# フィードバック診断実行
uv run rcwx diagnose

# 確認項目:
# - 同じインターフェースを入出力に使用していないか
# - ループバックデバイス (Stereo Mix等) を使用していないか
# - Windowsの「このデバイスを聴く」が無効か
```

## Development

### Lint & Format

```powershell
uv run ruff check rcwx
uv run ruff format rcwx
```

### デバッグ実行

```powershell
# 詳細ログ付きで実行
uv run rcwx --verbose

# ログ確認
uv run rcwx logs --tail 100

# フィードバック診断
uv run rcwx diagnose
```

### テスト実行

#### チャンク処理統合テスト

リアルタイム変換のチャンク処理ロジックを検証するテスト。実際の`RealtimeVoiceChanger`モジュールを直接呼び出してバッチ処理と比較します。

```powershell
# チャンク処理テスト (バッチ vs ストリーミング)
uv run python tests/test_realtime_chunk_processing.py
```

**このテストは何をするか**:
1. WAVファイルをバッチ処理で変換
2. 同じWAVを小さなチャンク（20ms単位）に分割して処理
3. 実際の`RealtimeVoiceChanger`の公開メソッドを使用:
   - `process_input_chunk()`: 入力追加
   - `process_next_chunk()`: 推論実行
   - `get_output_chunk()`: 出力取得
4. バッチとストリーミングの出力を比較

**期待される結果**:
- 相関係数 >= 0.93（実用上十分）
- MAE (平均絶対誤差) <= 0.05
- エネルギー比 ≈ 1.0（0.9-1.1の範囲）

**出力ファイル**:
- `test_output/batch_output.wav` - バッチ処理結果
- `test_output/streaming_output.wav` - ストリーミング処理結果

**テスト対象**:
- チャンク間の連続性 (特徴量キャッシュ)
- コンテキストバッファリング
- SOLA クロスフェード
- リサンプリングの一貫性

#### テスト実装の重要ポイント

**1. バッチ処理もチャンク単位で処理**

公平な比較のため、バッチ処理も**ストリーミングと同じチャンクロジック**で実装：

```python
def process_batch(pipeline, audio, chunk_sec=0.35, context_sec=0.05):
    # ストリーミングと同じ順序で処理
    # 1. 48kHzでチャンク分割（マイクレート）
    # 2. チャンクごとに16kHzへリサンプル（処理レート）
    # 3. 推論実行
    # 4. 48kHzへリサンプル（出力レート）
    # 5. コンテキストトリミング

    main_pos = 0
    chunk_idx = 0
    outputs = []

    while main_pos < len(audio):
        if chunk_idx == 0:
            # 最初のチャンク: コンテキストなし
            chunk = audio[0:chunk_samples]
        else:
            # 2番目以降: 左コンテキスト + メイン
            start = main_pos - context_samples
            end = main_pos + chunk_samples
            chunk = audio[start:end]

        # 処理...
        main_pos += chunk_samples  # 常にchunk_samplesずつ進む
```

**2. ChunkBufferの最初のチャンク処理**

`ChunkBuffer`は最初のチャンクを特別扱い：

```python
class ChunkBuffer:
    def __init__(...):
        self._is_first_chunk = True  # フラグ追加

    def has_chunk(self) -> bool:
        if self._is_first_chunk:
            # 最初: コンテキストなし
            required = self.chunk_samples + self.lookahead_samples
        else:
            # 2番目以降: コンテキストあり
            required = self.chunk_samples + self.context_samples + self.lookahead_samples
        return len(self._input_buffer) >= required

    def get_chunk(self):
        chunk = self._input_buffer[:required].copy()

        if self._is_first_chunk:
            # 最初のチャンク: (chunk - context)だけ進める
            # これにより次のチャンクの左コンテキストが残る
            advance = self.chunk_samples - self.context_samples
            self._input_buffer = self._input_buffer[advance:]
            self._is_first_chunk = False
        else:
            # 2番目以降: chunk_samplesずつ進める
            self._input_buffer = self._input_buffer[self.chunk_samples:]

        return chunk
```

**3. 出力のコンテキストトリミング**

推論後、各チャンクから**左コンテキスト部分を削除**（最初のチャンクを除く）：

```python
# realtime.py の process_next_chunk() 内
if self.stats.frames_processed > 0 and self.config.context_sec > 0:
    context_samples_output = int(self.config.output_sample_rate * self.config.context_sec)
    if len(output) > context_samples_output:
        output = output[context_samples_output:]  # 左コンテキスト削除
```

**4. feature_cacheのクリア**

テスト前に必ず`pipeline.clear_cache()`を実行してウォームアップの影響を除去：

```python
# バッチ処理開始前
pipeline.clear_cache()

# ストリーミング処理開始前
pipeline.clear_cache()
```

**5. 出力バッファサイズの調整**

テスト用に大きなバッファを設定（本番では小さくしてレイテンシを抑える）：

```python
# テスト用: すべての出力を保持
expected_chunks = int(audio_duration / chunk_sec) + 2
max_output = expected_chunks * chunk_output_size * 2
changer.output_buffer.set_max_latency(max_output)
```

#### なぜ相関係数が0.93なのか

完全一致（0.95+）を達成できない理由：

1. **浮動小数点演算の累積誤差**: 150チャンク × 複数回のリサンプル
2. **SOLAの適応的位相揃え**: チャンクごとに最適オフセットを探索
3. **feature_cacheの動的更新**: キャッシュの状態遷移がわずかに異なる
4. **最初のチャンクのウォームアップ**: 初期10チャンクで相関0.92（50チャンクで0.94に改善）

**相関係数0.93は音響信号処理として十分高いレベル**であり、実用上問題ありません。

このテストにより、実際のリアルタイム変換ロジックが正しく動作しているか検証できます。

### パフォーマンス最適化

**torch.compile**:
- Windows: 無効化 (Triton非対応)
- Linux: 自動有効化 (推論速度20-30%向上)
- XPU: 初回実行時にカーネルコンパイル (数秒〜数十秒)

**デノイズ最適化**:
- DeepFilterNet: CPU専用、高品質だが遅い (推論時間+10-30ms)
- Spectral Gate: 軽量、リアルタイム向け (推論時間+1-3ms)
- 推奨: ノイズが少ない環境では無効化

**F0最適化**:
| 方式 | チャンク最小 | 推論時間 | レイテンシ | 品質 |
|------|------------|---------|-----------|------|
| RMVPE | 320ms | 40-60ms | 150-250ms | 高 |
| FCPE | 100ms | 20-30ms | 80-150ms | 中 |
| none | 任意 | 10-20ms | 50-100ms | - |

**ウォームアップ**:
- 初回推論は遅い (XPU/CUDAカーネルコンパイル)
- `start()`時に自動ウォームアップ実行 (4回推論)
- 各コードパス (silence, voice, index) を網羅

**メモリ使用量**:
- HuBERT (v2): ~300MB (768-dim)
- RMVPE: ~100MB
- Synthesizer: ~50-100MB (モデルによる)
- 合計: 約500-600MB VRAM

### 重要なクラスと実装場所

| クラス | ファイル | 主要メソッド | 役割 |
|--------|----------|------------|------|
| `RCWXApp` | gui/app.py | `_on_start_stop()` | メインGUIアプリケーション |
| `RealtimeVoiceChanger` | pipeline/realtime.py:112 | `start()`, `_inference_thread()` | リアルタイム処理統合 |
| `RVCPipeline` | pipeline/inference.py:47 | `infer()` | RVC推論パイプライン |
| `ChunkBuffer` | audio/buffer.py | `get_chunk()` | w-okada style入力バッファ |
| `OutputBuffer` | audio/buffer.py | `add()`, `get()` | 動的レイテンシ制御 |
| `SOLAState` | audio/crossfade.py | - | SOLA状態保持 |
| `apply_sola_crossfade()` | audio/crossfade.py:30 | - | RVC式クロスフェード |
| `HuBERTLoader` | models/hubert_loader.py:10 | `extract()` | HuBERT統合ローダー |
| `RMVPE` | models/rmvpe.py:8 | `infer()` | F0抽出 (高品質) |
| `FCPE` | models/fcpe.py:9 | `infer()` | F0抽出 (低レイテンシ) |
| `SynthesizerLoader` | models/synthesizer.py:18 | `infer()` | RVC Synthesizer |

### 重要な処理フロー

#### リアルタイム処理 (realtime.py)

```python
# 1. 入力コールバック (_on_audio_input, L346)
#    - 入力オーディオをChunkBufferに追加
#    - チャンク取得可能なら Input Queue に送信

# 2. 推論スレッド (_inference_thread, L431)
#    - Input Queue からチャンク取得
#    - 入力ゲイン適用 (L469)
#    - リサンプル mic_sr → input_sr (L488)
#    - デノイズ適用 (L501)
#    - RVCPipeline.infer() 実行 (L519)
#    - リサンプル model_sr → output_sr (L531)
#    - SOLA クロスフェード (L546)
#    - フィードバック検出 (L570)
#    - Output Queue に送信 (L616)

# 3. 出力コールバック (_on_audio_output, L385)
#    - Output Queue から処理済み音声取得
#    - OutputBuffer に追加
#    - プリバッファ確認後、出力開始
```

#### RVC推論 (inference.py)

```python
# 1. 前処理 (L343-425)
#    - リサンプル → 16kHz (L347)
#    - ノイズ除去 (L353)
#    - ハイパスフィルタ (L372)
#    - リフレクションパディング (L412)

# 2. HuBERT特徴抽出 (L437)
#    - レイヤー指定 (v1: 9, v2: 12)
#    - 次元指定 (v1: 256, v2: 768)

# 3. FAISS Index検索 (L443)
#    - k=8 近傍探索
#    - 逆距離二乗重み付け平均

# 4. Feature Cache ブレンディング (L448)
#    - 前チャンク末尾 10フレームとクロスフェード

# 5. 特徴補間 (L469)
#    - 50fps → 100fps (2x線形補間)

# 6. F0抽出 (L486-540)
#    - RMVPE/FCPE実行
#    - ピッチシフト適用
#    - 長さ整合、F0キャッシュブレンド
#    - メルスケール量子化 (1-255)

# 7. Synthesizer推論 (L607)
#    - NSF decoder (F0あり) or Standard (F0なし)

# 8. Voice Gate適用 (L633)
#    - strict/expand/energy モード

# 9. 後処理 (L700-734)
#    - パディング除去
#    - 長さ調整 (リサンプル or トリム)
```

## Model Files

| ファイル | 場所 | 用途 | 備考 |
|----------|------|------|------|
| `hubert_base.pt` | ~/.cache/rcwx/models/ | HuBERT特徴抽出 | RVC WebUI互換 |
| `rmvpe.pt` | ~/.cache/rcwx/models/ | F0抽出 (RMVPE) | 高品質、320ms min |
| `*.pth` | 任意 | RVCモデル | v1/v2対応 |
| `*.index` | モデルと同じディレクトリ | FAISSインデックス | 自動検出 |

**FCPE**: torchfcpeパッケージから動的ロード (100ms min, 低レイテンシ)

## Advanced Features

### Noise Cancellation

ノイズキャンセリング機能の設定:

| 方式 | 説明 | 品質 | 速度 |
|------|------|------|------|
| `auto` | DeepFilterNet利用可能ならML、なければSpectral | 自動 | 自動 |
| `deepfilter` | DeepFilterNet (Facebook AI, 機械学習ベース) | 高 | 中 |
| `spectral` | Spectral Gate (DSP処理) | 中 | 高 |

**注意**: DeepFilterNetはCPU実行のため、RVC推論がXPU/CUDAでもデノイズはCPUで処理される。

### Voice Gate Modes

無声区間の処理方法を制御 (F0ありモデル専用):

| モード | 説明 | 用途 | アルゴリズム |
|--------|------|------|------------|
| `off` | ゲートなし、全音声通過 | 最大品質、ノイズ残る可能性 | - |
| `strict` | F0ベースのみ (破裂音カットの可能性) | クリーン、破裂音に注意 | voiced_mask (f0 > 0) |
| `expand` | 有声区間を拡張して破裂音を含める | **推奨**、自然な会話 | max_pool1d (k=5, 約30ms拡張) |
| `energy` | エネルギー+F0併用 (破裂音も通過) | 最も自然だがノイズ混入の可能性 | voiced OR energy > threshold |

**処理詳細**:

1. **strict**: F0 > 0 のフレームのみ出力、他は無音
2. **expand**: 有声フレームを前後に拡張 (max pooling)、破裂音を含む
3. **energy**: 短時間エネルギーを計算、閾値超えまたは有声なら出力

**energy_threshold**: energyモード時の閾値
- 0.01 = 敏感 (小さい音も通過、ノイズ混入リスク)
- 0.05 = バランス (デフォルト)
- 0.2 = 鈍感 (大きい音のみ、静かな声がカットされる可能性)

**実装場所**: `pipeline/inference.py:633-693`

### Feature Caching

チャンク間の連続性を保つためのHuBERT/F0特徴キャッシング:

**処理詳細**:

1. **HuBERT特徴キャッシング**:
   - 前チャンクの最後10フレーム (50fps @ 100ms) をキャッシュ
   - 現チャンクの最初10フレームとリニアクロスフェード
   - α: 1 (前チャンク) → 0 (現チャンク) でブレンド

2. **F0キャッシング**:
   - 前チャンクの最後20フレーム (100fps @ 200ms) をキャッシュ
   - 両方が有声 (f0 > 0) の場合のみブレンド
   - 片方が無声の場合は現チャンクを使用

**効果**:
- チャンク境界でのピッチジャンプ軽減
- 音質の滑らかな遷移
- 特にF0ありモデルで効果大

**無効化**:
- `use_feature_cache=false` で完全に独立したチャンク処理
- ログファイル分析時やデバッグ時に有用

**実装場所**: `pipeline/inference.py:447-540`

### w-okada Style Processing

低レイテンシと高品質を両立する処理方式 ([w-okada RVC](https://github.com/w-okada/voice-changer)方式):

```
入力チャンク構造: [left_context | main | right_context]
                   ↓
                処理 (RVC推論)
                   ↓
出力: mainのみ保持、contextは破棄
      ↓
   SOLAでクロスフェード
```

**処理詳細**:

1. **入力バッファリング** (ChunkBuffer):
   - `context_samples`: 前チャンクの末尾を左コンテキストとして保持
   - `lookahead_samples`: 先読みバッファ (右コンテキスト用、レイテンシ増加)
   - チャンク取得: `context + main + lookahead` を返す

2. **推論**: 全体 (`context + main + lookahead`) を処理

3. **出力トリミング**:
   - 左コンテキスト分を先頭から削除
   - 右コンテキスト分を末尾から削除
   - `main`部分のみ保持

4. **SOLA Crossfade**:
   - 前チャンクの末尾と現チャンクの先頭をブレンド
   - 相関係数が最大になる位置を探索 (位相整合)
   - RVC WebUIと同じアルゴリズム

**パラメータ**:
- `context_sec`: 左右のコンテキスト長 (0.05s推奨、安定した端処理)
- `lookahead_sec`: 右側の先読み (レイテンシ増加、通常0で十分)
- `crossfade_sec`: チャンク境界のクロスフェード長 (0.05s)
- `extra_sec`: 追加カット (エッジアーティファクト除去用、通常0)
- `use_sola`: 最適クロスフェード位置の自動探索
  - **デフォルト: true** (w-okada対応モード実装済み)
  - 左コンテキストで最適オフセットを探索
  - 位相整合されたクロスフェードで滑らかな音質

**利点**:
- エッジ効果軽減: コンテキストにより端での処理が安定
- 低レイテンシ: 先読み不要 (`lookahead_sec=0`)
- 高品質: SOLAにより位相整合されたクロスフェード

### F0 Extraction Methods

| 方式 | 最小チャンク | 品質 | 速度 | 説明 |
|------|-------------|------|------|------|
| `rmvpe` | 320ms | 高 | 中 | 安定した高精度F0抽出 |
| `fcpe` | 100ms | 中 | 高 | 低レイテンシF0抽出 (要torchfcpe) |
| `none` | - | - | 最高 | F0なしモデル専用 |

**FCPE**: `pip install torchfcpe` で利用可能、低遅延モード向け

## Troubleshooting

### 自動フィードバック検出

RCWXは入出力の相関係数を監視し、フィードバックループを自動検出します:

- **検出条件**: 相関係数 > 0.3
- **警告メッセージ**: ログおよびGUIに表示
- **推奨対処**: `rcwx diagnose` で設定確認

### フィードバック/エコー問題 (ピッチが蓄積する)

**症状**: ピッチシフト+5に設定しているのに、時間とともに+10, +15...と上がっていく

**原因**:
1. 同じオーディオインターフェース（例: High Definition Audio Device）を入出力に使用
2. Windowsの「このデバイスを聴く」機能が有効
3. Stereo Mix/ループバックデバイスを入力に選択

**解決策**:
1. **診断ツール実行**: `rcwx diagnose` で現在の設定を確認
2. **異なるインターフェースを使用**:
   - 入力: USBマイク（例: Fifine K420）
   - 出力: オンボードスピーカー/ヘッドホン
3. **Windowsの設定確認**:
   - Win + R → `mmsys.cpl`
   - 録音タブ → マイクのプロパティ → 「聴く」タブ
   - 「このデバイスを聴く」のチェックを外す

### XPUが認識されない

```powershell
# 確認
uv run python -c "import torch; print(torch.__version__, torch.xpu.is_available())"

# 期待: 2.x.x+xpu True
# NG: 2.x.x+cpu False
```

**解決策**: `uv sync --reinstall` でXPU版PyTorchを再インストール

### 推論が遅い

1. **ログ確認**:
   ```powershell
   uv run rcwx --verbose
   uv run rcwx logs --tail 100
   ```
   `[INFER]` タグで推論時間確認 (目安: <60ms)

2. **デノイズ無効化でテスト**:
   - MLデノイズ: CPU実行で+10-30ms
   - 無効化して速度改善するか確認

3. **F0方式変更**:
   - RMVPE → FCPE (低レイテンシ)
   - F0あり → F0なし (最速)

4. **チャンクサイズ調整**:
   - 大きくする: レイテンシ増加、安定性向上
   - 小さくする: レイテンシ減少、処理負荷増加

5. **XPU/CUDA確認**:
   ```powershell
   uv run python -c "import torch; print(torch.__version__, torch.xpu.is_available())"
   # 期待: 2.x.x+xpu True
   ```

### 音質が悪い

1. **Voice Gate確認**:
   - `strict` → `expand` (破裂音復活)
   - `expand` → `energy` (より自然)
   - `off` (ゲートなし、ノイズは増加)

2. **Feature Cache有効化**:
   - チャンク境界の不連続性軽減
   - デフォルトで有効

3. **SOLA確認**:
   - 位相整合されたクロスフェード
   - デフォルトで有効

4. **チャンクサイズ**:
   - 小さすぎる (<200ms): エッジアーティファクト増加
   - 推奨: 350ms (RMVPE), 150ms (FCPE)

5. **Indexファイル使用**:
   - モデルと同じディレクトリに `.index` ファイル配置
   - GUIでIndex率 0.5-0.75 に設定

### レイテンシが高い

**目標レイテンシ**:
- RMVPE: 150-250ms
- FCPE: 80-150ms
- F0なし: 50-100ms

**最適化手順**:

1. **F0方式変更** (最大効果):
   - RMVPE (320ms min) → FCPE (100ms min)
   - または F0なしモードに変更

2. **チャンクサイズ削減**:
   - RMVPE: 350ms → 320ms (最小値)
   - FCPE: 150ms → 100ms (最小値)

3. **Prebuffer削減**:
   - デフォルト: 1チャンク
   - 0に設定: 最低レイテンシだがアンダーラン増加

4. **Lookahead無効化** (デフォルトで0):
   - `lookahead_sec=0` 確認

5. **デノイズ無効化**:
   - DeepFilterNet: +10-30ms
   - Spectral: +1-3ms

**レイテンシ計算**:
```
総レイテンシ = chunk_sec + inference_ms + buffer_ms + lookahead_sec
             = 350ms + 50ms + 50ms + 0ms
             = 450ms (RMVPE典型値)
```

## Configuration Details (config.py)

### AudioConfig

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `input_device_name` | None | 入力デバイス名 (自動検出) |
| `output_device_name` | None | 出力デバイス名 (自動検出) |
| `sample_rate` | 16000 | 処理サンプルレート |
| `output_sample_rate` | 48000 | 出力サンプルレート |
| `chunk_sec` | 0.10 | チャンクサイズ (秒、FCPE公式最小値) |
| `crossfade_sec` | 0.05 | クロスフェード長 (秒) |
| `input_gain_db` | 0.0 | 入力ゲイン (dB) |
| `prebuffer_chunks` | 1 | プリバッファチャンク数 |
| `buffer_margin` | 0.3 | バッファマージン (タイト) |

### InferenceConfig

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `pitch_shift` | 0 | ピッチシフト (半音) |
| `use_f0` | true | F0使用 |
| `f0_method` | fcpe | F0抽出方法 (fcpe=低レイテンシ, rmvpe=高品質) |
| `use_index` | false | FAISSインデックス使用 |
| `index_ratio` | 0.5 | インデックス比率 |
| `use_compile` | false | torch.compile使用 (Windows XPU: 安定性のため無効) |
| `use_parallel_extraction` | true | 並列HuBERT+F0抽出 (10-20%高速化) |
| `voice_gate_mode` | expand | ボイスゲートモード |
| `energy_threshold` | 0.05 | エネルギー閾値 |
| `use_feature_cache` | true | 特徴キャッシング |
| `context_sec` | 0.05 | コンテキスト長 |
| `lookahead_sec` | 0.0 | 先読み長 |
| `extra_sec` | 0.0 | 追加カット |
| `crossfade_sec` | 0.05 | クロスフェード長 |
| `use_sola` | true | SOLA使用 (w-okada対応) |

### DenoiseConfig (InferenceConfig内)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `enabled` | false | ノイズ除去有効化 |
| `method` | auto | 方式 (auto/deepfilter/spectral) |
| `threshold_db` | 6.0 | Spectral Gate閾値 |
| `reduction_db` | -24.0 | Spectral Gate削減量 |

### 設定ファイルの場所

```
~/.config/rcwx/config.json
```

設定はGUI終了時に自動保存され、次回起動時にロードされます。

## Diagnostic Tool (diagnose.py)

`rcwx diagnose` コマンドは以下をチェックします:

1. **オーディオデバイス一覧**
   - 入力デバイス (マイク)
   - 出力デバイス (スピーカー/ヘッドホン)
   - ループバックデバイスの警告

2. **同一インターフェース検出**
   - 入出力が同じ "High Definition Audio" を使用している場合に警告

3. **仮想オーディオデバイス検出**
   - VoiceMeeter, VB-Cable等の検出
   - ルーティング設定の確認推奨

4. **Windows設定チェック手順**
   - 「このデバイスを聴く」の無効化方法
   - Stereo Mixの無効化方法

5. **PyTorchデバイス情報**
   - XPU/CUDA/CPU利用可能状況
   - デバイス名と数

6. **フィードバックテスト手順**
   - ピッチシフト+5で開始
   - ピッチが蓄積するか確認

## FAQ

### Q1: XPU版とCUDA版のどちらが速いですか?

**A**: 同クラスのGPUであればほぼ同等です。Intel Arc A770 ≈ RTX 3060程度のパフォーマンス。

### Q2: CPUのみで動作しますか?

**A**: 動作しますが、推論が遅くリアルタイムは厳しいです。バッチ処理 (`rcwx run`) は可能。

### Q3: RVC WebUIのモデルは使えますか?

**A**: はい、v1/v2両方対応。`.pth`ファイルと`.index`ファイルを同じディレクトリに配置してください。

### Q4: ピッチが蓄積していく問題の原因は?

**A**: オーディオフィードバックループです。`rcwx diagnose` で診断し、入出力デバイスを確認してください。

### Q5: FCPEとRMVPEの違いは?

**A**:
- **RMVPE**: 高品質、320ms最小チャンク、レイテンシ150-250ms
- **FCPE**: 低レイテンシ、100ms最小チャンク、レイテンシ80-150ms、品質やや劣る

### Q6: Voice Gateはどのモードが推奨ですか?

**A**: **expand**が推奨。破裂音を含みつつノイズを抑制します。音質優先なら`off`、ノイズ優先なら`strict`。

### Q7: Indexファイルは必須ですか?

**A**: 任意です。使用すると音質が向上する場合があります。Index率 0.5-0.75 が推奨。

### Q8: torch.compileが無効化されるのはなぜ?

**A**: WindowsではTritonが非対応のため。Linux環境では自動有効化され、20-30%高速化します。

### Q9: DeepFilterNetとSpectral Gateの使い分けは?

**A**:
- **DeepFilterNet**: 高品質、人声保持、CPU実行で遅い
- **Spectral Gate**: 軽量、高速、単純な周波数処理
- 静かな環境なら無効推奨

### Q10: Feature Cacheを無効にするメリットは?

**A**: デバッグやログ分析時に、各チャンクが完全に独立するため原因特定しやすくなります。通常は有効推奨。

### Q11: SOLAを無効にするとどうなりますか?

**A**: 単純な線形クロスフェードになり、位相不整合によるアーティファクトが増える可能性があります。通常は有効推奨。

### Q12: チャンクサイズの推奨値は?

**A**:
- **RMVPE**: 350ms (最小320ms)
- **FCPE**: 150ms (最小100ms)
- **F0なし**: 100-200ms

小さくするとレイテンシ低下、大きくすると安定性向上。

### Q13: バッファアンダーランが頻発する

**A**:
1. チャンクサイズを増やす (350ms → 400ms)
2. `buffer_margin` を増やす (0.5 → 1.0)
3. F0方式変更 (rmvpe → fcpe または none)
4. デノイズ無効化

### Q14: ログファイルの見方は?

**A**:
```
[INFER] Chunk #1: in=16800, out=14000, infer=45ms, latency=180ms
```
- `infer`: 推論時間 (目安: <60ms)
- `latency`: 総レイテンシ
- `under/over`: バッファアンダーラン/オーバーラン回数

### Q15: モデルのバージョン確認方法は?

**A**:
```powershell
uv run rcwx info model.pth
```
v1/v2、F0あり/なし、スピーカー数等が表示されます。

## References

- [RVC WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- [PyTorch XPU](https://pytorch.org/docs/stable/notes/get_start_xpu.html)
- [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)
- [Facebook Denoiser](https://github.com/facebookresearch/denoiser)
- [ContentVec](https://github.com/auspicious3000/contentvec)
- [FCPE (torchfcpe)](https://github.com/CNChTu/FCPE)
- [w-okada Voice Changer](https://github.com/w-okada/voice-changer)
