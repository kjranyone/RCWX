# RCWX

RVC Real-time Voice Changer on Intel Arc (XPU)

## Features

- **Intel Arc GPU対応** - PyTorch XPU による高速推論
- **XPU Accelerator Graph** - HuBERTと定常状態のRVC合成をcapture/replayしてCPU起動オーバーヘッドを削減
- **RVC v1/v2両対応** - 256次元・768次元特徴量モデルに対応
- **F0あり/なしモデル対応** - RMVPE F0抽出または低遅延モード
- **リアルタイム変換** - クロスフェード処理による低遅延・高品質変換
- **ノイズキャンセリング** - ML (Facebook Denoiser, オプション) / Spectral Gate 切替可能
- **ASIO対応** - プロオーディオインターフェース対応（WASAPI/DirectSound/MMEも利用可能）
- **レベルモニター** - 出力レベルメーター＋レベル調整（メインパネル）、入力レベル常時モニター（排他ASIOデバイスでは変換中にパイプラインからタップ表示）
- **CustomTkinter GUI** - モダンなダークテーマUI
- **フルスクラッチ実装** - rvc-python等の依存なし

## Requirements

- Windows 10/11
- Python 3.11 or 3.12
- Intel Arc GPU (A770, A750, B580, etc.) または CUDA GPU
- [uv](https://github.com/astral-sh/uv) パッケージマネージャ

## Installation

### Intel Arc GPU (XPU) の場合

```powershell
git clone https://github.com/grand2-products/rcwx.git
cd rcwx

# PyTorch XPU版を含む全依存関係をインストール
uv sync

# XPU確認
uv run python -c "import torch; print(f'XPU: {torch.xpu.is_available()}')"
```

> **Note**: `pyproject.toml` で PyTorch XPU インデックスが設定済みのため、`uv sync` だけで XPU 版がインストールされます。

### NVIDIA GPU (CUDA) の場合

pyproject.toml を編集して CUDA インデックスに変更:

```toml
[[tool.uv.index]]
name = "pytorch-xpu"
url = "https://download.pytorch.org/whl/cu124"  # XPU → CUDA に変更
explicit = true
```

```powershell
uv sync

# CUDA確認
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### CPU のみの場合

pyproject.toml の `[tool.uv.sources]` セクションを削除またはコメントアウト:

```powershell
uv sync

# CPU版が使用される
uv run python -c "import torch; print(torch.__version__)"
```

## Quick Start

```powershell
# 1. 依存関係インストール
uv sync

# 2. (オプション) ML Denoiser をインストール (CC BY-NC 4.0)
uv sync --extra ml-denoise

# 3. 必要モデル (HuBERT, RMVPE) のダウンロード
uv run rcwx download

# 4. XPUとAccelerator Graphの確認
uv run rcwx diagnose

# 5. GUI起動
uv run rcwx
```

### 対話式ランチャー (rcwx.ps1)

配布・環境確認向けに、対話メニュー付きのランチャーを同梱しています。uv / PyTorch(XPU) / 必須モデル / ML デノイザの状態をチェックし、GUI起動・デバイス一覧・モデルDL・診断・ログ表示などをメニューから実行できます。

```powershell
.\rcwx.ps1                    # 環境チェック → 対話メニュー
.\rcwx.ps1 gui                # メニューを介さず GUI を直接起動
.\rcwx.ps1 -Denoise gui       # ML デノイザ有効化 (--extra ml-denoise) で起動
```

> ML デノイザ（任意依存 `ml-denoise` extra）は既定で未インストール。`uv run`/`uv sync` を extra なしで実行すると prune されるため、ランチャーは有効時に全ての `uv run` を自動で `--extra ml-denoise` 付きにします。メニューから随時インストールも可能です。

**デフォルト設定** (最適化済み):
- F0方式: **RMVPE** (高品質)
- チャンクサイズ: **300ms**
- ノイズ除去: **ML** (Facebook Denoiser, 要 `--extra ml-denoise`) / Spectral (標準)
- FAISS インデックス: **有効** (ratio=0.15)
- 詳細は [Inference Settings](#inference-settings) 参照

## XPU Accelerator Graph

PyTorch `2.13.0+xpu` では、XPU利用時にAccelerator Graphが自動的に有効になります。HuBERT特徴抽出と、入力shapeが固定された定常状態のRVC Synthesizerが対象です。GUI側の設定操作は必要ありません。

- 変換開始前のウォームアップでHuBERT履歴を満たし、定常状態のSynthesizer Graphを1回capture
- 実ストリーム開始前に音声履歴とリサンプラー状態をリセットし、capture済みGraphだけを保持
- Graph API非対応またはcapture失敗時はeager推論へ自動フォールバック
- `use_compile=true` のSynthesizerはAccelerator Graphを併用せず、従来のcompile経路を使用

利用可否は診断コマンドで確認できます。

```powershell
uv run rcwx diagnose
# [OK] XPU Accelerator Graph: True
```

ドライバーやモデル固有の問題を切り分ける場合は、環境変数で無効化できます。

```powershell
$env:RCWX_ACCELERATOR_GRAPH = "0"
uv run rcwx

# 次回以降、自動判定へ戻す
Remove-Item Env:RCWX_ACCELERATOR_GRAPH
```

## RVC Models Directory

RVCモデル（`.pth`）を格納したディレクトリを指定すると、ドロップダウンに自動列挙されます。

```
rvc_models/
├── ModelA/
│   ├── ModelA.pth
│   └── ModelA.index
├── ModelB/
│   ├── voice.pth
│   └── voice.index
└── ModelC.pth            ← ルート直下も検出
```

- **設定方法**: 詳細設定タブ →「RVCモデルディレクトリ」→ 参照ボタンでフォルダ選択
- **表示名**: サブフォルダ内の `.pth` はフォルダ名、ルート直下はファイル名（拡張子なし）
- **更新ボタン**: モデル追加後にドロップダウンを再スキャン
- **「開く...」ボタン**: ディレクトリ外のモデルも従来通り選択可能
- 設定は `~/.config/rcwx/config.json` に保存され、次回起動時に自動復元

## CLI Commands

```powershell
# デバイス一覧
uv run rcwx devices

# ファイル変換
uv run rcwx run input.wav model.pth -o output.wav --pitch 5

# モデル情報表示
uv run rcwx info model.pth
```

## GUI Layout

3つのタブ（**メイン** / **オーディオ** / **詳細設定**）で構成されます。

### メインタブ（2カラム）

```
┌──────────────────────────────────────────────────────────────────────┐
│  RCWX - RVC Voice Changer                                  [─][□][×] │
├──────────────────────────────────────────────────────────────────────┤
│  [ メイン ] [ オーディオ ] [ 詳細設定 ]                                │
├──────────────────────────────────────────────────────────────────────┤
│  ┌─ 左カラム ───────────────┐   ┌─ 右カラム ─────────────────┐        │
│  │ ■ モデル選択             │   │ ■ オーディオデバイス       │        │
│  │ [▼ ModelA] [更新][開く]  │   │ 🎤 入力  🔊 出力  ⚡ XPU   │        │
│  │ 状態: F0あり(v2)|index有 │   │                            │        │
│  │                          │   │ ■ オーディオテスト         │        │
│  │ ■ ピッチ制御             │   │ [🎤 テスト(3秒録音→再生)]  │        │
│  │ -24 ──●── +24   +5半音   │   │ [ ] WAVファイルをループ    │        │
│  │ F0: (●)RMVPE ( )FCPE     │   │                            │        │
│  │ Moe / F0 / noise         │   │ ┌────────────────────────┐ │        │
│  │                          │   │ │        ▶ 開始         │ │        │
│  │ ■ FAISSインデックス      │   │ └────────────────────────┘ │        │
│  │ [✓]有効   ratio 0.15     │   │ 出力レベル ▮▮▮▮▮▯▯  -8 dB  │        │
│  │                          │   │ レベル調整 ───●───  +0 dB  │        │
│  │ ■ ノイズキャンセリング   │   │                            │        │
│  │ [✓]有効   方式[ml ▼]     │   │                            │        │
│  │                          │   │                            │        │
│  │ ■ ボイスゲート           │   │                            │        │
│  │ [off ▼]                  │   │                            │        │
│  └──────────────────────────┘   └────────────────────────────┘        │
├──────────────────────────────────────────────────────────────────────┤
│ デバイス: Intel Arc A770 | レイテンシ: 145ms | 推論: 45ms             │
└──────────────────────────────────────────────────────────────────────┘
```

### オーディオタブ

入出力デバイスの選択（ホストAPIフィルタ: WASAPI/ASIO/DirectSound/MME）、チャンネル選択、**入力ゲイン＋入力レベルメーター**、ポストプロセッシング（トレブルブースト／ノーマライザ／リミッター）、レイテンシ設定を配置。

### 詳細設定タブ

推論デバイス（xpu/cuda/cpu）、dtype、RVCモデルディレクトリ、モデル配置先などの設定。

### レベルモニター

| 種別 | 場所 | 挙動 |
|------|------|------|
| **出力レベル** | メインパネル（開始ボタン直下） | 変換出力のRMSメーター（スケール **-24〜0 dB**）＋レベル調整スライダー（**-12〜+12 dB**、ポストプロセッシング後・最終クリップ前に適用） |
| **入力レベル** | オーディオタブ | **常時モニター**。非ASIOデバイスは専用ストリームで常時計測。排他ASIOデバイスは別ストリームを開けない（PortAudio -9985）ため、**変換の実行中**にパイプラインの入力からタップして表示 |

## Noise Cancellation

騒音環境でのマイク入力を改善するノイズキャンセリング機能:

| 方式 | 説明 | 用途 |
|------|------|------|
| `auto` | ML利用可能ならML、なければSpectral | 推奨 |
| `ml` | Facebook Denoiser (PyTorch) | 高品質、人声保持 |
| `spectral` | Spectral Gate (DSP) | 軽量、低遅延 |

- **ML方式**: 機械学習ベースで人間の声を認識・保持しながらノイズを除去（要 `uv sync --extra ml-denoise`、CC BY-NC 4.0）
- **Spectral方式**: 周波数スペクトルの閾値処理による従来型ノイズ除去（標準同梱）

## Inference Settings

推論パイプラインの各パラメータの解説です。設定は `~/.config/rcwx/config.json` に永続化されます。

### 音声合成

| パラメータ | デフォルト | 範囲 | 説明 |
|-----------|-----------|------|------|
| `pitch_shift` | 0 | -24〜+24 | ピッチシフト（半音単位） |
| `noise_scale` | 0.45 | 0.0〜1.0 | VITS合成器のVAEノイズ量。0.0=完全決定的、0.66=RVC原典デフォルト。低いほどクリーン、高いほど自然なゆらぎが出る |
| `moe_boost` | 0.45 | 0.0〜1.0 | 萌え声スタイル強度。詳細は [Moe Boost](#moe-boost) 参照 |

### F0（基本周波数）制御

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `f0_method` | `rmvpe` | F0抽出方式。`rmvpe`=高品質（320ms最小チャンク）、`fcpe`=低遅延（100ms最小チャンク） |
| `f0_lowpass_cutoff_hz` | 16.0 | F0ローパスフィルタのカットオフ周波数（Hz）。100fps F0に対してButterworth 2次フィルタを適用。低い=滑らか、高い=ピッチの細かい変化を保持 |
| `enable_octave_flip_suppress` | true | F0抽出のオクターブ誤検出（±1オクターブのフレーム間ジャンプ）を自動補正。隣接フレーム比が2.0±0.16の場合にトリガー |
| `enable_f0_slew_limit` | true | フレーム間のF0変化量を制限し、ピッチの急激な飛びを抑制 |
| `f0_slew_max_step_st` | 3.6 | スルーリミッターの最大ステップ幅（半音/フレーム）。各F0値を前フレームの `2^(±step/12)` 倍以内にクランプ |

### FAISS インデックス

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `use_index` | true | FAISSベクトル検索による声質補正の有効化 |
| `index_ratio` | 0.15 | HuBERT特徴量とインデックス検索結果のブレンド比率。0.0=HuBERTのみ、1.0=インデックスのみ。`blended = ratio × retrieved + (1-ratio) × original` |
| `index_k` | 4 | 近傍検索数。4=高速、8=高品質。L2距離の逆二乗で重み付け平均 |

### チャンク処理・クロスフェード

リアルタイム処理におけるチャンク境界の連続性を制御するパラメータ群です。

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `overlap_sec` | 0.20 | HuBERT入力に付加する音声オーバーラップ長。20ms境界（HuBERT 320サンプルホップ）に丸められる。長いほどHuBERTのコンテキストが豊かになり特徴抽出の精度が上がるが、計算量が増える |
| `crossfade_sec` | 0.08 | SOLAクロスフェード長。前チャンクの末尾と次チャンクの先頭をHann窓でブレンドする区間。長いほど継ぎ目が滑らかだが遅延が増加 |
| `use_sola` | true | SOLA（Synchronized Overlap-Add）の有効化。相互相関で最適な接合位置を探索する |
| `sola_search_ms` | 10.0 | SOLA探索窓（ms）。±この範囲で最良の接合点を探す。広いほど良い位置が見つかるが計算コストが増える |

**処理フロー**: `overlap_sec` の音声を入力に付加 → HuBERT+F0抽出 → 合成 → `crossfade_sec` 区間で前チャンクとSOLAブレンド（`sola_search_ms` 窓内で最適位置探索）

### ボイスゲート

無声区間の出力制御モードです。

| モード | 動作 |
|--------|------|
| `off` | ゲートなし。全音声を通過（デフォルト） |
| `strict` | F0が有声（>0）の区間のみ出力。破裂音・気息音がカットされる |
| `expand` | strict の有声区間を前後~30msに拡張し、破裂音を保持 |
| `energy` | F0有声 **または** フレームエネルギーが `energy_threshold` 以上の区間を出力 |

`energy_threshold`（デフォルト 0.2）: energyモードの正規化RMSエネルギー閾値。低い=感度高（微弱音も通すがノイズも通りやすい）、高い=感度低（ノイズ除去に強いが柔らかい音がカットされやすい）

## Moe Boost

低音域の男声F0を明るいアニメ声風の抑揚に変換するF0スタイリング機能。RVCモデルの再学習なしにリアルタイムで動作する。

### 概要

`moe_boost`（強度 `s`、0.0〜1.0）は以下の4段階でF0コンター（ピッチ曲線）を変形する:

1. **短ギャップ補間** — 無声フレームの短い隙間を線形補間で埋め、ガサつき（frame flicker）を軽減
2. **レジスタシフト** — 声の基底音域を目標メディアンに向けて引き上げ
3. **コンター整形** — 上昇アクセントを強調、下降ディップを抑制し「可愛い」抑揚パターンを生成
4. **フロア制約** — 胸声域の低すぎるディップを防止

### パラメータ（強度 s による自動導出）

| 要素 | 式 | s=0 | s=0.45 | s=1.0 |
|------|-----|-----|--------|-------|
| 目標メディアン | `165 + 55s` Hz | 165Hz | 190Hz | 220Hz |
| 最大上方シフト | `1.5 + 4.5s` st | +1.5st | +3.5st | +6st |
| 上昇ゲイン | `1 + 0.45s` | 1.0x | 1.20x | 1.45x |
| 下降ゲイン | `1 - 0.25s` | 1.0x | 0.89x | 0.75x |
| フレーズバイアス | `0.10 + 0.45s` st | +0.1st | +0.3st | +0.55st |
| ギャップ補間 | `2 + 4s` frames | 20ms | 40ms | 60ms |
| トレンド窓 | `odd(7 + 14s)` | 7 | 13 | 21 |
| フロア（相対） | `target × (0.55 + 0.08s)` Hz | 91Hz | 111Hz | 139Hz |
| フロア（絶対） | `85 + 45s` Hz | 85Hz | 105Hz | 130Hz |

### 処理フロー

```
入力F0 (100fps)
  → 短ギャップ補間 (20-80ms未満の無声区間を線形補間)
  → log2空間に変換
  → トレンド抽出 (avg_pool1d, 窓幅 7-21)
  → デビエーション = 12 × (log2_f0 - trend) semitones
  → 非対称ゲイン (上昇: 1.0-1.8x, 下降: 1.0-0.7x)
  → ソフト飽和 (極端なオーバーシュート防止)
  → レジスタシフト + フレーズバイアス加算
  → 2^(log2) で Hz に復元
  → フロア適用 [85-260Hz] + シーリング [940Hz]
  → 後段フィルタ (lowpass, octave suppress, slew limit)
```

### CLI

```powershell
uv run rcwx run input.wav model.pth -o out.wav --pitch 6 --moe-boost 0.45
```

### 制約

- F0コンターの変形のみ。フォルマント（声道共鳴）のシフトは行わない
- 効果は入力声の音域に依存: 男声（100-160Hz）で最も効果的、女声（220Hz+）ではほぼ変化なし
- 強度1.0では抑揚が不自然になる場合がある。推奨範囲は **0.40〜0.80**

## Project Structure

```
rcwx/
├── accelerator_graph.py  # XPU/CUDA固定shape Graphキャッシュ
├── config.py              # 設定管理 (JSON永続化)
├── device.py              # デバイス選択 (xpu/cuda/cpu)
├── downloader.py          # HuggingFace モデルダウンロード
├── cli.py                 # CLI エントリポイント
├── diagnose.py            # フィードバック診断
├── audio/
│   ├── input.py           # マイク入力 (sounddevice)
│   ├── output.py          # 音声出力
│   ├── buffer.py          # RingOutputBuffer
│   ├── resample.py        # StatefulResampler
│   ├── sola.py            # SOLA クロスフェード
│   ├── denoise.py         # ノイズキャンセリング (ML/Spectral)
│   └── wav_input.py       # WAVファイルループ入力
├── models/
│   ├── hubert_loader.py   # ContentVec 特徴抽出 (transformers)
│   ├── rmvpe.py           # RMVPE F0抽出
│   ├── fcpe.py            # FCPE F0抽出 (低レイテンシ)
│   ├── swiftf0.py         # SwiftF0 F0抽出 (ONNX/CPU)
│   ├── synthesizer.py     # RVC合成器
│   └── infer_pack/        # RVC コアモジュール
├── pipeline/
│   ├── inference.py       # infer / infer_streaming
│   └── realtime_unified.py # リアルタイム処理
└── gui/
    ├── app.py             # メインアプリケーション
    ├── realtime_controller.py
    ├── model_loader.py    # 非同期モデルロード
    ├── file_converter.py  # ファイル変換テスト
    └── widgets/           # UIコンポーネント
```

## Latency

### レイテンシモード

レイテンシ設定には`Balanced`、`Aggressive`、`Sub-100`、`Frontier`があります。モード変更時はI/Oストリームと固定shape Graphを再ウォームアップします。

| モード | SOLAクロスフェード | 持続リングfloor | 非ASIOコールバック | 用途 |
|--------|-------------------|-----------------|----------------------|------|
| Balanced | chunkの25%、最大80ms | プリバッファを維持 | chunk / 4 | 安定性優先 |
| Aggressive | chunkの10%、最大20ms | 0.75 hopでtrim、0.25 hopへ復帰 | 最大10ms | 低遅延優先 |
| Sub-100 | 10ms | 初期1 hop、定常20-35msを適応維持 | 最大5ms | ASIO + SwiftF0向け |
| Frontier | 10ms | 初期1 hop、定常10-17.5msを適応維持 | 最大2.5ms | 実験的な20ms deadline |

Aggressiveでも開始時に最初の生成チャンクを待つため、起動直後の無音アンダーランは発生させません。持続的なバッファ滞留だけを2 hopの観測窓で判定し、通常のチャンク内burst/drainはtrim対象から除外します。

`Sub-100`はF0方式の下限へchunkを自動設定します。SwiftF0/Noneは40ms、FCPEは100ms、RMVPEは320msです。開始時に2 hopを確保し、最初の20 hopは40msのfloorを維持します。その後は直近推論の`p99 - p50 + callback`から20-35msのjitter guardを選びます。アンダーラン時は2 hopを再確保して、連続した音切れを防ぎます。

`Frontier`はSwiftF0/Noneのchunk下限を20msまで下げます。開始時とアンダーラン復旧時は3 hopを確保し、20 hopの統計が揃った後は同じ適応式で10-17.5msのjitter guardへ縮めます。FCPEとRMVPEは入力要件のため、それぞれ100msと320msが下限です。

Sub-100/Frontier実行中は保存設定を変更せず、HuBERT contextを最大560ms、SwiftF0 contextを最大100msへ一時的に短縮します。また、モデルと出力のsample rateが異なる場合は、Synthesizer出力をCPUへ戻す前にtorchaudio sinc resampleをXPU Graphで実行します。ASIOの実レートが設定値と異なる場合も、音声開始前に実レート用Graphを再ウォームアップします。deadlineを守るためDenoiserは実行中だけバイパスされ、Balanced/Aggressiveへ戻すと保存済みの品質設定が再び有効になります。

Sub-100/Frontierのruntime warmupでは、FAISS `IndexIVFFlat`（L2、`nprobe=1`）のinverted listと再構築特徴をXPUへ常駐させ、coarse centroid選択、候補L2検索、top-k加重平均をAccelerator Graphで実行します。これによりHuBERT特徴をFAISS検索のためだけにCPUへ同期する経路を除去します。最大list長が256を超えるindexや未対応形式では、自動的に既存のCPU FAISSへ戻ります。XPU側ではindex特徴をFP16/BF16で保持するため、158k×768のindexで約240MBの追加VRAMを使用します。Balanced/Aggressiveやファイル変換では構築しません。

Intel Arc B570、SwiftF0、FAISS ratio 0.45、40ms hopの実モデル測定では、本番同等のGraphウォームアップとGUIのSOLA/decoder余白を含む処理時間がp50 14.5ms、p95 16.4ms、p99 18.8msでした（40ms deadline miss 0/50）。20msの定常jitter guardとASIO入出力約12msを含む通常時のE2E目安は90-100msです。ドライバー、モデル、OSスケジューリング、同時GPU負荷による単発スパイクは残ります。

Frontier第1段のXPU IVF検索では、同じArc B570と158,193件のindexで検索単体が中央値0.62ms、RVCストリーミング全経路のp50がCPU FAISSの13.56msから12.59msへ短縮しました。全158,193候補を保持した比較で最終波形相関は0.999956、40ms deadline missは0/40、IVF Graphはcapture 1回、replay 140回、fallback 0でした。

Frontier 20ms hopでは、SOLAが必要とする合成末尾を`crossfade + search + decoder overlap`へ修正し、従来の二重search余白を除去しました。同じArc B570の定常測定はp50 11.87ms、p95 14.51ms、p99 17.37ms、20ms deadline miss 0/60です。全SOLA出力は960 samplesで一致し、HuBERT/Synthesizer/IVF Graphのfallbackは0でした。これは実機のdeadline成立を示す値であり、OSやドライバーの単発スパイクまで保証するものではありません。

Frontierのホットパス監査では、音声推論以外にも毎hopのXPU timing event、ThreadPool生成・破棄、GUI/GPUメモリtelemetry、未使用feature cache clone、SOLA境界ログが残っていました。診断を10 hopごと、GUI telemetryを10Hzへ間引き、SwiftF0 workerを永続化してHuBERTを推論threadから直接dispatchします。同一60-hop ablationではp50 12.51msから9.44ms、p95 15.98msから11.70ms、p99 16.45msから12.14msへ短縮し、20ms deadline missは0でした。音声tensor、Graph shape、SOLA出力長は変更していません。

ライブのレイテンシ表示は`1 hop + 推論 + 持続リングfloor + 出力キュー + SOLA`です。100ms Aggressiveでは、通常の100ms出力チャンクがリング内に残っているだけの状態を追加レイテンシとして二重計上しません。

推論表示には直近256 hopのp95も表示します。ログの100 hopごとの`p50` / `p95` / `p99` / `deadline_miss`で、短いhopを継続的に処理できているか確認できます。

**最適化済み** (2026-01-31):

| F0方式 | チャンクサイズ | 処理時間 | プリバッファ | 総レイテンシ | RTF | 品質 |
|--------|--------------|---------|------------|------------|-----|------|
| **FCPE** (推奨) | **150ms** | **90ms** | **150ms** | **~385ms** | **0.56x** | ✅ 高品質 |
| RMVPE | 250ms | 135ms | 250ms | ~530ms | 0.54x | ✅ 最高品質 |
| F0なし | 100ms | 60ms | 100ms | ~260ms | 0.60x | ⚠️ ピッチシフト不可 |

**品質検証済み**（バッチ vs ストリーミング比較）:
- Correlation: ≥ 0.94
- MAE: ≤ 0.02
- Buffer underruns: 0（音切れなし）

### 最適化の成果

1. **FCPE NaN問題解消** - F0抽出のNaN値を自動的に0（無声）に置き換え
2. **Buffer underrun完全解消** - prebuffer_chunks=1で安定動作
3. **レイテンシ27%削減** - RMVPE 530ms → FCPE 385ms（-145ms）
4. **品質維持** - FCPE（Correlation 0.945）≈ RMVPE（Correlation 0.948）
5. **処理速度37%向上** - FCPEはRMVPEより高速（RTF 0.56x vs 0.54x）

### Accelerator Graph実測

Intel Arc B570、PyTorch `2.13.0+xpu`、固定shapeのRVC v2 Synthesizerでの測定値です。モデルやチャンク設定によって結果は変動します。

| 経路 | eager中央値 | Graph中央値 | 削減率 |
|------|------------:|------------:|-------:|
| RVC Synthesizerコア | 18.0ms | 4.5ms | 約75% |

Graph定常時のストリーミング推論全体は中央値約18.6msでした。`noise_scale=0` の決定論的条件ではeager出力とのRMSEと最大絶対誤差はいずれも0です。

## Supported Models

- RVC v1 (256-dim HuBERT features)
- RVC v2 (768-dim HuBERT features)
- F0モデル (NSF decoder)
- No-F0モデル (standard decoder)

モデルは [RVC-Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) で作成したものを使用できます。

## Configuration (pyproject.toml)

PyTorch XPU 版は `[tool.uv]` セクションで設定:

```toml
[tool.uv]
# Windows限定 (triton-xpuはLinux専用)
environments = ["sys_platform == 'win32'"]

# triton-xpu依存を除外
override-dependencies = [
    "triton-xpu; sys_platform == 'linux'",
    "pytorch-triton-xpu; sys_platform == 'linux'",
]

[tool.uv.sources]
torch = { index = "pytorch-xpu" }

# TorchAudio 2.11 is installed from PyPI and supports future Torch releases.

[[tool.uv.index]]
name = "pytorch-xpu"
url = "https://download.pytorch.org/whl/xpu"
explicit = true
```

## Testing

```powershell
# 主なテスト
uv run python tests/integration/test_diagnostic.py
uv run python tests/integration/test_infer_streaming.py
uv run python tests/integration/test_moe_f0_processing.py
uv run python tests/crossfade/test_sola_compensation.py
uv run python tests/models/test_inference.py
uv run python tests/models/test_rmvpe.py
uv run python tests/models/test_accelerator_graph.py
uv run python tests/models/test_synthesizer_graph.py
uv run python tests/models/test_runtime_graph_warmup.py
```

## Development

```powershell
# 開発用依存関係をインストール
uv sync --extra dev

# Lint
ruff check rcwx

# Format
ruff format rcwx
```

## Troubleshooting

### XPU が認識されない

```powershell
# PyTorch バージョン確認
uv run python -c "import torch; print(torch.__version__)"
# → 2.13.0+xpu のように +xpu が付いていることを確認

# +cpu の場合は uv.lock を再生成
del uv.lock
uv sync
```

### uv run でパッケージが入れ替わる

`uv run` は `uv.lock` に同期するため、手動インストールしたパッケージが上書きされることがあります。
`pyproject.toml` で XPU インデックスを設定済みであれば `uv sync` で正しくインストールされます。

## License

This project is licensed under the MIT License.

### Third-Party Licenses

| Component | License | Source |
|-----------|---------|--------|
| RVC WebUI | MIT | [RVC-Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) |
| ContentVec | MIT | [auspicious3000/contentvec](https://github.com/auspicious3000/contentvec) |
| fairseq / HuBERT | MIT | [facebookresearch/fairseq](https://github.com/facebookresearch/fairseq) |
| RMVPE | Apache 2.0 | [Dream-High/RMVPE](https://github.com/Dream-High/RMVPE) |
| Facebook Denoiser (optional) | CC BY-NC 4.0 | [facebookresearch/denoiser](https://github.com/facebookresearch/denoiser) |
| sounddevice | MIT | [spatialaudio/python-sounddevice](https://github.com/spatialaudio/python-sounddevice) |
| PyTorch | BSD | [pytorch/pytorch](https://github.com/pytorch/pytorch) |
| CustomTkinter | MIT | [TomSchimansky/CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) |
| transformers | Apache 2.0 | [huggingface/transformers](https://github.com/huggingface/transformers) |

### Note on Model Files

RVC model files (`.pth`) are subject to their own licensing terms. Ensure you have appropriate rights for any voice models you use, including consent from the original voice owner.
