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
- Python 3.11 or 3.12（`uv` が自動で用意）
- Intel Arc GPU (A770, A750, B580, B570, etc.)
- Intel Arc GPU ドライバ（最新版推奨）
- [uv](https://github.com/astral-sh/uv) パッケージマネージャ

> 本プロジェクトは **Intel Arc (XPU)** 向けです。CUDA / NVIDIA 環境は未検証のためサポート対象外です。

## Installation / Quick Start

初回セットアップから日常起動まで、同梱の `rcwx.ps1` が自動で行います。

```powershell
git clone https://github.com/grand2-products/rcwx.git
cd rcwx

# 1. uv が未導入ならインストール（管理者権限不要）
irm https://astral.sh/uv/install.ps1 | iex
# または: winget install astral-sh.uv

# 2. 環境チェック → 対話メニュー
#    ・uv sync（PyTorch XPU 版を含む依存関係）
#    ・PyTorch / XPU の確認
#    ・必須モデル (HuBERT / RMVPE) の有無
#    ・ML デノイザ（任意）の有効化確認
.\rcwx.ps1
```

実行ポリシーで弾かれる場合:

```powershell
powershell -ExecutionPolicy Bypass -File .\rcwx.ps1
```

### 対話メニューでできること

| 番号 | 操作 |
|------|------|
| 1 | GUI を起動 |
| 2 | オーディオデバイス一覧 |
| 3 | 必須モデル (HuBERT / RMVPE) をダウンロード |
| 4 | フィードバック診断（XPU / Accelerator Graph 含む） |
| 5 | 最新ログを表示 (tail 50) |
| 6 | 環境チェックを再実行 |
| 7 | 依存関係を同期 (`uv sync`) |
| 8 | 任意の `rcwx` コマンドを入力 |
| 9 | ML デノイザを有効化 (`--extra ml-denoise`) |

### 直接起動（メニューを介さない）

```powershell
.\rcwx.ps1 gui                # GUI を直接起動
.\rcwx.ps1 -Denoise gui       # ML デノイザ有効化で GUI 起動
.\rcwx.ps1 download           # 必須モデルのみダウンロード
.\rcwx.ps1 diagnose           # 診断のみ
```

渡した引数はそのまま `uv run rcwx ...` へ転送されます。`-Denoise` 指定時（または既存の denoiser 検出時）は、全ての `uv run` に `--extra ml-denoise` が付きます。

> ML デノイザ（任意依存 `ml-denoise` extra、CC BY-NC 4.0）は既定で未インストールです。`uv run` / `uv sync` を extra なしで実行すると prune されるため、ランチャーは有効時に自動で `--extra ml-denoise` を付けます。

詳細なドライバ確認・トラブルシュートは [docs/SETUP.md](docs/SETUP.md) を参照してください。

**デフォルト設定**:
- F0方式: **RMVPE** (高品質)
- チャンクサイズ: **300ms**
- ノイズ除去: **ML** (Facebook Denoiser, 要 `--extra ml-denoise`) / Spectral (標準)
- FAISS インデックス: **有効** (ratio=0.15)
- 詳細は [Inference Settings](#inference-settings) 参照
### 手動セットアップ（開発者向け）

ランチャーを使わずに手順を踏む場合:

```powershell
uv sync
uv run rcwx download          # 必須モデル (HuBERT, RMVPE)
uv run rcwx diagnose          # XPU / Accelerator Graph 確認
uv run rcwx                   # GUI 起動

# (オプション) ML Denoiser
uv sync --extra ml-denoise
```

> `pyproject.toml` で PyTorch XPU インデックスが設定済みのため、`uv sync` だけで XPU 版が入ります。

## XPU Accelerator Graph

PyTorch `2.13.0+xpu` では、XPU利用時にAccelerator Graphが自動的に有効になります。HuBERT特徴抽出と、入力shapeが固定された定常状態のRVC Synthesizerが対象です。GUI側の設定操作は必要ありません。

- 変換開始前のウォームアップでHuBERT履歴を満たし、定常状態のSynthesizer Graphを1回capture
- 実ストリーム開始前に音声履歴とリサンプラー状態をリセットし、capture済みGraphだけを保持
- Graph API非対応またはcapture失敗時はeager推論へ自動フォールバック
- `use_compile=true` のSynthesizerはAccelerator Graphを併用せず、従来のcompile経路を使用

利用可否は診断コマンドで確認できます。

```powershell
.\rcwx.ps1 diagnose
# または: uv run rcwx diagnose
# [OK] XPU Accelerator Graph: True
```

ドライバーやモデル固有の問題を切り分ける場合は、環境変数で無効化できます。

```powershell
$env:RCWX_ACCELERATOR_GRAPH = "0"
.\rcwx.ps1 gui

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

推論デバイス（xpu/cpu/auto）、dtype、RVCモデルディレクトリ、モデル配置先などの設定。

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
- **除去強度**: `0.50x-2.00x`。`1.00x`はDNS64を1回処理し、MLは`1.00x`超で2回目のDNS64出力へ連続的に混合します。環境音が大きい場合は`1.25x`から上げ、声の欠けや金属音が出る手前で止めてください。Spectralでは閾値と減衰量を同時に深くします。

MLの2段処理は計算量も増えます。GUI の `Aggressive` では deadline を守るため Denoiser を強制 OFF にするため、強いノイズ除去を使う場合は `Normal` を選択します。

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
| `f0_method` | `rmvpe` | F0抽出方式。`rmvpe`=高品質（320ms最小チャンク）、`fcpe`=低遅延（100ms最小チャンク）、`swiftf0`=超低遅延 ONNX/CPU（Normal 40ms / Aggressive 20ms 最小） |
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

リアルタイム処理におけるチャンク境界の連続性を制御するパラメータ群です。GUI のレイテンシ設定では `chunk_sec` / `latency_mode` から自動導出され、保存済みの個別値より優先されます。

| パラメータ | 設定デフォルト | GUI自動導出 | 説明 |
|-----------|---------------|-------------|------|
| `overlap_sec` | 0.20 | chunkの100%（60–300ms、20ms刻み） | HuBERT入力に付加する音声オーバーラップ長。20ms境界（HuBERT 320サンプルホップ）に丸められる |
| `crossfade_sec` | 0.08 | chunkの10%（10–20ms、10ms刻み） | SOLAクロスフェード長。前チャンク末尾と次チャンク先頭をHann窓でブレンド |
| `use_sola` | true | true固定 | SOLA（Synchronized Overlap-Add）の有効化 |
| `sola_search_ms` | 15.0 | 15.0固定 | SOLA探索窓（ms）。最低出力F0 70Hzの1周期+マージン。レイテンシ表示には計上されるが探索幅自体は遅延を増やさない |
| `prebuffer_chunks` | 1 | Normal=1 / Aggressive=3 | 出力開始前（および Aggressive のアンダーラン再アーム）に確保する hop 数 |
| `buffer_margin` | 0.25 | Normal=0.25 / Aggressive=0.1 | Normal の持続リング floor 判定に使用（threshold ≈ 0.5+margin hop） |

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
├── accelerator_graph.py  # XPU固定shape Graphキャッシュ
├── config.py              # 設定管理 (JSON永続化)
├── device.py              # デバイス選択 (xpu/cpu)
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

レイテンシ設定は `Normal` と `Aggressive` の2モードです。モード変更時は I/O ストリームと固定 shape Graph を再ウォームアップします。

| モード | SOLAクロスフェード | 持続リングfloor | 非ASIOコールバック | 用途 |
|--------|-------------------|-----------------|----------------------|------|
| Normal | chunkの10%、最大20ms | 0.75 hopでtrim、0.25 hopへ復帰 | 最大10ms | 品質機能を維持した通常利用 |
| Aggressive | chunkの10%、最大20ms | 初期1 hop、定常0.5–0.875 hopを適応維持 | 最大2.5ms | 短い hop の deadline を狙う低遅延利用 |

#### Normal

- chunk 下限: SwiftF0 / None = 40ms、FCPE = 100ms、RMVPE = 320ms
- `prebuffer_chunks` = 1、`buffer_margin` = 0.25
- 開始時は最初の生成チャンクを待つ。持続的なバッファ滞留だけを 2 hop の観測窓で判定し、通常の chunk 内 burst/drain は trim しない
- 持続 floor: trim 閾値 ≈ `(0.5 + buffer_margin)` hop = 0.75 hop、復帰目標 = 0.25 hop
- Denoiser を含む品質機能は保存設定どおり使用
- decoder overlap = 既定 5 frames（1 frame = 10ms）

#### Aggressive

- chunk 範囲: SwiftF0 / None = 20–100ms（20ms 刻み）。FCPE = 100ms、RMVPE = 320ms が下限
- `prebuffer_chunks` = 3（開始時とアンダーラン再アームで同じ hop 数を再確保）
- 最初の 20 hop は 1 hop の guard（threshold は 1.25 hop）、その後は `max(0.5·hop, p99-p50+callback)` を `0.875·hop` で上限した jitter guard（20ms chunk では 10–17.5ms）
- HuBERT context 最大 560ms、SwiftF0 の F0 context 最大 100ms に一時短縮（保存設定は変えず runtime のみ）
- GUI 経路では Denoiser を強制 OFF（GUI のチェック状態は保持。`Normal` 復帰で再適用）。過負荷時（1 秒以内に Queue full が 3 回）も 2 秒間 denoise をバイパス
- 初回および catchup 直後は実音声 hop を左側へ reflect 展開し、`prime_hubert_history` で HuBERT 履歴を固定 shape まで即時充填 → 1 hop 目からウォームアップ済み Synthesizer Graph を使用
- decoder overlap = 0。SOLA 合成末尾は `crossfade + search` のみ保持
- モデル SR ≠ 出力 SR のとき、Synthesizer 出力の D2H 前に torchaudio sinc resample を XPU Graph で実行
- ASIO 実レートが設定と異なる場合は、音声開始前に実レート用 Graph を再ウォームアップ

#### Aggressive の XPU IVF（FAISS）

runtime warmup で FAISS `IndexIVFFlat`（L2、`nprobe=1`）の inverted list と再構築特徴を XPU へ常駐させ、coarse centroid 選択・候補検索・top-k 加重平均を Accelerator Graph で実行する。HuBERT 特徴を FAISS のためだけに CPU へ同期しない。

- 有効条件: L2 / `nprobe=1` / 最大 list 長 ≤ 256。未対応 index は CPU FAISS へ自動 fallback
- 特徴は FP16/BF16 で保持（158k×768 index で約 240MB 追加 VRAM）
- 候補距離は feature norm 事前計算 + 内積から算出
- `Normal` やファイル変換では構築しない

#### ホットパスの現行実装

- XPU stage timing: 10 hop ごと（`STAGE_PROFILE_INTERVAL=10`。イベント非対応時も同サンプリング）
- GUI / GPU メモリ telemetry: Aggressive は 5 hop ごと（20ms hop 時に約 10Hz）、Normal は毎 hop
- 推論 p50/p95/p99 の再計算: 10 hop ごと（直近 256 hop の窓）
- HuBERT は推論 thread から直接 dispatch、F0（SwiftF0 含む）のみ永続 `ThreadPoolExecutor(max_workers=1)` で並列
- streaming TextEncoder は `all_frames_valid` 時に padding mask / 全1 attention mask を省略（attention 範囲は不変）
- HuBERT / IVF の時間軸近似 cache は未採用。Aggressive でも HuBERT context 最大 560ms と TextEncoder global attention は維持

### レイテンシ表示

- ライブ表示: `1 hop + 推論 + 持続リングfloor + 出力キュー + SOLA`
- SOLA 分は合成余白（`crossfade + search`、Normal は + decoder overlap 5 frames）を計上。Aggressive 20ms では約 30ms
- 通常の出力チャンクがリングに残っているだけの状態は、floor として二重計上しない（post-read の持続 floor のみ）
- 推論表示は直近 256 hop の p95。ログは 100 hop ごとに `p50` / `p95` / `p99` / `deadline_miss`

### 性能目安（Intel Arc B570）

モデル・ドライバ・OS・同時 GPU 負荷で変動する。単発スパイクの非発生を保証するものではない。

| 条件 | p50 | p95 | p99 | deadline miss | 備考 |
|------|----:|----:|----:|--------------:|------|
| Aggressive 20ms hop（定常） | 8.3ms | 10.9ms | 11.8ms | 0/60 | 本番同等 Graph、SOLA 余白込みの推論系 |
| Aggressive 40ms hop | 14.5ms | 16.4ms | 18.8ms | 0/50 | SwiftF0、FAISS ratio 0.45 |
| XPU IVF 検索単体（~158k list） | 0.62ms | — | — | — | L2 / nprobe=1 |
| Synthesizer コア（Graph） | 4.5ms | — | — | — | eager 中央値は約 18.0ms |
| Graph 定常のストリーミング推論全体 | ~18.6ms | — | — | — | 固定 shape RVC v2、条件により変動 |

- Aggressive 20ms の E2E 目安: 推論 ~8–12ms + 定常 guard ~10ms + ASIO 入出力 ~12ms → **約 80–84ms**
- Aggressive 40ms の E2E 目安（20ms 相当 guard + ASIO ~12ms を含む）: **約 90–100ms**
- `noise_scale=0` の決定論的条件では、Synthesizer Graph と eager の RMSE / 最大絶対誤差は 0
- バッチ vs ストリーミングの品質目安: Correlation ≥ 0.94、MAE ≤ 0.02

## Supported Models

- RVC v1 (256-dim HuBERT features)
- RVC v2 (768-dim HuBERT features)
- F0モデル (NSF decoder)
- No-F0モデル (standard decoder)

モデルは [RVC-Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) で作成したものを使用できます。

## Configuration (pyproject.toml)

PyTorch XPU 版は `[tool.uv]` セクションで設定（現行 `pyproject.toml` と一致）:

```toml
[tool.uv]
# Resolve only for Windows
environments = ["sys_platform == 'win32'"]

# triton-xpu is Linux-only, but pytorch-triton-xpu works on Windows
override-dependencies = [
    "triton-xpu; sys_platform == 'linux'",
]

[tool.uv.sources]
torch = { index = "pytorch-xpu" }
pytorch-triton-xpu = { index = "pytorch-xpu" }

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
# ランチャーで環境チェックを再実行
.\rcwx.ps1
# メニュー 6) 環境チェック / 4) 診断

# または手動確認
uv run python -c "import torch; print(torch.__version__)"
# → 2.13.0+xpu のように +xpu が付いていることを確認

# +cpu の場合は uv.lock を再生成
del uv.lock
uv sync
```

### uv run でパッケージが入れ替わる

`uv run` は `uv.lock` に同期するため、手動インストールしたパッケージが上書きされることがあります。
`pyproject.toml` で XPU インデックスを設定済みであれば `uv sync`（または `.\rcwx.ps1`）で正しく入ります。
ML デノイザを使う場合は `.\rcwx.ps1 -Denoise ...` かメニュー 9) で有効化し、extra 付き同期を維持してください。

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
