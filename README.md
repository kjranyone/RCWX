# RCWX

RVC Real-time Voice Changer on Intel Arc (XPU)

## Features

- **Intel Arc GPU対応** - PyTorch XPU + torch.compile による高速推論
- **RVC v1/v2両対応** - 256次元・768次元特徴量モデルに対応
- **F0あり/なしモデル対応** - RMVPE F0抽出または低遅延モード
- **リアルタイム変換** - クロスフェード処理による低遅延・高品質変換
- **ノイズキャンセリング** - ML (Facebook Denoiser) / Spectral Gate 切替可能
- **ASIO対応** - プロオーディオインターフェース対応（WASAPI/DirectSound/MMEも利用可能）
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

# 2. (推奨) 低レイテンシF0抽出をインストール
uv sync --extra lowlatency

# 3. 必要モデル (HuBERT, RMVPE) のダウンロード
uv run rcwx download

# 4. GUI起動
uv run rcwx
```

**デフォルト設定** (最適化済み):
- F0方式: **RMVPE** (高品質)
- チャンクサイズ: **300ms**
- ノイズ除去: **ML** (Facebook Denoiser)
- FAISS インデックス: **有効** (ratio=0.15)
- 詳細は [Inference Settings](#inference-settings) 参照

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

```
┌─────────────────────────────────────────────────────────────────┐
│  RCWX - RVC Voice Changer                              [─][□][×]│
├────────────────────────────────────────────────────────────────┤
│  [メイン] [オーディオ] [詳細設定]                               │
├────────────────────────────────────────────────────────────────┤
│  ■ モデル選択                                                   │
│  [▼ ModelA                          ] [更新][開く...]           │
│  状態: F0あり (RVCv2) | インデックス: あり                     │
│                                                                 │
│  ■ ピッチシフト                                                 │
│  -24 ────────────●──────────────────────── +24                 │
│  現在値: +5 半音                                                │
│                                                                 │
│  ■ F0モード                                                     │
│  (●) RMVPE (高品質)    ( ) なし (低遅延)                       │
│                                                                 │
│  ■ ノイズキャンセリング                                         │
│  [✓] 有効   方式: [auto ▼]                                     │
│                                                                 │
│              ┌────────────────────────┐                        │
│              │       ▶ 開始          │                        │
│              └────────────────────────┘                        │
├────────────────────────────────────────────────────────────────┤
│ デバイス: Intel Arc A770 | レイテンシ: 145ms | 推論: 45ms      │
└─────────────────────────────────────────────────────────────────┘
```

## Noise Cancellation

騒音環境でのマイク入力を改善するノイズキャンセリング機能:

| 方式 | 説明 | 用途 |
|------|------|------|
| `auto` | ML利用可能ならML、なければSpectral | 推奨 |
| `ml` | Facebook Denoiser (PyTorch) | 高品質、人声保持 |
| `spectral` | Spectral Gate (DSP) | 軽量、低遅延 |

- **ML方式**: 機械学習ベースで人間の声を認識・保持しながらノイズを除去
- **Spectral方式**: 周波数スペクトルの閾値処理による従来型ノイズ除去

## Inference Settings

推論パイプラインの各パラメータの解説です。設定は `~/.config/rcwx/config.json` に永続化されます。

### 音声合成

| パラメータ | デフォルト | 範囲 | 説明 |
|-----------|-----------|------|------|
| `pitch_shift` | 0 | -24〜+24 | ピッチシフト（半音単位） |
| `noise_scale` | 0.45 | 0.0〜1.0 | VITS合成器のVAEノイズ量。0.0=完全決定的、0.66=RVC原典デフォルト。低いほどクリーン、高いほど自然なゆらぎが出る |
| `moe_boost` | 0.45 | 0.0〜1.0 | 萌え声スタイル強度。F0に声質特性を付与し、アニメ声的な抑揚を生成する |

### F0（基本周波数）制御

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `f0_method` | `rmvpe` | F0抽出方式。`rmvpe`=高品質（320ms最小チャンク）、`fcpe`=低遅延（100ms最小チャンク） |
| `f0_lowpass_cutoff_hz` | 16.0 | F0ローパスフィルタのカットオフ周波数（Hz）。100fps F0に対してButterworth 2次フィルタを適用。低い=滑らか、高い=ピッチの細かい変化を保持 |
| `enable_octave_flip_suppress` | true | F0抽出のオクターブ誤検出（±1オクターブのフレーム間ジャンプ）を自動補正。隣接フレーム比が2.0±0.16の場合にトリガー |
| `enable_f0_slew_limit` | true | フレーム間のF0変化量を制限し、ピッチの急激な飛びを抑制 |
| `f0_slew_max_step_st` | 3.6 | スルーリミッターの最大ステップ幅（半音/フレーム）。各F0値を前フレームの `2^(±step/12)` 倍以内にクランプ |
| `pre_hubert_pitch_ratio` | 0.08 | ピッチシフトの一部をHuBERT入力音声に事前適用する比率（0.0〜1.0）。HuBERTがシフト後の音色特徴を認識できるため、音色の精度が向上する。`scipy.signal.resample_poly` で実装（~1ms） |

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

## Project Structure

```
rcwx/
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
torchaudio = { index = "pytorch-xpu" }

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
uv run python tests/integration/test_pre_hubert_pitch.py
uv run python tests/crossfade/test_sola_compensation.py
uv run python tests/models/test_inference.py
uv run python tests/models/test_rmvpe.py
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
# → 2.10.0+xpu のように +xpu が付いていることを確認

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
| Facebook Denoiser | CC BY-NC 4.0 | [facebookresearch/denoiser](https://github.com/facebookresearch/denoiser) |
| sounddevice | MIT | [spatialaudio/python-sounddevice](https://github.com/spatialaudio/python-sounddevice) |
| PyTorch | BSD | [pytorch/pytorch](https://github.com/pytorch/pytorch) |
| CustomTkinter | MIT | [TomSchimansky/CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) |
| transformers | Apache 2.0 | [huggingface/transformers](https://github.com/huggingface/transformers) |

### Note on Model Files

RVC model files (`.pth`) are subject to their own licensing terms. Ensure you have appropriate rights for any voice models you use, including consent from the original voice owner.
