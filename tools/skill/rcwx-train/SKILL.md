---
name: rcwx-train
description: Train a pure-Rust RVC voice model on this machine with RCWX — preprocess a wav dataset, fine-tune from the base model, export, build the feature index, and smoke-test with rcwx-cli. Use when the user wants to create or train a voice model (RCWX 音声モデル学習), convert a speaker's audio into an RVC model, or run the RCWX training pipeline end-to-end.
---

# rcwx-train: pure-Rust RVC training pipeline

RCWX の学習はすべて純 Rust (`rcwx-cli`) で動く。Python は RVC 互換 .pth の
最終パッケージ化 (tools/export_pth.py) にしか使わない。

## 前提 (初回のみ確認)

1. `cargo build --release -p rcwx-cli` が通っていること (リポジトリルートで)。
2. 共通重み (HuBERT / RMVPE): なければ `rcwx-cli download` を実行。
3. base モデル (事前学習済み G/D) が
   `%USERPROFILE%\.cache\rcwx\train\pretrained\f0G48k_v2.pth` と `f0D48k_v2.pth`
   にあること。なければ HuggingFace から:
   `https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G48k.pth`
   (D も同様に f0D48k.pth)。
4. データセット: 1 話者・モノラル wav 10〜30 分程度。**Irodori TTS で作るなら
   sttts-say render (skill sttts-say) で 200 行以上生成し、必ず
   `sttts-say stop` してから学習する (GPU 排他)**。

## パイプライン (作業フォルダ = リポジトリルート)

```powershell
# 1) 前処理: 3s/0.3s チャンク + F0 + HuBERT 特徴 + spec キャッシュ
target\release\rcwx-cli.exe train-preprocess <wav_dir> --exp NAME

# 2) 学習 (base から fine-tune)
target\release\rcwx-cli.exe train --exp NAME --base-g $env:USERPROFILE\.cache\rcwx\train\pretrained\f0G48k_v2.pth --base-d $env:USERPROFILE\.cache\rcwx\train\pretrained\f0D48k_v2.pth --epochs 20

# 3) エクスポート + index 構築
target\release\rcwx-cli.exe train-export --exp NAME
target\release\rcwx-cli.exe train-index --exp NAME
```

成果物: `~\.cache\rcwx\train\NAME\model\` に `synth.safetensors` + `synth.json`
(rcwx 推論が直接読む形式) と IVF index。

## 学習済みモデルでの変換・リアルタイム

```powershell
# オフライン変換 (RCWX_MODELS_DIR でモデルバンドルを切り替え)
$env:RCWX_MODELS_DIR = "$env:USERPROFILE\.cache\rcwx\train\NAME\model"
target\release\rcwx-cli.exe run in.wav -o out.wav

# リアルタイム
target\release\rcwx-cli.exe rt
# または GUI: cargo run -p rcwx-gui --release (学習タブからも実行可能)
```

RVC WebUI 互換 .pth が必要なら:

```powershell
uv run python tools/export_pth.py --dir ~\.cache\rcwx\train\NAME\pth_out --out NAME.pth
```

## 判断のルール

- 学習時間は 1 epoch = データ長程度 (目安)。20 epoch で十分なことが多い。
- ログの `mel=` が単調低下していれば順調。`disc`/`gen` は GAN なので
  上下して正常。
- 前処理で 0 chunks なら wav が短すぎる (3 秒以上必要)。
- base .pth は WN (weight_v/g) 保持の学習 checkpoint 形式が必要
  (deployed .pth からの fine-tune も可能だが推奨は base から)。

## 注意

- 学習と推論 (rt / sttts-say) は GPU を排他する。同時に走らせない。
- v2 (768-dim HuBERT) のみ対応。v1 モデルは未対応。
- base モデルのライセンス (RVC 共有物) を確認すること。
