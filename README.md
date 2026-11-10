# RCWX — RVC Real-time Voice Changer on Intel Arc

RVC v2 モデルによるリアルタイムボイスチェンジャー。**製品は Rust 単一バイナリで、
実行時に Python / PyTorch への依存は一切ありません。** Python 実装
(`rcwx/`) は開発時のパリティ検証 (参照実装) としてのみ残っています。

対象環境: **Windows + Intel Arc (XPU)**。推論は独自の wgpu + WGSL
コンピュートエンジンで Intel Arc 上に直接実装しています (burn / ort /
ONNX Runtime 不使用、CUDA は未検証)。

## Quick Start

```powershell
.\rcwx.ps1                    # 環境チェック (Rust) → 対話メニュー
.\rcwx.ps1 gui                # GUI 直接起動
.\rcwx.ps1 rt --seconds 10    # リアルタイム変換を 10 秒実機確認
.\rcwx.ps1 run in.wav -o out.wav --pitch 5
```

初回のみ、モデルバンドル (変換済み重み) の生成が開発環境で必要です:

```powershell
.\dev.ps1 check               # 不足物の確認
.\dev.ps1 golden              # tools/convert_*.py で重み変換 + golden 生成
```

`rcwx.ps1` は cargo ビルド後にバイナリを起動します (cargo が無い環境では
`target/release/*.exe` を直接使用)。実行経路に Python は登場しません。

## 主な機能

- **リアルタイム変換** — 20ms hop (Aggressive) で全 hop 予算内の推論レイテンシ
  (p50 16.7ms / p99 18.2ms、hop バッチング方式、30 秒 1500 hop 実測)
- **GUI (egui)** — モデル / オーディオデバイス (入力ch選択) / レイテンシ
  (自動設定+推定表示) / Pitch・F0 (F0なし対応) / Index 検索 / ノイズ
  キャンセリング (spectral + 強度) / 入力ノイズゲート (感度) / ポスト処理
  (高域ブースト・リミッター・正規化) / 入力レベルメーター / 統計表示 /
  WAV 変換テスト
- **CLI** — `run` (オフライン変換) / `rt` (リアルタイム、`--virtual` は
  デバイスなし検証モード) / `info` (自前 .pth リーダ) / `devices` /
  `download` / `logs`
- **RVC v1/v2 両対応** — 256/768 次元特徴量、F0 あり/なしモデル
- **設定互換** — `~/.config/rcwx/config.json` を旧 Python 実装と共有

## なぜ pure Rust か (これまでの経緯)

このプロジェクトは元々 Python / PyTorch (Intel XPU) 実装として開発され、
Issue #6 で **完全 Rust シングルバイナリ化** をマイルストーン M0〜M7 として
計画・完遂しました。フレームワーク選定の経緯:

1. **burn (wgpu)** — 検証の結果、HuBERT 相当の transformer が WGSL 経由で
   約 4 倍遅く、native SPIR-V パスには数値バグがあったため不採用
2. **ort + DirectML** — 正しいが Python フリーの ONNX エクスポート
   ツールチェーンが必要で単一バイナリの目標に合わないため不採用
3. **独自 wgpu + WGSL エンジン** — 採用。matmul (tiled) / conv / softmax /
   graph 記録・plan 再生 (uniform slot システム) を自前実装し、HuBERT・
   Synthesizer (TextEncoder / Flow / GeneratorNSF / SineGen)・RMVPE・
   SwiftF0・FAISS IVF 相当の index 検索まで全て WGSL に移植

パリティは段階的に golden データで検証しています:
HuBERT cosine 1.0 / Synthesizer 0.9996 / RMVPE 0.9999 / SwiftF0 0.999 /
DSP 各モジュール (resample・SOLA・spectral gate・noise gate・postprocess)
は golden 完全一致 / **オフライン E2E corr 0.983** (Python `rcwx run` との
比較、gate 0.95)。カーネル最適化 (implicit-GEMM・活性化 fuse・FFT 相関) で
20ms hop を全 hop 予算内に収めました。fp16 (SHADER_F16) は DXC が packed
命令を生成しないため不採用 (検証記録あり)。

## GUI フレームワークの方針

現行の GUI は **egui (eframe)** 実装です (単一バイナリ・Windows 安定性を
優先)。**GPUI (Zed) への移行が方針として確定しています** — egui 実装は
旧 Python GUI に対する機能 Parity に到達しているため、GPUI 移行は描画層の
置き換えとして次期作業で実施します (エンジン層 `rcwx-nn` / `rcwx-rt` への
影響なし)。

## ディレクトリ構成

```text
crates/
├── rcwx-nn/     # 独自 wgpu+WGSL 推論エンジン (HuBERT/synth/RMVPE/SwiftF0/IVF)
├── rcwx-dsp/    # resample/SOLA/spectral gate/noise gate/postprocess/drift/feedback
├── rcwx-rt/     # リアルタイムエンジン (ストリーミング推論 + cpal + SPSC ring)
├── rcwx-cli/    # run / rt / info / devices / download / logs
└── rcwx-gui/    # egui GUI
tools/           # 重み変換 + golden 生成 (dev 専用・Python)
rcwx/            # 旧 Python 実装 (パリティ検証のリファレンス)
dev.ps1          # 開発ランチャー (check/build/test/golden/bench/e2e/fmt/clippy)
rcwx.ps1         # 製品ランチャー (純 Rust・Python 非依存)
```

## 開発

```powershell
.\dev.ps1 check               # 環境チェック
.\dev.ps1 test                # 全テスト (golden パリティ 17 件)
.\dev.ps1 bench               # 30 秒持続ベンチ (batch 1/4)
.\dev.ps1 e2e                 # Python 実装との E2E corr 比較
.\dev.ps1 golden              # 重み変換 + golden 再生成
```

詳細な開発ガイドは [AGENTS.md](AGENTS.md) を参照してください。

## モデルとライセンス

- 必須モデル: HuBERT (`hubert_base.pt`) / RMVPE (`rmvpe.pt`) — `rcwx.ps1 download` が取得
- RVC モデル (`.pth`) と FAISS index は各自の権利範囲でご利用ください
  (音声モデルの権利と本人同意を必ず確認してください)
- 本体 MIT。GTCRN (MIT) は Python 参照実装側、ML Denoiser (CC BY-NC 4.0) は
  Python 参照実装側の任意依存
