# RCWX Development Guide

## Project Overview

**RCWX** = RVC Real-time Voice Changer on Intel Arc (XPU)

RVC v1/v2モデルを使ったリアルタイムボイスチェンジャー。**製品は Rust 単一バイナリ**
(`crates/` ワークスペース) で、Python 実装 (`rcwx/`) はパリティ検証の
リファレンスとして残している。

対象環境は **Windows + Intel Arc (XPU)**。CUDA / NVIDIA は未検証（コード上に分岐は残るがサポート対象外）。

## これまでの経緯 (pure Rust 化の記録)

Issue #6 として **完全 Rust シングルバイナリ化** を計画 (M0〜M7) し、完遂した。

| マイルストーン | 内容 | 検証 |
|---|---|---|
| M0 | burn/wgpu 検証 → **burn 不採用、独自 wgpu+WGSL エンジンを採用** | burn は transformer が約 4 倍遅い + SPIR-V に数値バグ。ort+DirectML も ONNX エクスポート要件で不採用 |
| M1 | HuBERT WGSL 移植 (fairseq 重みの weight_norm fold・QKV fusion) | golden と cosine 1.0 (v1/v2) |
| M2 | Synthesizer (TextEncoder/Flow/GeneratorNSF/SineGen) | cosine 0.9996 |
| M3 | F0: RMVPE (BN fold) + SwiftF0 (ONNX 図を自力解析し WGSL 移植) + f0 後処理鎖 | RMVPE 0.9999 / SwiftF0 0.999 |
| M4 | ホスト DSP 全般 (scipy 厳密一致の resample/SOLA/spectral/noise gate/postprocess/drift/feedback) + IVF + moe | DSP 全 golden 一致・IVF maxdiff 1e-3 |
| M5 | CLI (`run` E2E corr 0.983 > gate 0.95 / info・devices・download・logs) | info 出力が Python と同一 |
| M6 | リアルタイムエンジン (cpal・SPSC ring・drift 制御・hop バッチング) | 20ms hop p50 16.7/p99 18.2ms・全 hop 予算内・実機 MiniFuse で定常 underrun 0 |
| M7 | GUI (egui) + 純 Rust ランチャー化 | 旧 GUI の全セクション機能復元 |

主要な設計判断の記録:

- **ランチャー分離** — `rcwx.ps1` = 製品 (純 Rust・Python 非依存)、`dev.ps1` = 開発検証
  (golden 生成・E2E 比較のみ uv/Python を使用)。旧 Python ランチャーは git 履歴
  (`git show 84a75fe:rcwx.ps1`) に保存
- **重み形式** — 共通重みは `~/.cache/rcwx/models/rust/*.safetensors`。
  生成は `rcwx-cli download` が **Rust 内蔵の変換器**で実行 (tools/convert_*.py 不要)。
  Synth は RVC `.pth` を実行時に直接ロード
- **カーネル研究の知見** (30 秒 1500 hop 持続 A/B) は後述のセクションに集約
- **GUI フレームワーク**: 現行 **egui**。**GPUI (Zed) 移行が方針として確定**
  (ユーザー指定) — egui 実装は機能 Parity 済みのため、GPUI 化は描画層置換として
  次期作業。エンジン層 (rcwx-nn/rcwx-rt) への影響なし
- **GUI デザイン (gpui-kit 版)**: グラスモーフィズム「トワイライト・オーロラ」。
  `crates/rcwx-gui/src/theme.rs` に集約 — (1) `apply()` で `Theme::global` を桜/ラベンダー
  配色に塗り替え (gpui-kit コンポーネントは自動追従)、(2) 窓は
  `WindowBackgroundAppearance::Blurred` (DWM アクリル) + 半透明グラデーション + 
  呼吸するオーロラオーブ (`with_animation`)、(3) カードはグラデーション縁 + 
  半透明ガラス + ソフトシャドウ (`glass_panel`)、(4) ピル型タブ・VU メーター・
  チップ等の部品も theme.rs。view.rs はロジックのみ持ち、描画は theme.rs の部品を
  組むだけ。アイコンは `gpui_kit::assets::AllAssets` (Lucide 全カタログ)。
  GPUI は backdrop-blur 非対応のためガラスは半透明 + リムライトで表現。
  実機確認は `RCWX_START_TAB=1/2` で初期タブを変えられる

## Rust 単一バイナリ (Issue #6 — M0〜M7 実装済み)

Python/torch 依存ゼロの完全 Rust 実装。`crates/` ワークスペース:

```text
crates/
├── rcwx-nn/     # 独自 wgpu+WGSL 推論エンジン (burn/ort 不使用)
│   │   # pth.rs: torch zip+pickle 完全リーダ / convert.rs: .pt・ONNX→safetensors 変換器
│   ├── engine/  #   matmul(tiled)/conv1d/conv2d/softmax/graph/plan 再生
│   ├── hubert.rs / synth.rs / rmvpe.rs / swiftf0.rs / ivf.rs / f0.rs
├── rcwx-dsp/    # scipy 厳密一致ポリフェーズ resample / SOLA / spectral gate /
│   #            # noise gate / postprocess / drift / feedback / moe
├── rcwx-rt/     # リアルタイムエンジン: ストリーミング推論 (特徴/F0 sigmoid
│   #            # ブレンド・decode 受容野マージン・SOLA hold-back) + cpal + SPSC ring
├── rcwx-cli/    # run / info (自前 .pth pickle リーダ) / devices / download / logs / rt
└── rcwx-gui/    # GPUI GUI (モデル/デバイス/レイテンシ/pitch/denoise/統計)
                  # 見た目は theme.rs のグラスモーフィズム (下記参照)
```

- 開発ランチャー `dev.ps1` (rcwx.ps1 の Rust 版・同じ流儀):
  `check` 環境チェック / `build` / `test` パリティ一式 / `golden` 重み変換+golden 再生成 /
  `bench` 30 秒持続ベンチ (batch 1/4・bench30s.wav 自動生成) / `e2e` Python との corr 比較 /
  `fmt`・`clippy`・`clean` / `rt ...`・`run ...` 等は rcwx-cli へ転送 / 引数なしで対話メニュー。
  例: `.\dev.ps1 bench` / `.\dev.ps1 e2e` / `.\dev.ps1 rt --seconds 10` (UTF-8 BOM 必須・
  引数は `$args` で生受取し、`-o` 等の共通パラメータ衝突を回避)
- 重み変換は dev ツール (`tools/convert_*.py`) で `~/.cache/rcwx/models/rust/` へ
  (`RCWX_MODELS_DIR` で重みディレクトリを切替可能)
- パリティ: HuBERT cosine 1.0 / synth 0.9996 / **formant (return_length2) 0.9993** /
  RMVPE 0.9999 / swift-f0 0.999 /
  DSP 全モジュール golden 一致 / オフライン E2E corr 0.983 (>0.95)
- **本家 RVC 由来の品質機能 (2026-11 追加・いずれも既定 OFF で出力は従来同一)**:
  - **formant shift** (`--formant <st>` / GUI「フォルマント」): ピッチを保ったまま
    声質を移動。upstream `rtrvc.py`/`return_length2` と同一数学 — F0 を
    `pitch − formant` で逆補償し sinegen F0 を ×factor、z/har を
    `interp_time` カーネル (torch F.interpolate linear/align_corners=False 相当) で
    時間伸長して decode、`floor(factor·zc):zc` の polyphase でリサンプル戻し。
    decode が ×factor になるため大きい値は hop 予算に注意 (+5 で p99 約 +2ms)
  - **protect** (`--protect <0..0.5>` / GUI「子音保護」): 無声フレーム (f0≤0) で
    特徴を index 混入前へブレンドし子音・息の濁りを防止 (upstream pipeline.py 相当)
  - **rms_mix (音量エンベロープ追従)** (`--rms-mix <0..1>` / GUI スライダー%):
    出力ラウドネス包絡を入力へ追従 (upstream realtime_gui 相当・1=off)。
    `rcwx-dsp/src/envelope.rs`
  - **出力ノイズ除去** (`rt --output-denoise` / GUI チェック): 既存
    `StreamingSpectralGate` を出力側 (output_sr・リサンプル後) にも適用
  - `index_k` 既定を 4→8 (upstream と同一の 1/d² 重み付き和)
  - golden: `tools/dump_synth_formant.py` → `synth_formant_{pos,neg}.safetensors`
    (伸長状態で Python と照合・`synth_formant_parity` テスト)
- **モデルは RVC v1 (256-dim) / v2 (768-dim) 両対応**。synth.json の `version` から
  HuBERT 抽出 (v1: layer 9 + final_proj / v2: layer 12) と TextEncoder 形状
  (n_layers 3/6・kernel 5/3・emb_phone 入力次元) を切替。モデル SR (40k/48k 等) も
  synth.json から反映 (rt/cli とも従来 48k 決め打ちだった)。IVF index は次元が
  モデルと不一致なら blend を無効化して警告。実モデルが無い環境での検証は
  `uv run python tools/make_test_model.py` (ランダム重みの v1/v2 .pth 生成) →
  `tools/convert_synth.py <pth> --out ~/.cache/rcwx/models/rust-v1` →
  `cargo test -p rcwx-nn --test synth_parity` (synth_parity_v1 は rust-v1 ディレクトリ
  または `RCWX_MODELS_DIR_V1` があるときのみ実行)。ランダム重みモデルの parity は
  分布の関係で v1 0.9998 / v2 0.996 程度に落ちるため gate を 0.995 に緩和
  (synth.json の `random_test` フラグで判定・実モデルは従来どおり 0.999)
- 実行: `cargo run -p rcwx-cli --release -- run in.wav -o out.wav [--moe 0.45 --index-rate 0.15 --formant 5 --protect 0.3 --rms-mix 0.4]`、
  `... rt --virtual in.wav -o out.wav --chunk 0.06 [--formant 5 --protect 0.3 --rms-mix 0.4 --output-denoise]` (デバイスなし検証)、`... rt` (実デバイス)、
  `cargo run -p rcwx-gui --release`
- **リアルタイム性能 (20ms hop)**: implicit-GEMM conv カーネル (matmul_conv・
  im2col 廃止・lrelu fuse・z GPU 直結 slice_ch) で 125ms/hop → 45ms、さらに
  **hop バッチング** (入力は 20ms hop のまま推論を 4 hop 分統合・decoder マージン
  と固定費を償却) で **p50 16.8ms / p99 18.9ms・全 hop 予算内・リアルタイム追従OK**
  (rt --virtual --chunk 0.02 で検証)。E2E レイテンシ ≈ 80ms (入力蓄積) + 推論。
  chunk < 50ms で自動的に batch 4 (GUI/CLI とも)。バッチ内境界は連続合成で
  スプライスなし、バッチ境界のみ SOLA。
  fp16 (SHADER_F16) は DXC が packed 命令を出さず効果なし (検証済み・無効化)。
  単発 (バッチなし) では p50 ≈ 45ms/hop — さらなる単発高速化は今後の課題
- **オフライン変換の高速化 (fused attention + plan 再生)**: HuBERT の attention を
  **flash 形式の fused カーネル** (`flash_attn`・online softmax・T² スコア行列を
  材料化しない) に置換し、synth の decode chunk ループを**固定 staging での
  プラン 1 回構築 + replay** 化 (毎 chunk のグラフ再構築と 12MB z 再アップロードを
  廃止)。起動は学習カーネル除外 (`RCWX_SKIP_TRAIN_KERNELS=1`・cli/rt とも設定) +
  swiftf0 時の RMVPE 遅延ロード + kernel 並列コンパイルで削減。
  60s 変換が 125s → **~13.0s** (engine 3.3s / hubert 0.7 / f0 1.2 / decode 6.6) —
  **交互 5 ラウンドで warm Python (13.8-14.3s) を全ラウンド下回り PyTorch 超え達成**。
  decode chunk は 64 フレーム + margin 4 (rt と同じ・整相 corr 1.0 で品質等価)。
  **複 chunk の一括 submit (RCWX_DEC_BATCH) は逆効果**: このドライバは in-flight の
  コマンド量が増えると superlinear に遅くなる (batch2 で 1.5×・batch3 で 3.4×・実測) —
  既定 1 のまま。詳細は下記「シェーダコンパイラと attention 経路」
- **シェーダコンパイラと attention 経路 (Arc B570 実測)**:
  - wgpu 既定 (Auto) は dxcompiler.dll が見つからないと **FXC (SM5.1) に黙って
    フォールバック**する。FXC は matmul 系のコンパイルが ~3s/個・flash_attn は
    **30s** と遅いが、**生成コードの実行は DXC 比で ~2 倍高速** (rt p50 7.15 vs
    14.4ms・decode rb 8.5 vs 13.4s)。DXC は逆 (コンパイル高速・実行遅い)。
  - 既定: **rt/GUI = FXC + 古典 attention** (T≤560 では T² も微小)。
    **cli `run` は入力長で切り替え** — 30s 未満は DXC (起動 0.8s)、以上は FXC
    (コンパイル 3.3s だが実行 ~1.9× 高速)。env: `RCWX_DXC=0|1` (強制) /
    `RCWX_DXC_PATH=fxc|<path>` / `RCWX_ATTN=flash|classic` /
    `RCWX_DX12_SM=6.x` (診断) / `RCWX_COMPILE_THREADS=N`。
  - カーネルコンパイルは 8 スレッド round-robin 並列 (FXC 44s → 6.3s・起動も
    44s → 6.5s)。wgpu の pipeline cache は DX12 が stub で効かないため、
    将来は dxcompiler.dll を実行ファイル横に同梱して DXC 併用が本命。
  - **古典 attention (T² scores 材料化) は persistent scratch 必須**: 従来
    build_plan が層ごとに作業バッファを確保し plan が握り続けるため t≈1700 で
    scores 12 層分 ~1.7GB が resident 化し、WDDM residency 圧迫から**以降の全
    ステージが 5〜20 倍減速** (decode 8.5→90s) したり 2 窓で device lost した。
    全層で 9 本の scratch (qkv/scores/ctx/o/x3/ff/ff2/x ping-pong) を再利用する
    ことで resident は ~190MB/窓に収まり両方解消 (数値は flash と corr 1.0)。
    SwiftF0 は長尺でも内部 ~1.4GB/27s 級の中間バッファを持つため 25s+1s 窓分割が
    必要 (単発は 2GB 上限で落ちる)。`RCWX_HUBERT_WIN` で窓長上書き可
  - flash_attn の WGSL は行 max を SLM に出版するフェーズを入れると FXC の
    コンパイルが爆発するため、各スレッドが私的に max を取る 2-barrier 構造
    (p 再計算の exp 増と引き換え)。vec4→スカラー化・手動展開・部分和分割では
    直らない (FXC のループネスト最適化の superlinear なコスト)
- **重みの実行時調達 (頒布対応)**: 共通重み (HuBERT/RMVPE/SwiftF0) は
  `rcwx-cli download` が公式配布元から .pt/wheel を落とし **バイナリ内蔵の変換器**
  (`rcwx-nn/src/convert.rs` + `pth.rs` の torch zip+pickle 完全リーダ) で
  safetensors 化する — Python 不要。**Synth は RVC .pth を実行時に直接ロード**
  (`Synth::load_from_pth`・weight_norm fold・enc_q 除外・v1/v2/Nono 対応)。
  GUI のモデル選択は `RtConfig.model_path` 経由で実際の声に反映される
  (cli は `run`/`rt` の `--model`)。`dev.ps1 golden` はパリティ検証用の
  参照生成が主目的に。
- SwiftF0 のオフライン `run` は長尺入力で GPU のバッファバインド
  上限 (アダプタにより 128MB) を超える場合あり (rt の hop 窓経路では問題なし)。


## Quick Start

製品は **Rust 単一バイナリ**。`rcwx.ps1` は cargo ビルド→バイナリ起動のみを行い、
実行経路に Python 依存は一切ない。

```powershell
.\rcwx.ps1                    # 環境チェック (Rust) → 対話メニュー
.\rcwx.ps1 gui                # GUI 直接起動
.\rcwx.ps1 rt --seconds 10    # リアルタイム 10 秒実機確認
.\rcwx.ps1 run in.wav --model rvc_models/voice.pth -o out.wav --pitch 5
.\rcwx.ps1 devices / info rvc_models/voice.pth / logs --tail 50
```

初回のみ、共通重み (HuBERT/RMVPE/SwiftF0) のダウンロードと変換
(バイナリ内蔵・Python 不要):

```powershell
.\rcwx.ps1 download           # 公式配布元から取得 → ~/.cache/rcwx/models/rust/ へ変換
```

RVC モデル (.pth) は配置して GUI/CLI で選ぶだけで動く (変換不要)。GUI は
モデル設定タブの「初回セットアップ」ボタンでも同じ処理を行う。

### リリース (タグ → 配布アーカイブ自動生成)

`v*` タグを push すると `.github/workflows/release.yml` が Windows x64 向けに:

1. `rcwx-cli` / `rcwx-gui` を release ビルド (CI のみ env で `debug=0` + `strip=symbols`。
   ローカル開発用の `debug = true` は Cargo.toml から変更しない)
2. `cargo about` (`about.toml` + `templates/about.hbs`) で `THIRD_PARTY_NOTICES.md` を生成
   (リポジトリにはコミットしない・`.gitignore` 済み)
3. zip を組み立てて GitHub Release に添付: `rcwx-<tag>-windows-x64.zip`
   (中身: rcwx-cli.exe / rcwx-gui.exe / rcwx.ps1 / README.md / SETUP.md / LICENSE /
   THIRD_PARTY_NOTICES.md)

`rcwx.ps1` はスクリプト横に `rcwx-cli.exe` があると配布レイアウトとして扱い
(cargo ビルドをスキップ)、`target\release\` があれば開発レイアウトで従来どおり。

## 開発・検証 (Python 参照実装)

Python (`.venv` / `rcwx/`) は **パリティ検証のリファレンス専用**。製品経路からは
完全に分離されており、`dev.ps1` 経由でのみ使う:

```powershell
.\dev.ps1 test                # Rust テスト一式 (golden パリティ)
.\dev.ps1 bench               # 30 秒持続ベンチ (batch 1/4)
.\dev.ps1 e2e                 # Python rcwx run との corr 比較 (gate 0.95)
.\dev.ps1 fmt / clippy / clean
```

旧 Python ランチャー (`uv sync` / ML デノイザ等) の機能は git 履歴
(`git show 84a75fe:rcwx.ps1`) に保存。Python 実装の直接実行:

```powershell
uv sync
uv run rcwx diagnose          # フィードバック診断 (Rust は rt ログ/GUI の corr 表示で代替)
```

補足:

- `torchfcpe` / `swift-f0` は本体依存（`--extra lowlatency` は存在しない）
- ML デノイザだけ optional extra `ml-denoise`。`uv run` を extra なしで実行すると prune される
- `pyproject.toml` で PyTorch XPU インデックス設定済み。`torch.__version__` に `+xpu` が付くこと

## Current Architecture

```text
AudioInput (mic rate)
  -> Input accumulator (hop単位)
  -> [Input Queue]
  -> Inference Thread
       1) input gain
       2) StatefulResampler (mic -> 16k)
       3) optional denoise (auto/ml/gtcrn/spectral)
       3.5) optional input noise gate（dBFS閾値のダウンワードエクスパンダー）
       4) [overlap | new_hop] を組み立て
       5) RVCPipeline.infer_streaming()
       6) StatefulResampler (model_sr -> output_sr)
          ※ Aggressive + XPU かつ SR 変換時は D2H 前にデバイス sinc resample
       7) soft clip + SOLA crossfade
       8) feedback detection
       9) [Output Queue]
  -> RingOutputBuffer
  -> AudioOutput (48kHz)
```

実装上の要点:

- リアルタイム経路は `pipeline/realtime_unified.py` のみ
- チャンク境界連続性は **audio-level overlap** + `infer_streaming()` + SOLA
- PyTorch 2.13 Accelerator Graph を XPU で自動適用（HuBERT + 定常 Synthesizer + Aggressive 時 IVF）
- Synthesizer は HuBERT 履歴が満杯になるまで eager。開始前 warmup で定常 Graph を capture
- warmup 後に音声履歴 / リサンプラ / SOLA 状態はリセットし、capture 済み Graph は保持
- `RCWX_ACCELERATOR_GRAPH=0` で Graph 無効化
- 過負荷時（直近1秒で Queue full が3回以上）は一時的に **denoise のみバイパス**（`f0_method` / `index_rate` は変更しない）。2秒後に自動復帰
- denoise は全モードでユーザートグルに従う（Aggressive でも強制 OFF しない）。**方式は Aggressive で `gtcrn` に固定**（品質ポリシー: 学習型ストリーミング・CPU ~2ms/hop。保存設定・GUI 表示は変更しない）。負荷が hop を超える場合は上記の過負荷バイパスが受け止める
- リアルタイムの spectral は `StreamingSpectralGate`（永続 STFT、n_fft=512/hop=256、最小統計ノイズ追跡、追加遅延ゼロ: 最新 ≤16ms は dry クロスフェード近似）。ステートレス呼び出し（オフライン・直呼び）は hop が n_fft 2048 未満だとパススルー
- 入力ノイズゲート（`noise_gate_enabled`）は denoise 後・推論前に residual noise を減衰し、ノイズの誤変換を防ぐ（guitar NS 風: 3dB ヒステリシス + 60ms hold、減衰下限 -40dB 固定）。しきい値は自動: RMS エンベロープ（τ=20ms）の 1s 窓最小フロア + 感度マージン、-30dBFS 超では無条件で開放（loud 持続信号保護）。GUI はチェックボックス + 感度（低/中/高）のみで `ノイズキャンセリング` セクション内。手動 dBFS は `noise_gate_auto=false`（config 経由のみ）

## Directory Structure

```text
rcwx/
├── accelerator_graph.py   # XPU 固定shape Graph キャッシュ
├── cli.py
├── config.py
├── device.py
├── diagnose.py
├── downloader.py
├── audio/
│   ├── input.py           # AudioInput
│   ├── output.py          # AudioOutput
│   ├── duplex.py          # ASIO 全二重
│   ├── buffer.py          # RingOutputBuffer
│   ├── resample.py        # StatefulResampler
│   ├── sola.py
│   ├── denoise.py         # auto/ml/gtcrn/spectral
│   ├── gtcrn.py           # GTCRN streaming denoiser (MIT, CPU ONNX)
│   ├── noise_gate.py      # 入力側ノイズゲート（guitar NS 風エクスパンダー）
│   ├── postprocess.py     # treble + normalizer + limiter
│   ├── wav_input.py
│   └── stream_base.py
├── models/
│   ├── hubert_loader.py
│   ├── accelerator_index.py  # デバイス常駐 IVF
│   ├── rmvpe.py
│   ├── fcpe.py
│   ├── swiftf0.py         # ONNX/CPU
│   ├── synthesizer.py
│   └── infer_pack/
├── pipeline/
│   ├── inference.py       # infer / infer_streaming
│   ├── realtime_config.py # RealtimeConfig / RealtimeStats / SOLA 余白計算
│   ├── drift_control.py   # shed policy + FloorTracker
│   └── realtime_unified.py
└── gui/
    ├── app.py
    ├── realtime_controller.py
    ├── model_loader.py
    ├── audio_test.py
    ├── file_converter.py
    └── widgets/
        ├── audio_settings.py
        ├── latency_settings.py
        ├── latency_monitor.py
        ├── model_selector.py
        ├── pitch_control.py
        └── postprocess_settings.py
```

## Configuration (Current)

永続化: `~/.config/rcwx/config.json`（`RCWXConfig`）。
GUI レイテンシ枠は `chunk_sec` / `latency_mode` から overlap 等を **自動導出して実行時に上書き**する。

### `AudioConfig` (`rcwx/config.py`)

| Key                       |  Default | Notes |
| ------------------------- | -------: | ----- |
| `sample_rate`             |  `16000` | 内部処理入力レート |
| `output_sample_rate`      |  `48000` | 出力レート |
| `chunk_sec`               |    `0.3` | 保存設定上のチャンク長（20ms 境界） |
| `latency_mode`            | `normal` | `normal` / `aggressive` |
| `prebuffer_chunks`        |      `1` | GUI では mode で 1 or 2 に再設定 |
| `buffer_margin`           |   `0.25` | GUI: Normal=0.25 / Aggressive=0.1 |
| `input_gain_db`           |    `0.0` | 入力ゲイン（GUI スライダー -30〜+24 dB） |
| `output_gain_db`          |    `0.0` | 出力レベル調整 |
| `input_channel_selection` |   `auto` | left/right/average/auto |
| `output_channel_selection`|   `auto` | auto / `"0,1"` 等 |
| `input_hostapi_filter`    | `WASAPI` | Windows 向け |
| `output_hostapi_filter`   | `WASAPI` | Windows 向け |
| `asio_buffer_size`        |      `0` | frames。0=ドライバパネル準拠 |

### `InferenceConfig` (`rcwx/config.py`)

| Key                           |  Default | Notes |
| ----------------------------- | -------: | ----- |
| `f0_method`                   | `swiftf0` | `swiftf0` / `rmvpe` / `fcpe` |
| `use_f0`                      |   `true` | |
| `use_index`                   |   `true` | |
| `index_ratio`                 |   `0.15` | |
| `index_k`                     |      `4` | |
| `use_compile`                 |  `false` | 既定 OFF。compile 時は Synthesizer Graph 非併用 |
| `resample_method`             | `linear` | `linear` / `poly` |
| `use_parallel_extraction`     |   `true` | HuBERT は推論 thread、F0 は永続 worker |
| `voice_gate_mode`             |    `off` | off/strict/expand/energy |
| `energy_threshold`            |    `0.2` | energy モード |
| `overlap_sec`                 |   `0.20` | 設定デフォルト。GUI は chunk 100%（60–300ms）で上書き |
| `crossfade_sec`               |   `0.08` | 設定デフォルト。GUI は chunk 10%（10–20ms）で上書き |
| `use_sola`                    |   `true` | `false`（または `crossfade_sec=0`）にすると SOLA 余白の生成自体を止める（余白を消費する splice が走らないため）。変更は再起動で反映 |
| `sola_search_ms`              |   `10.0` | 100Hz 1周期。GUI 固定 10。`crossfade + search ≤ 20ms` で SOLA 余白が 10ms モデルフレーム1枚分縮む |
| `hubert_context_sec`          |    `1.0` | Aggressive runtime は最大 0.56 に cap |
| `f0_context_sec`              |   `0.32` | Aggressive + SwiftF0 は最大 0.10 に cap。`<=0` で全コンテキスト抽出 |
| `moe_boost`                   |   `0.45` | |
| `noise_scale`                 |   `0.45` | |
| `fixed_harmonics`             |   `true` | |
| `f0_lowpass_cutoff_hz`        |   `16.0` | |
| `f0_hole_fill_ms`             |   `30.0` | 有声内の短い無声穴補間上限。`<=0` で無効 |
| `uv_ramp_ms`                  |    `5.0` | NSF 有声/無声励振クロスフェード |
| `enable_octave_flip_suppress` |   `true` | |
| `enable_f0_slew_limit`        |   `true` | |
| `f0_slew_max_step_st`         |    `3.6` | |
| `denoise.enabled`             |   `true` | |
| `denoise.method`              |     `ml` | `auto` / `ml` / `gtcrn` / `spectral`。**Aggressive では `gtcrn` に固定** |
| `denoise.strength`            |    `1.0` | `0.5–2.0`。ML は 1.0 超で2段 |
| `noise_gate_enabled`          |  `false` | 入力側ノイズゲート（denoise後・推論前）。ノイズ誤変換防止 |
| `noise_gate_auto`             |   `true` | しきい値自動（1s 窓最小フロア + 感度マージン）。GUI は自動のみ |
| `noise_gate_sensitivity`      |    `mid` | `low`(+10dB) / `mid`(+6dB) / `high`(+3dB) マージン |
| `noise_gate_threshold_db`     |   `-40.0` | 手動閾値 dBFS（`noise_gate_auto=false` 時のみ・config 経由） |

### `RCWXConfig` トップレベル

| Key               |                Default | Notes |
| ----------------- | ---------------------: | ----- |
| `models_dir`      | `~/.cache/rcwx/models` | HuBERT / RMVPE |
| `rvc_models_dir`  |                 `None` | ドロップダウンスキャン |
| `last_model_path` |                 `None` | |
| `device`          |                 `auto` | 実質 `auto` / `xpu` / `cpu`（`cuda` は未検証） |
| `dtype`           |              `float16` | float16/float32/bfloat16 |

### `RealtimeConfig`（GUI から生成される実行時）

`RealtimeConfig` dataclass 自体のフィールド既定は GUI 経路と一致しない場合がある。
製品パスは `realtime_controller` が GUI / `RCWXConfig` から埋める。

GUI 既定起動時の代表値:

- `chunk_sec` ≈ 0.30（20ms 境界）、`latency_mode` = `normal`
- `overlap_sec` / `crossfade_sec` / `prebuffer_chunks` / `buffer_margin` = 下記 GUI 自動導出
- `sola_search_ms` = 10.0、`use_sola` = true
- `hubert_context_sec` = 1.0、`f0_context_sec` = 0.32
- `f0_method` / `noise_scale` / denoise / postprocess は GUI・config 準拠
- `decoder_overlap_frames` = 5（Aggressive では実行時に 2 扱い）
- `max_queue_size` = 8

## GUI Latency Model (Current)

`LatencySettingsFrame` でユーザーが変えるのは **`chunk_sec` と `latency_mode`**。
他は `_auto_params()` と runtime で決定。

### 自動導出 (`latency_settings.py`)

- `overlap_sec` = chunk の 100%（60–300ms、20ms 刻み）
- `crossfade_sec` = chunk の 10%（10–20ms、10ms 刻み）両モード共通
- `sola_search_ms` = 10.0 固定
- `use_sola` = true
- Normal: `prebuffer_chunks` = 1、`buffer_margin` = 0.25
- Aggressive: `prebuffer_chunks` = 2、`buffer_margin` = 0.1

### chunk 下限 / 範囲

| mode | SwiftF0 / none | FCPE | RMVPE |
|------|----------------|------|-------|
| Normal | ≥ 40ms（〜600ms、20ms 刻み） | ≥ 100ms | ≥ 320ms |
| Aggressive | 20–100ms（20ms 刻み） | ≥ 100ms | ≥ 320ms |

### runtime 挙動 (`realtime_unified.py`)

**Normal**

- 非 ASIO コールバック最大 10ms
- 持続リング floor: trim 閾値 ≈ `(0.5 + buffer_margin)` hop = 0.75 hop、復帰 0.25 hop
- 2 hop 観測窓。chunk 内 burst/drain は trim しない
- Denoiser は設定どおり
- decoder overlap = 設定値（既定 5 frames = 50ms）

**Aggressive**

- 非 ASIO コールバック最大 2.5ms
- 開始時・アンダーラン再アームで `prebuffer_chunks`（2）hop 確保
- 最初の 20 hop は 1 hop guard（threshold 1.25 hop）。以降 `max(0.5·hop, p99-p50+callback)` を `0.875·hop` で上限。callback は実測コールバック長（ASIO 実ブロック）を反映、ヒステリシス帯は `max(hop/4, callback)`
- 推論が budget の 90% 超の hop は `[SPIKE]` ログ（pre/infer/post 内訳 + idle_gap + queue 深度 + GC 停止時間、0.5s スロットル）
- start 時に WDDM GPU スケジューリング優先度を HIGH（不可なら ABOVE_NORMAL）へ引き上げ、DWM プリエンプション対策（`rcwx/win_gpu_priority.py`、`RCWX_GPU_PRIORITY=0` で無効）
- HuBERT context 最大 0.56s、SwiftF0 の f0 context 最大 0.10s（保存設定は変更しない）
- denoise はユーザートグル準拠（強制 OFF しない）。ML denoiser 等で hop を超えると過負荷バイパスが一時的に外す
- FAISS IVF を warmup 時に XPU 配置（L2 / nprobe=1 / max list ≤256）。未対応は CPU FAISS
- 初回・catchup は実 hop を reflect + `prime_hubert_history` で固定 shape 充填 → 1 hop 目から定常 Synthesizer Graph
- decoder overlap 実行時 2 frames (20ms)。SOLA 余白は `crossfade + search` + decoder overlap
- モデル SR ≠ 出力 SR 時は D2H 前に XPU Graph sinc resample
- ASIO 実レート不一致時は開始前に実レート用 Graph を再 warmup

### ホットパス（AS-IS）

- stage timing プロファイル: 10 hop ごと（`STAGE_PROFILE_INTERVAL=10`）
- GUI / GPU メモリ telemetry: Aggressive は 5 hop ごと（20ms hop で約 10Hz）、Normal は毎 hop
- 推論 p50/p95/p99: 直近 256 hop、再計算は 10 hop ごと
- HuBERT は推論 thread 直 dispatch、F0 は永続 `ThreadPoolExecutor(max_workers=1)`
- streaming TextEncoder は `all_frames_valid` で mask 処理省略
- HuBERT/IVF 時間軸 cache は未採用
- IVF 候補距離は feature norm 事前計算 + 内積
- PERF ログ: 100 hop ごと。slow 警告は 50 hop ごと（推論 > 0.8·chunk）

### レイテンシ表示

`1 hop + 推論 + 持続リング floor + 出力キュー + SOLA余白`

SOLA 余白は `crossfade + search` + decoder overlap（両モード共通）。リング内の通常 hop 滞留は二重計上しない。

## CLI Commands

```powershell
.\rcwx.ps1                   # 推奨ランチャー
uv run rcwx                  # GUI（command 省略時デフォルト）
uv run rcwx gui
uv run rcwx devices
uv run rcwx download
uv run rcwx run in.wav model.pth -o out.wav --pitch 5 --f0-method rmvpe
uv run rcwx info model.pth
uv run rcwx diagnose
uv run rcwx logs
uv run rcwx logs --tail 50
uv run rcwx logs --open
```

## Logs & Diagnostics

ログ: `~/.config/rcwx/logs/rcwx_YYYYMMDD_HHMMSS.log`

よく出るログ:

- `[INPUT] Queue full, dropping chunk`
- `[INFER] Chunk #...`
- `[PERF] Inference slow ...`（pre/hubert/f0/faiss/synth 内訳）
- `[PERF] Stage breakdown: ...`（100 hop ごと。GPU はデバイスイベント計測、ホスト同期なし。イベント非対応時は 10 hop サンプリング + sync フォールバック。並列抽出時 hubert/f0 は時間重複 → 正確な帰属には `use_parallel_extraction=false`）
- `[FEEDBACK] Detected feedback (corr=...)`
- `[WARMUP] ...`
- `[UNDERRUN] ...`（Aggressive は prebuffer 再アーム）

トラブルシュート:

- Queue full 頻発: `chunk_sec` 増、denoise OFF、F0 見直し、Aggressive 負荷確認
- 遅延増加: mode / chunk / buffer floor / 出力デバイス確認
- フィードバック: `.\rcwx.ps1 diagnose` または `uv run rcwx diagnose`
- XPU 未認識: `torch.__version__` に `+xpu`。だめなら `del uv.lock` → `uv sync` または `.\rcwx.ps1`

## Models

- `~/.cache/rcwx/models/hubert/hubert_base.pt`
- `~/.cache/rcwx/models/rmvpe/rmvpe.pt`
- RVC `*.pth`（任意）
- FAISS `*.index`（モデル隣接・自動検出）

## Tests

```powershell
uv run python tests/integration/test_diagnostic.py
uv run python tests/integration/test_infer_streaming.py
uv run python tests/integration/test_moe_f0_processing.py
uv run python tests/integration/test_moe_clarity_scoring.py
uv run python tests/crossfade/test_sola_compensation.py
uv run python tests/integration/test_decoder_overlap.py
uv run python tests/models/test_inference.py
uv run python tests/models/test_rmvpe.py
uv run python tests/models/test_cumulative_context.py
uv run python tests/models/test_pitch_clarity.py
uv run python tests/models/test_config_wiring.py
uv run python tests/models/test_hubert_weight_audit.py
uv run python tests/models/test_accelerator_graph.py
uv run python tests/models/test_synthesizer_graph.py
uv run python tests/models/test_runtime_graph_warmup.py
uv run python tests/models/test_accelerator_index.py
uv run python tests/models/test_text_encoder_fastpath.py
uv run python tests/models/test_preproc_pipeline.py
uv run python tests/audio/test_gtcrn_denoise.py
uv run python tests/audio/test_spectral_stream.py
uv run python tests/audio/test_noise_gate.py
uv run python tests/audio/test_denoise_aggressive_policy.py
```

## References

- RVC WebUI
- PyTorch XPU
- Facebook Denoiser (`denoiser`)
- torchfcpe / swift-f0
- ユーザー向けセットアップ: [docs/SETUP.md](docs/SETUP.md) / [README.md](README.md)

## 学習 (Issue #8 — 純 Rust パイプライン)

推論と同じ wgpu エンジン上に autograd tape (`rcwx-nn/src/engine/tape.rs`) を実装し、
学習も完全純 Rust 化 (`rcwx-train` crate + `rcwx-cli train-*`)。

- **tape autograd**: 1 学習ステップ = 1 Graph = 1 submit (forward + backward +
  AdamW 更新を単一 compute pass 内の dispatch 順序で実現)。WN は `_v`/`_g` を
  学習パラメータとし行単位 fold (PyTorch dim=0 と同一)。uniform slot は使い捨て
  Graph で回収 (rt 推論経路は不変)。
- **検証**: PyTorch golden との fwd+bwd パリティ — enc_p (attention 6 層) と
  enc_q/flow/decoder すべて cos=1.0 (`tools/train_ref.py` / `train_ref2.py` →
  `crates/rcwx-train/tests/module_parity*.rs`)。op 単位の FD gradcheck も全通過。
- **パイプライン**: `train-preprocess` (3s/0.3s チャンク + RMVPE F0 + HuBERT 特徴 +
  spec キャッシュ) → `train --base-g/--base-d` (LSGAN + FM + 45·mel-L1 + KL、
  AdamW 1e-4/(0.8,0.99)、fp32) → `train-export` (WN fold → synth.safetensors/
  synth.json = 推論ローダ無変更 + RVC 互換 .pth 中間生成) → `train-index`
  (IVF nlist=min(16√N, N/39) → 既存 ivf.rs 形式)。base は本家
  `f0G48k_v2.pth`/`f0D48k_v2.pth` (WN 保持 checkpoint 形式)。
- **Applio 差分取り込み**: 前処理は Applio Automatic/Simple 相当。MPD は
  periods [2,3,5,7,11,17,23,37] + DiscriminatorS。`--mel-loss multi`
  (MultiScaleMel) 実装済み。RefineGAN は後続課題。
- **GUI**: 学習タブ (`rcwx-gui/src/train_tab.rs`) から前処理・学習・エクスポート。
- **Skill**: `tools/skill/rcwx-train/SKILL.md` がパイプラインを自動化。
- **実スケール学習 (検証済み)**: Tape の分割 submit (`maybe_flush`, 256 dispatch
  ごとに途中 submit + uniform slot 回収) で device loss を解消。本家
  f0G48k_v2/f0D48k_v2 base からの fine-tuning を実走行。分割累積 matmul 導入後は
  ~90s/epoch (DFT chunk 16 の追加分含む)。device loss 時は `--resume` で
  checkpoint から再開 — エポック単位 checkpoint を実装。
- **tiny データ (8 発話) fine-tune の知見**: engine 修正後 (正確な D 勾配・
  cond bias 勾配) の学習は GAN 圧が本物になり、少数データでは
  cond 系バイアス (512 次元) が大きくドリフトして過学習する
  (推論出力が rms 0.003 に縮退・KL -1.6)。旧エンジン (D 勾配欠損) は
  結果的に正則化として働き軟らかいモデルを出していた。E2E corr は
  モデルの鋭さ/縮退度で 0.79〜0.97 に変動 — **同一重み比較の E2E ゲート
  (corr ≥ 0.98) は base モデルで 0.9967 (best-lag 0.9986) を維持**。
  実データでの本格 fine-tune には十分なデータ量とレシピ調整
  (lr/epoch/cond 系の制約) が別途必要。
  エクスポート: native (推論即使用) + .pth (WN v/g 保持 + weight_g 形状復元 —
  `load_state_dict` missing 0 keys)。ただし RVC 48k v2 checkpoint は判別器キーが
  `discriminators.{i}` 形式のためローダでリマップ。
- **E2E corr ゲート (≥0.98) 達成** — 学習済みモデル (e2e_test) の
  full-pipeline 比較で **py(fp32) vs rust = 0.9956〜0.9968**
  (`tools/e2e_parity.py` — 製品経路・fp32・monkeypatch なし)。原因調査の結論:
  1. **py CLI は既定 `--dtype float16`**。fp16 のまま比較すると py(fp16) vs
     py(fp32) 自体が corr 0.988 しか出ない (fp16 ノイズフロア)。ゲートは
     **fp32 同士**で測ること。
  2. 学習前の旧調査で見えた波形デコリレーション (−0.36 等) は **末尾 f0 の
     `stabilize_f0_boundaries` クリフ**: 最終有声フレームが `n-1-edge_frames`
     境界を跨ぐと片側だけ平滑化が発火し ±197 cent の f0 ダイバージェンス →
     位相ドリフト。両実装は bit-exact 同一アルゴリズム (クリフはアルゴリズム
     自身の感度)。定常部の f0 一致は 0.0〜0.3 cent。
  3. rust `run` の `resample_linear` を scipy 厳密 `resample_poly` に変更済み。
  - 併せて実機確認済み: `rcwx-cli rt` 10 秒 (FB200 mic → E2x2、定常 underrun 0、
    hop 予算内)、index ブレンド `--index-rate 0.5` が実際に効くこと
    (出力が index=0 と corr 0.982 で変化)。
- **E2E 調査で見つかった形式バグの修正** (いずれも再発防止に回帰テストあり):
  - `train-index` の IVF 書き出しが推論ローダと非互換 (json キー `dim`→`d`、
    slots を u32→i64、`packed_features` をリスト連結→**ID 順テーブル**、
    slots を可変長→`[nlist, cap]` 固定ストライド -1 パディング)。
  - 学習前処理 `butter5_hpf` が 16k 用 scipy 係数を 48k に流用していた
    (sr 引数を無視)。48k 設計の係数に分離。golden は scipy lfilter と
    maxdiff 7.4e-6 (`tools/dump_preproc_golden.py` → `tests/preproc_parity.rs`)。
  - `spectrogram` を f64 FFT 化 — torch.stft (f64) と rtol 0 (f32 だと累積誤差
    ~4e-4 で 1e-4 ゲート超過)。
- **M-T4 3-step パリティ** (`tools/train_step_ref.py` → `tests/step_parity.rs`):
  step 1 の全損失が一致 (d 0.12% / adv 0.34% / mel 0.008% / **kl 0.000%**)、
  mel/kl は 3 step 通して追従 (kl rel 0.00000)。**G param delta cos 0.989**。
  この検証で engine の実バグ 3 件を発見・修正:
  1. `t_mul_window` が `B2[i / n_frames]` でバッチ 1 の窓バッファを読み越え
     (B>1 で後続バッチの窓掛けがゼロ化) → `% n_fft` 修正。実学習は B=1 だった
     ため潜在化していた。
  2. `t_kl_elem`/`t_kl_bwd_*` が [b, t] マスクを [b, inter, t] フラット index で
     読んでいた (B>1 で out-of-bounds) → b/t 復元 index に修正。
  3. `conv2d` backward の dcol matmul で w_row/w_k ストライドが入れ替わって
     いた (dX が全滅) → 修正 + FD gradcheck 追加 (`tests/mpd_gradcheck.rs`)。
- **WN fold × grouped conv (g≥2) dW 逆伝達バグ — 修正済み**。3-step parity 構築で
 発覚した engine の実バグ **7 件**をすべて修正 (いずれも回帰テスト付き):
  1. `t_im2col` がバッチストライドに `cin_g` (グループ内 ch 数) を使っていた
     (batch>1 で後続バッチが別チャネルを読む) → 入力側バッチストライドを
     独立 word (`b_stride = in_c·in_len`) に分離。**forward も壊れていた**。
  2. `conv1d_grouped` / `conv2d` の forward matmul で col オペランドの
     `w_bs=0` (b≥1 が常に b0 の col を再利用)。小テストは x パターンが
     200 要素周期で偶然一致していたため潜在化。
  3. `disc_p` の手元 `oh` 計算が pad=2 なのに `+2` (pad=1 相当) → 出力形が
     py と 2 行ずれる。`(hh + 2·pad − k)/s + 1` に修正。
  4. `t_colsum` が dispatch x 上限 65535 を超える列数で大部分の列を計算せず
     uninit メモリを返していた (dW の norm が数百分の一に崩れる) →
     ワークグループ線形 index (`wid.x + wid.y·512`) に修正。
     **実トレーニングの G/D の大部分の weight 勾配がこれで黙って落ちていた**
     (bias と小さい層だけ学習していた — mel は 197→31.9 に下がっていたため
     気づかず)。修正後 D が本格学習 (disc/fm が実値に)。
  5. KL 損失の mask 参照が [b, inter, t] フラット index で [b, t] マップを
     読み越えていた (B>1) → b/t 復元 index に修正。
  6. `T_REFLECT_PAD` (fwd) が両側 **symmetric** (縁複製) 実装だった
     (正: reflect = 縁複製なし。左 `x[pad−o]`・右 `x[L−2−j]`、numpy
     `'reflect'` / torch `F.pad(..., "reflect")` と同一)。
     **端フレームのみ破損** (t=0/1/36–37、re rms 202/130/25/4 — パッド領域を
     含むフレームだけ。内部フレームは 4e-5 で完全)。DS FFT が疑われたが
     無実: f64 golden 帰属で rust backward は完全 (自 fwd sign 参照と
     cos 1.000001)、fwd sign が 1176/77900 bin (ナイフエッジ) で f64 と
     不一致 → その全量がこの pad バグだった。fwd/bwd を真の reflect に
     修正し既知値回帰テスト (`kahan_ab.rs reflect_pad_exact_reflect`,
     fwd bit-exact・bwd 随伴 0) を追加。段階 bisect 全段階 cos 1.000000。
     (旧記録の「bwd 右ミラー 2L−2→2L−1 修正」は誤った symmetric fwd に
     bwd を合わせる変更だった — fwd が本体のバグ)
  7. `mm` の bias 逆伝達が `bias_axis` を無視して常に bias[n] を計算 —
     bias_axis=1 (cond 系: dec.cond / enc_q.cond_layer, n=1) で
     dec.cond.bias 勾配が 3.4 倍に崩壊。2 段 colsum による bias[m] 経路を追加。
- **D 再 forward を upstream RVC 仕様に統一**: D 更新後に WN を refold して
  更新後重みで G 側 adv/fm を計算 (旧: 更新前 fold を使い、bias だけ更新後の
  混在セマンティクス。py 参照も旧重み snapshot を使っており双方が変則)。
  これで adv も 5 桁一致。
- **M-T4 3-step パリティ完全合格 (計画ゲート)** (`tools/train_step_ref.py` →
  `tests/step_parity.rs`): **全 15 損失 (5 種 × 3 step) が rel ≈ 0**
  (表示 5 桁で全て 0.00000)、**D param delta cos = 1.000000**、
  **G param delta cos = 0.999849 ≥ 0.999** (ゲートは計画値 0.999 に復帰)。
  logmel 勾配は py-f64 golden と cos 1.000000・relnorm 4.9e-6
  (= py-f32 と py-f64 の差と同水準 — 相互 f32 ノイズフロア到達)。
  G delta は 0.989 → 0.992 (engine bug 6/7 件目) → **0.999849**
  (8 件目 reflect fwd バグ修正)。gg0 個別勾配は 143/144 パラメータ cos≥0.99。
- **分割累積 (multi-pass) matmul 実装・実測済み**: 検証指令の
  「中間バッファ経由の分割加算」を `matmul_tiled_kahan` として実装。
  k を chunk 毎 (既定 256・logmel DFT は 16) に分割し、パス間を
  ping-pong GPU バッファ (B4→B5) で接続 — **物理的な演算境界**のため
  naga の式変形で消去不可 (`Graph::matmul_tiled_split`、バッファは
  keepalive で submit まで生存保証)。効果:
  - rust 自身の誤差 (f64 参照比): k=4096 で 1.5×、実 STFT データで
    max 1.7e-5 (単発 tiled は 1.2e-5→良化)。
  - **trainer の G forward z は cos 1.00000000 (相対誤差 median 7.5e-8)**
    — 分割前は 5e-4。
  - **GAN 経路勾配は cos 1.00000・relnorm 2-5e-4 (ほぼ完全一致)**。
  - 一時「残る ~0.008 は py 側 f32 桁数が下限」と結論したが**誤り** —
    py-f32 vs py-f64 は 4.9e-6 なのに rust vs py-f64 は relnorm 1.7e-1
    (cos 0.985875 = rust vs py-f32 と同一) → 残差は rust 側にあった。
    f64 golden による帰属で backward は完全・fwd sign 不一致 (上記
    8 件目の reflect バグ) と確定し、修正で解消。
  - **radix-2 バタフライ FFT**: `T_RFFT2048`/`T_RFFT2048_BWD` (f32) と
    double-single 版 `T_RFFT2048_DS`/`_BWD` (Dekker (hi,lo) 演算・host f64
    twiddle を split 供給・shared memory 32KB/フレーム) を実装し logmel の
    DFT を置換 (n_fft=2048 のみ。他は dft_mm 分割 matmul フォールバック)。
    DS fwd 相対 ~2.5e-8 (f32 版 ~5e-7)、bwd 2e-8。reflect バグ修正後は
    re fwd 誤差 max 3e-5 (py と同水準) で段階 bisect 全段階 cos 1.000000。
    FFT は forward 1 dispatch で chunk-16 分割 (128 dispatch) より高速な
    ため採用。DS 化の意義はバグ発見時に「DFT 精度が原因でない」ことを
    実測で切り分けられた点。
- 長系列 (T=2310, B=2) の generator golden も追加
  (`tools/train_ref2_long.py` → `RCWX_MP2_LONG=1 cargo test -p rcwx-train
  --test module_parity2`)。

- **QoL**: `--mel-loss mono|multi` (MultiScaleMel: 7 スケール win 2048..32・
  hop=win/4・n_mels 320·win/2048)、mel EMA 5 epoch 改善なしで `[OVERFIT?]`
  警告、学習完了後に index 自動生成、D も checkpoint 保存 (`d.` prefix、
  resume で復元 — 旧 checkpointは base-d にフォールバック)。
