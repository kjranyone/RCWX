# RCWX Development Guide

## Project Overview

**RCWX** = RVC Real-time Voice Changer on Intel Arc (XPU)

RVC v2モデルを使ったリアルタイムボイスチェンジャー。現行のリアルタイム処理は
`RealtimeVoiceChangerUnified` を中心とした単一パス実装です。

対象環境は **Windows + Intel Arc (XPU)**。CUDA / NVIDIA は未検証（コード上に分岐は残るがサポート対象外）。

## Quick Start

推奨は同梱ランチャー。`uv sync` / XPU 確認 / モデル有無 / ML デノイザ確認を自動で行う。

```powershell
# uv 未導入時
irm https://astral.sh/uv/install.ps1 | iex

.\rcwx.ps1                    # 環境チェック → 対話メニュー
.\rcwx.ps1 gui                # GUI 直接起動
.\rcwx.ps1 -Denoise gui       # ML デノイザ有効化で GUI
.\rcwx.ps1 download           # 必須モデル DL
.\rcwx.ps1 diagnose           # 診断
```

手動（開発者向け）:

```powershell
uv sync
uv run rcwx download
uv run rcwx diagnose
uv run rcwx

# (オプション) ML Denoiser — CC BY-NC 4.0
uv sync --extra ml-denoise
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
       3) optional denoise (auto/ml/spectral)
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
- GUI の Aggressive では denoise を強制 OFF（UI の保存状態は維持。Normal 復帰で再適用）

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
│   ├── denoise.py         # auto/ml/spectral
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
| `prebuffer_chunks`        |      `1` | GUI では mode で 1 or 3 に再設定 |
| `buffer_margin`           |   `0.25` | GUI: Normal=0.25 / Aggressive=0.1 |
| `input_gain_db`           |    `0.0` | 入力ゲイン |
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
| `use_sola`                    |   `true` | |
| `sola_search_ms`              |   `15.0` | 70Hz 1周期+マージン。GUI 固定 15 |
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
| `denoise.method`              |     `ml` | `auto` / `ml` / `spectral` |
| `denoise.strength`            |    `1.0` | `0.5–2.0`。ML は 1.0 超で2段 |

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
- `sola_search_ms` = 15.0、`use_sola` = true
- `hubert_context_sec` = 1.0、`f0_context_sec` = 0.32
- `f0_method` / `noise_scale` / denoise / postprocess は GUI・config 準拠
- `decoder_overlap_frames` = 5（Aggressive では実行時に 0 扱い）
- `max_queue_size` = 8

## GUI Latency Model (Current)

`LatencySettingsFrame` でユーザーが変えるのは **`chunk_sec` と `latency_mode`**。
他は `_auto_params()` と runtime で決定。

### 自動導出 (`latency_settings.py`)

- `overlap_sec` = chunk の 100%（60–300ms、20ms 刻み）
- `crossfade_sec` = chunk の 10%（10–20ms、10ms 刻み）両モード共通
- `sola_search_ms` = 15.0 固定
- `use_sola` = true
- Normal: `prebuffer_chunks` = 1、`buffer_margin` = 0.25
- Aggressive: `prebuffer_chunks` = 3、`buffer_margin` = 0.1

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
- 開始時・アンダーラン再アームで `prebuffer_chunks`（3）hop 確保
- 最初の 20 hop は 1 hop guard（threshold 1.25 hop）。以降 `max(0.5·hop, p99-p50+callback)` を `0.875·hop` で上限
- HuBERT context 最大 0.56s、SwiftF0 の f0 context 最大 0.10s（保存設定は変更しない）
- GUI 経路で denoise 強制 OFF
- FAISS IVF を warmup 時に XPU 配置（L2 / nprobe=1 / max list ≤256）。未対応は CPU FAISS
- 初回・catchup は実 hop を reflect + `prime_hubert_history` で固定 shape 充填 → 1 hop 目から定常 Synthesizer Graph
- decoder overlap 実行時 0。SOLA 余白は `crossfade + search` のみ
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

SOLA 余白は `crossfade + search`（Normal は + decoder overlap）。リング内の通常 hop 滞留は二重計上しない。

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
```

## References

- RVC WebUI
- PyTorch XPU
- Facebook Denoiser (`denoiser`)
- torchfcpe / swift-f0
- ユーザー向けセットアップ: [docs/SETUP.md](docs/SETUP.md) / [README.md](README.md)
