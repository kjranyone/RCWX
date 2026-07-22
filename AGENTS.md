# RCWX Development Guide

## Project Overview

**RCWX** = RVC Real-time Voice Changer on Intel Arc (XPU)

RVC v2モデルを使ったリアルタイムボイスチェンジャー。現行のリアルタイム処理は
`RealtimeVoiceChangerUnified` を中心とした単一パス実装です。

## Quick Start

```powershell
# 1) 依存関係インストール（PyTorch XPU版を使用）
uv sync

# 2) XPU確認
uv run python -c "import torch; print(torch.__version__, torch.xpu.is_available())"

# 3) 必須モデルをダウンロード（HuBERT / RMVPE）
uv run rcwx download

# 4) (推奨) FCPE低レイテンシF0を追加
uv sync --extra lowlatency
# または: pip install torchfcpe

# 4b) (オプション) SwiftF0超高速F0を追加
pip install swift-f0

# 5) (オプション) ML Denoiser利用可否を確認
uv run python -c "from rcwx.audio.denoise import is_ml_denoiser_available; print(is_ml_denoiser_available())"

# 6) (推奨) フィードバック診断
uv run rcwx diagnose

# 7) GUI起動
uv run rcwx
```

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
       7) soft clip + SOLA crossfade
       8) feedback detection
       9) [Output Queue]
  -> RingOutputBuffer
  -> AudioOutput (48kHz)
```

実装上の要点:

- リアルタイム経路は `pipeline/realtime_unified.py` のみを使用
- チャンク境界連続性は **audio-level overlap** + `infer_streaming()` + SOLA で処理
- PyTorch 2.13のAccelerator GraphをHuBERTと定常状態のRVC Synthesizerに自動適用
- SynthesizerはHuBERT履歴が満杯になるまでeager実行し、開始前warmupで定常Graphをcapture
- `RCWX_ACCELERATOR_GRAPH=0` でAccelerator Graphを無効化可能
- 旧互換の `context_sec` / `lookahead_sec` / `set_context()` / `set_lookahead()` は廃止
- 旧GUIトグルの `use_feature_cache` は廃止
- 過負荷時は一時的に `f0_method="none"` と `index_rate=0.0` に自動退避

## Directory Structure

```text
rcwx/
├── accelerator_graph.py  # XPU/CUDA共通の固定shape Graphキャッシュ
├── cli.py
├── config.py
├── device.py
├── diagnose.py
├── downloader.py
├── audio/
│   ├── input.py           # AudioInput (マイク入力)
│   ├── output.py          # AudioOutput
│   ├── buffer.py          # RingOutputBuffer
│   ├── resample.py        # StatefulResampler
│   ├── sola.py            # simple SOLA
│   ├── denoise.py         # auto/ml/spectral
│   ├── wav_input.py       # WAVファイルループ入力
│   └── stream_base.py     # ストリーム基底
├── models/
│   ├── hubert_loader.py
│   ├── accelerator_index.py # XPU/CUDA常駐IVF検索
│   ├── rmvpe.py
│   ├── fcpe.py
│   ├── swiftf0.py         # SwiftF0 (ONNX/CPU)
│   ├── synthesizer.py
│   └── infer_pack/
├── pipeline/
│   ├── inference.py       # infer / infer_streaming
│   └── realtime_unified.py
└── gui/
    ├── app.py
    ├── realtime_controller.py
    ├── model_loader.py    # 非同期モデルロード
    ├── audio_test.py      # オーディオテスト
    ├── file_converter.py  # ファイル変換
    └── widgets/
        ├── audio_settings.py
        ├── latency_settings.py
        ├── latency_monitor.py
        ├── model_selector.py
        └── pitch_control.py
```

## Configuration (Current)

### `AudioConfig` (`rcwx/config.py`)

| Key                       |  Default | Notes                   |
| ------------------------- | -------: | ----------------------- |
| `sample_rate`             |  `16000` | 内部処理入力レート      |
| `output_sample_rate`      |  `48000` | 出力レート              |
| `chunk_sec`               |    `0.3` | 保存設定上のチャンク長  |
| `latency_mode`            | `balanced` | `balanced` / `aggressive` / `sub100` / `frontier` |
| `prebuffer_chunks`        |      `1` | 出力プリバッファ        |
| `buffer_margin`           |    `0.5` | バッファ余裕            |
| `input_gain_db`           |    `0.0` | 入力ゲイン              |
| `input_channel_selection` |   `auto` | left/right/average/auto |
| `input_hostapi_filter`    | `WASAPI` | Windows向け             |
| `output_hostapi_filter`   | `WASAPI` | Windows向け             |

### `InferenceConfig` (`rcwx/config.py`)

| Key                           |  Default | Notes                           |
| ----------------------------- | -------: | ------------------------------- |
| `f0_method`                   |  `rmvpe` | `rmvpe` / `fcpe` / `swiftf0`    |
| `use_f0`                      |   `true` | F0有効化                        |
| `use_index`                   |   `true` | FAISS有効化                     |
| `index_ratio`                 |   `0.15` | FAISS混合率                     |
| `index_k`                     |      `4` | 近傍数                          |
| `use_compile`                 |  `false` | 既定OFF                         |
| `resample_method`             | `linear` | `linear` / `poly`               |
| `use_parallel_extraction`     |   `true` | HuBERT+F0並列                   |
| `voice_gate_mode`             |    `off` | off/strict/expand/energy        |
| `energy_threshold`            |    `0.2` | energyモード閾値                |
| `overlap_sec`                 |   `0.20` | 音声オーバーラップ              |
| `crossfade_sec`               |   `0.08` | SOLAクロスフェード長            |
| `use_sola`                    |   `true` | SOLA有効化                      |
| `sola_search_ms`              |   `10.0` | SOLA探索窓                      |
| `moe_boost`                   |   `0.45` | F0-only Moeボイススタイル強度 (0.0-1.0) |
| `noise_scale`                 |   `0.45` | 合成ノイズスケール (0.0-1.0)    |
| `f0_lowpass_cutoff_hz`        |   `16.0` | F0ローパスカットオフ (Hz)       |
| `enable_octave_flip_suppress` |   `true` | 1オクターブF0飛び補正           |
| `enable_f0_slew_limit`        |   `true` | フレーム間F0変化量制限          |
| `f0_slew_max_step_st`         |    `3.6` | 最大F0ステップ (semitones)      |
| `denoise.enabled`             |   `true` | ノイズ除去                      |
| `denoise.method`              |     `ml` | `auto` / `ml` / `spectral`      |

### `RCWXConfig` (`rcwx/config.py`) トップレベル

| Key               |                Default | Notes                                             |
| ----------------- | ---------------------: | ------------------------------------------------- |
| `models_dir`      | `~/.cache/rcwx/models` | HuBERT・RMVPEモデルディレクトリ                   |
| `rvc_models_dir`  |                 `None` | RVCモデルディレクトリ（ドロップダウンスキャン用） |
| `last_model_path` |                 `None` | 最後に使用したモデルパス                          |
| `device`          |                 `auto` | auto/xpu/cuda/cpu                                 |
| `dtype`           |              `float16` | float16/float32/bfloat16                          |

### `RealtimeConfig` (`rcwx/pipeline/realtime_unified.py`)

実行時にGUI設定から生成される主要値:

- `chunk_sec` (既定 0.30、20ms境界に丸め)
- `latency_mode` (`balanced` / `aggressive` / `sub100` / `frontier`)
- `overlap_sec` (既定 0.20、20ms境界に丸め)
- `crossfade_sec` (既定 0.08)
- `sola_search_ms` (既定 10.0)
- `prebuffer_chunks` (既定 1)
- `buffer_margin` (既定 0.5)
- `f0_method` (既定 `rmvpe`)
- `noise_scale` (既定 0.45)
- `f0_lowpass_cutoff_hz` (既定 16.0)
- `max_queue_size` (既定 8)

## GUI Latency Model (Current)

`LatencySettingsFrame` では `chunk_sec` と `latency_mode` を変更できます。
他は自動導出されます。

自動導出ルール (`rcwx/gui/widgets/latency_settings.py`):

- `overlap_sec` = chunkの100%（60-300msにクランプ、20ms刻み）
- Balanced: `crossfade_sec` = chunkの25%（10-80msにクランプ、10ms刻み）
- Aggressive: `crossfade_sec` = chunkの10%（10-20msにクランプ、10ms刻み）
- Sub-100: `chunk_sec` = F0方式の下限、`crossfade_sec` = 10ms
- Frontier: SwiftF0/Noneの`chunk_sec` = 20ms、`crossfade_sec` = 10ms
- Balanced/Aggressive: `prebuffer_chunks` = 1
- Sub-100: `prebuffer_chunks` = 2（アンダーラン時も2 hopを再確保して再開）
- Frontier: `prebuffer_chunks` = 3（アンダーラン時も3 hopを再確保して再開）
- Balanced: `buffer_margin` = 0.5
- Aggressive: `buffer_margin` = 0.25、持続リングfloorを0.75 hopでtrimして0.25 hopへ戻す
- Aggressiveの非ASIOコールバック長は最大10ms
- Sub-100: SwiftF0/Noneは40ms、FCPEは100ms、RMVPEは320msが下限
- Sub-100: 最初の20 hopは1.0 hop、その後は`p99 - p50 + callback`を20-35msにクランプしたfloorを維持
- Frontier: 最初の20 hopは1.0 hop、その後は同じ適応式を10-17.5msにクランプ
- Sub-100の非ASIOコールバック長は最大5ms、実行中はDenoiserをバイパス
- Frontierの非ASIOコールバック長は最大2.5ms、実行中はDenoiserをバイパス
- Sub-100/Frontier: HuBERT contextは最大0.56秒、SwiftF0 contextは最大0.10秒
- Sub-100/Frontier: sample rate変換が必要な場合はD2H前にXPU Graph sinc resample
- Sub-100/Frontierでは対応するFAISS IVF indexをwarmup時にXPUへ配置し、HuBERT特徴のCPU往復を省略
- Sub-100/Frontierは初回・catchup後の実音声hopをreflect展開してHuBERT履歴を即時充填し、定常Synthesizer Graphを1 hop目から使用
- FrontierではXPU stage timingを10 hopごと、GUI/GPUメモリtelemetryを10Hzで更新
- HuBERTは推論threadから直接dispatchし、SwiftF0だけを永続workerで並列実行
- 固定64-frame経路では未使用のstreaming feature cloneを作らない
- SOLA境界INFOログは開始3 hopと100 hopごとに限定
- Frontierは未参照のdecoder overlap余白を生成せず、`crossfade + search`だけを保持
- XPU IVFはfeature normを事前計算し、候補距離をnormと内積から求める
- streaming TextEncoderは全frame有効fast pathで全1 mask処理を省略する
- HuBERT/IVF結果の時間軸cacheは双方向contextとの不一致が大きいため使用しない
- ASIO実レートが設定と異なる場合は、ストリーム開始前に実レート用Graphを再ウォームアップ
- `use_sola` = true

補足:

- GUI起動時は `config.audio.chunk_sec` と `latency_mode` を復元し、上記の自動値を再計算します。
- 実行中変更時は `set_overlap()` / `set_crossfade()` / `set_chunk_sec()` を使用します。
- FAISSはTextEncoderのglobal self-attentionを保つため、HuBERT文脈全体を検索します。
- XPU IVFはL2 / nprobe=1 / 最大list長256以下で有効です。未対応indexはCPU FAISSへfallbackします。
- XPU IVFは再構築特徴をFP16/BF16で保持し、158k×768 indexでは約240MBの追加VRAMを使います。
- SwiftF0のF0補正はCPU上で完結し、pitch/pitchfを合成直前に一度だけXPUへ転送します。
- 推論統計は直近256 hopのp50/p95/p99とdeadline miss率を保持します。
- Sub-100/Frontierの出力guardは推論jitter統計から適応し、アンダーラン後はそれぞれ2/3 hopを再バッファします。
- SOLA合成末尾は`crossfade + search + decoder overlap`を10msフレームへ切り上げます。Frontierのdecoder overlapは0です。

## CLI Commands

```powershell
uv run rcwx                  # GUI起動（command省略時デフォルト）
uv run rcwx gui              # GUI起動
uv run rcwx devices          # デバイス一覧
uv run rcwx download         # 必須モデルダウンロード
uv run rcwx run in.wav model.pth -o out.wav --pitch 5 --f0-method rmvpe
uv run rcwx info model.pth   # モデル情報
uv run rcwx diagnose         # フィードバック診断
uv run rcwx logs             # ログ一覧
uv run rcwx logs --tail 50   # 最新ログ末尾
uv run rcwx logs --open      # 最新ログを開く
```

## Logs & Diagnostics

ログ保存先:

- `~/.config/rcwx/logs/rcwx_YYYYMMDD_HHMMSS.log`

よく出るログ:

- `[INPUT] Queue full, dropping chunk`
- `[INFER] Chunk #...`
- `[PERF] Inference slow ...`
- `[FEEDBACK] Detected feedback (corr=...)`
- `[WARMUP] ...`

トラブルシュート:

- Queue full が頻発: `chunk_sec` 増、デノイズOFF、F0方式見直し
- 遅延増加: `chunk_sec`/`buffer_margin` 見直し、出力デバイス設定確認
- フィードバック警告: `uv run rcwx diagnose` 実施、入出力ループ回避
- XPU未認識: `uv sync` 後に `torch.__version__` が `+xpu` か確認

## Models

既定配置:

- `~/.cache/rcwx/models/hubert/hubert_base.pt`
- `~/.cache/rcwx/models/rmvpe/rmvpe.pt`
- RVC本体モデル: 任意の `*.pth`
- FAISS index: モデル隣接の `*.index`（自動検出）

## Tests

主なテスト:

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
```

## References

- RVC WebUI
- PyTorch XPU
- Facebook Denoiser (`denoiser`)
- torchfcpe
