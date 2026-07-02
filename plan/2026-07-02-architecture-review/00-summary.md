# RCWX アーキテクチャレビュー 総評

- 実施日: 2026-07-02
- 対象ブランチ: `feature/improve202602`（未コミット変更: `rcwx/gui/app.py` +5行）
- 調査方法: コアパイプライン2ファイル（`realtime_unified.py` 1325行 / `inference.py` 2417行）+ `config.py` を精読、audio / GUI / models 各層を並列調査
- 詳細レポート:
  - [01-realtime-pipeline.md](01-realtime-pipeline.md) — コアパイプライン精読メモ
  - [02-audio-layer.md](02-audio-layer.md) — audio層
  - [03-gui-layer.md](03-gui-layer.md) — GUI層
  - [04-models-layer.md](04-models-layer.md) — models層 + device/config/cli/downloader/diagnose

## 総評

レイヤー構造そのもの（audio = leaf層、pipeline = コア、gui = 最上位）は健全で、リアルタイム経路のスレッド設計（Queue による生産者/消費者分離、リングバッファの単一スレッド専有）は堅実。一方で、**「実験の速度を優先して積み上げた配線」が3箇所で限界に近づいている**:

1. `pipeline/inference.py` の巨大な二重実装
2. パラメータ1個の追加に約6箇所の手作業配線が要る構造
3. GUI 層の結合スタイルの不統一

加えて、実害のあるバグ・ドキュメント乖離が複数見つかった。

## 良い点

- **audio層は完全な leaf 層**で pipeline/gui への上向き依存ゼロ。`RingOutputBuffer` はロック無しだが、出力コールバックスレッドが専有し推論スレッドとは `queue.Queue` 境界で分離されており、設計として正しい（`realtime_unified.py:763-824`）。
- **スレッド間 marshalling は概ね正しい**: モデルロード・変換・stats 更新はすべて `app.after(0, ...)` でメインスレッドに戻している。
- **infer_pack/ は境界が明確な移植コード**。RVC WebUI からの移植に RCWX 固有ロジックが侵入したのは SineGen の `fixed_harmonics` と `RCWX_SINEGEN_MODE` の2点のみで、循環依存もなし。
- **device/dtype は引数渡しで一貫**（グローバル状態なし）。SwiftF0 のみ例外（ONNX/CPU固定）。
- ドリフト制御を「サンプル破棄」でなく「≤1.5% の時間圧縮」で行う設計（`realtime_unified.py:782-818`）や、XPU のカーネル再コンパイルを避ける固定サイズ HuBERT 入力（`inference.py:1940-1954`）など、リアルタイム品質への配慮が行き届いている。

## 構造的問題（優先度順）

### 1. `pipeline/inference.py` の infer / infer_streaming 大規模重複 — 最優先

2417行のファイルに `infer()`（約840行）と `infer_streaming()`（約600行）が並存し、**pre-HuBERT シフト計算・HuBERT+F0 並列抽出クロージャ・F0後処理チェーン・mel量子化・voice gate がほぼコピペで二重化**している（例: 並列抽出は `inference.py:1239-1272` と `2022-2056`、voice gate は `1663-1724` と `2334-2375`）。既に両者は静かに乖離しており（`infer` のみ feature-cache blending と `stabilize_f0_boundaries` を適用、streaming は別のキャッシュ機構）、片方だけ直すバグ修正が構造的に発生しやすい。F0後処理は既にモジュールレベル関数群なので、「抽出 → F0チェーン → 合成」の共通部を関数に括り出す余地は大きい。

### 2. パラメータ追加の「配線税」— 1ノブ ≒ 6箇所

新しい調整パラメータを1個足すたびに、以下の約6箇所を手で揃える必要がある:

1. `InferenceConfig`（config.py）
2. `RealtimeConfig`（realtime_unified.py）
3. `RealtimeController.start()` の約50行の集約（realtime_controller.py:88-134）
4. `infer_streaming()` の引数（現在18個）
5. GUI ウィジェット + コールバック
6. `_save_config()` の手動列挙（app.py:1012-1050）

`fixed_harmonics` や `f0_slew_max_step_st` の追加履歴がまさにこのコストを払っている。`infer_streaming` にパラメータオブジェクト（dataclass）を渡す形にするだけでも、シグネチャ肥大と呼び出し側2箇所（本番 + warmup `realtime_unified.py:1219-1238`）の重複が消える。

### 3. GUI: app.py の God Object 化と結合スタイルの二重基準

- `app.py`（1145行）が約13の責務を抱え、特に `_setup_main_tab`（約340行）に Index/Denoise/VoiceGate 等の枠がウィジェット化されずベタ書き。
- **結合スタイルが2系統混在**: `widgets/` はコールバック注入で疎結合（良い）のに対し、`realtime_controller` / `audio_test` / `file_converter` / `model_loader` は `app` 逆参照で20以上の属性を直接読む密結合。
- ランタイム更新は controller を素通りして `app → realtime_controller.voice_changer.set_*()` の2階層貫通が約15箇所。未コミットの diff（app.py:971-975 の input_gain 直接代入）も setter を介さず `config` フィールドを直書きしており、この不整合をさらに一段深めている。`set_input_gain_db()` を追加して既存パターンに揃えることを推奨。
- **Tk スレッド安全性違反の可能性が2箇所**: ワーカースレッドから `ctk Variable.get()` を呼んでいる（`audio_test.py:198-207`、`file_converter.py:109-118`）。値をスレッド起動前にメインスレッドで読んで渡すべき。

### 4. config の三重定義と既定値の分散

- `DenoiseConfig` / `PostprocessConfig` が **config.py と audio層で二重定義**（`config.py:38,49` vs `audio/denoise.py:24`、`audio/postprocess.py:16`）。さらに `RealtimeConfig` が postprocess 5フィールドだけを運ぶため、**config.py の normalizer_* 設定はランタイムに一切届かない**（`realtime_unified.py:247-253` は normalizer を渡さず、audio層のデフォルトが常に使われる）。現状はデフォルト値が偶然一致しているだけの死んだ配線。
- 既定値の分散: `noise_scale` が 0.45 / 0.4 / 0.66666 の3値で4箇所に、F0 threshold がクラス既定とパイプライン定数で不一致（RMVPE 0.03 vs 0.015）、`use_compile` は CLI 既定 True vs config 既定 False で逆。
- `AudioConfig.crossfade_sec`（config.py:26）は `InferenceConfig.crossfade_sec` と重複した死にフィールド。

### 5. F0抽出器: 抽象の欠如 + 3種無条件ロード

RMVPE / FCPE / SwiftF0 は `infer(audio, threshold)` の慣習だけで揃えたダックタイピングで、`if/elif` ディスパッチが inference.py 内に4回重複。さらに `load()` は設定の `f0_method` に関わらず**3種すべてを eager ロード**する（`inference.py:774-814`）ため、起動時間とメモリを常時消費。Protocol 1枚 + レジストリ（遅延ロード）で両方解決できる。

## 実害のあるバグ・脆弱箇所

| 箇所 | 内容 |
|---|---|
| `audio/denoise.py:276-277` | `SpectralGateDenoiser.reset()` が `__init__` に存在しない `input_buffer`/`output_buffer` を参照 — **呼べば必ず AttributeError** |
| `config.py:185,191` | 未知キーフィルタが InferenceConfig にしか無く、**audio セクション/トップレベルに旧キーが残ると load() が TypeError** で起動失敗 |
| `inference.py:2227` ほか | `_streaming_feat_cache` 等3属性が `__init__` でなく `clear_cache()` でのみ定義。現状は warmup 経由で偶然動くが、`RVCPipeline` 直生成 → `infer_streaming()` で AttributeError になる脆い初期化 |
| `inference.py:692` | 型注釈 `Optional[HuBERTFeatureExtractor]` は**削除済みクラスへのデッド参照**（実体は `HuBERTLoader`。`from __future__ import annotations` で偶然生きている） |
| `realtime_unified.py:705-715` | `set_crossfade()` が稼働中に `_sola_state` を丸ごと差し替え。推論スレッドが `sola_crossfade` 実行中だと**前チャンクのテールを失い境界クリックの可能性**（レース） |
| `audio/stream_base.py:157-166,240-244` | ストリーム open 成功 → start 失敗時に `.close()` されず**フォールバック試行ごとに未クローズの PortAudio ストリームが蓄積**。`duplex.py` コンストラクタも同型 |
| `device.py:22-34` | `preferred="cuda"` 未搭載時など、**警告なしで別デバイスにサイレントフォールバック** |
| `audio/input.py:147-149`, `output.py:66-88` | コールバック例外の保護がなく、例外1発でストリーム即死（`duplex.py` は保護あり — 非対称） |

## CLAUDE.md との乖離（要更新）

- 「過負荷時は `f0_method="none"` と `index_rate=0.0` に自動退避」→ **実装は denoise バイパスのみ**（`realtime_unified.py:1137-1156`）。
- ディレクトリ構成に `audio/postprocess.py`（227行）、`audio/duplex.py`（210行）、`widgets/postprocess_settings.py` が未記載。
- `RealtimeConfig` の記載に `decoder_overlap_frames`、postprocess 系、`fixed_harmonics` が未反映。

## パフォーマンス上の注意（ホットパス）

- **F0後処理チェーンで GPU↔CPU 往復が最大4回/チャンク**（`smooth_f0_spikes` → `lowpass_f0` → `suppress_octave_flips` → `limit_f0_slew` がそれぞれ `.cpu().numpy()` → GPU 復帰）。しかも後2者はフレーム単位の Python ループ。1回 CPU に降ろして全フィルタを通し1回で戻す形に統合可能。
- **スライダードラッグ中、1ステップ毎に config 全体を JSON フルダンプ**（`_on_*_changed` → `_save_config`）。デバウンスするだけで解消。
- SwiftF0 の Viterbi/DP オクターブ補正（約170行の Python ネストループ）が毎推論走る。

## 推奨する着手順

1. **バグ修正（小粒・即効）**: `SpectralGateDenoiser.reset()`、config load の非対称フィルタ、`_streaming_feat_cache` の `__init__` 移動、input_gain setter 追加、Tk `Variable.get()` の2箇所。
2. **CLAUDE.md 更新**（overload 動作・ディレクトリ構成）— ドキュメント乖離は将来の自分への誤誘導なので安い割に効く。
3. **`infer_streaming` のパラメータオブジェクト化** — 配線税を下げ、以降の実験サイクルが速くなる。
4. **inference.py の共通部抽出**（F0チェーン・抽出・gate を関数化して infer/infer_streaming で共有）。
5. **GUI は「controller にランタイム setter を集約」から** — app.py 全面分割より先に、2階層貫通アクセス15箇所を controller 経由に寄せるだけで結合が大きく改善する。
