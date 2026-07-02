# コアパイプライン精読メモ

対象: `rcwx/pipeline/realtime_unified.py`（1325行）、`rcwx/pipeline/inference.py`（2417行）、`rcwx/config.py`（211行）

## realtime_unified.py — RealtimeVoiceChangerUnified

### スレッドモデル（概ね健全）

関与スレッド: (a) GUI/メインスレッド、(b) 入力コールバックスレッド、(c) 出力コールバックスレッド、(d) 推論スレッド `RCWX-Inference-Unified`。

- 入力コールバック → `_input_queue`（`queue.Queue`）→ 推論スレッド → `_output_queue` → 出力コールバックが `RingOutputBuffer` に注入して再生（`realtime_unified.py:746-824`）。生産者/消費者境界が Queue にあり、リングは出力コールバック専有。**この分離は正しい**。
- 推論スレッドのバックログ処理: 滞留分を捨てて最新チャンクのみ処理し、全ステートをリセット（`realtime_unified.py:836-854` の CATCHUP 処理）。レイテンシ有界化として妥当。
- `stop()` は `join(timeout=1.0)` 後に参照を捨てる（`realtime_unified.py:606-609`）。join タイムアウト時にスレッドが生きたまま参照だけ消える余地があるが、`_running=False` 済みなので実害は限定的。

### 指摘事項

1. **`set_crossfade()` のレース**（`realtime_unified.py:705-715`）: GUI スレッドが稼働中に `self._sola_state = SolaState(...)` で丸ごと差し替える。推論スレッドが `sola_crossfade`（`realtime_unified.py:964-969`）実行中・実行間だと、前チャンクのテール（`SolaState.buffer`）を失い境界クリックの可能性。`set_chunk_sec()` が stop/start で作り直すのと対照的に、こちらは無防備。

2. **`set_postprocess_config()` のカプセル化破壊**（`realtime_unified.py:669-682`）: `self._postprocessor._treble._design_filter()`、`._limiter._threshold`、`._limiter._release_coeff` と private 2階層に手を突っ込み、`Postprocessor.__init__` にあるはずの係数計算式（dB→線形、release係数）を呼び出し側で複製している。`Postprocessor.update_config(cfg)` メソッドに畳むべき。同様に `_recalculate_sizes()` が `self._sola_state._hann_fade_in = None`（`realtime_unified.py:390-391`）と SolaState の private を直接触る。

3. **コンストラクタが重い**（`realtime_unified.py:193-220`）: `__init__` でモデルロード + GPU ウォームアップ2回を実行。構築 = 副作用（数秒〜数十秒）であり、テスト・再構成を難しくしている。`prepare()` 等に分離する余地。

4. **config 直読みのランタイム変更**: 推論スレッドは毎チャンク `self.config.*` を直読みし、GUI スレッドが setter（`realtime_unified.py:634-742`）で書く。単一 float/bool の書き換えは GIL 下で実害は小さいが、`set_crossfade`（上記1）と `_apply_runtime_sample_rates` によるオブジェクト差し替え（`output_buffer`、resampler、`realtime_unified.py:406-425`）は「`_running=False` の間しか呼ばれない」という暗黙の前提に依存。前提はコードのどこにも明文化されていない。

5. **過負荷保護の実装とドキュメントの乖離**: `_is_overloaded()` の効果は **denoise バイパスのみ**（`realtime_unified.py:878`、警告ログも "temporarily bypassing denoise" `realtime_unified.py:1145`）。CLAUDE.md の「一時的に `f0_method="none"` と `index_rate=0.0` に自動退避」は旧実装の記述。

6. **GPU メモリ計測のインライン化**（`realtime_unified.py:991-1014`）: 推論ループ本体に `import torch` + XPU/CUDA 分岐がベタ書き。`device.py` 系ヘルパに出すべき小さな整理対象。

7. **テスト用 API `process_next_chunk()` は誤解を招く**（`realtime_unified.py:1312-1321`）: Queue から取り出して**末尾に戻す**だけなので、複数チャンク滞留時は順序が入れ替わる。「simplified test interface」とコメントされているが、実質何もしない/害があるので削除候補。

8. **`_on_audio_input` の蓄積バッファ**（`realtime_unified.py:751`）: 毎コールバック `np.concatenate` で再確保。チャンクの1/4サイズ・数十ms周期なので実害は小さいが、リング化すればアロケーションフリーにできる。

### 設計として良い点

- 適応ドリフト制御（`realtime_unified.py:782-818`）: バッファ超過時にサンプルを捨てず `np.interp` で ≤1.5% 時間圧縮。`x_base` のキャッシュ（`realtime_unified.py:336-338, 803-806`）までやっており、コールバック内アロケーション意識が高い。
- ASIO duplex の単一ストリーム化と二重 close 防止（`realtime_unified.py:461-512, 611-624`）。
- 出力ストリームを入力より先に開始して初期バーストを防ぐ順序制御（`realtime_unified.py:514-528` のコメント付き実装）。
- 実レート（要求と異なるデバイスレート）への追従 `_apply_runtime_sample_rates`（`realtime_unified.py:406-425, 550-579`）。

## inference.py — RVCPipeline

### 最大の問題: infer / infer_streaming の二重実装

`infer()`（`inference.py:963-1801`、約840行）と `infer_streaming()`（`inference.py:1803-2398`、約600行）に以下がほぼコピペで並存:

| 重複ブロック | infer | infer_streaming |
|---|---|---|
| pre-HuBERT シフト計算（median F0 推定→adaptive shift） | 1184-1216 | 1975-2007 |
| HuBERT+F0 並列抽出（ThreadPoolExecutor + クロージャ） | 1239-1272 | 2022-2056 |
| F0 逐次フォールバック（if/elif 連鎖） | 1377-1399 | 2086-2104 |
| F0 後処理チェーン（smooth→lowpass→octave→slew） | 1490-1501 | 2152-2163 |
| mel 量子化（pitch/pitchf 生成） | 1511-1528 | 2174-2183 |
| voice gate（expand/energy/補間/平滑） | 1663-1724 | 2334-2375 |
| 短入力パディング（reflect/replicate 分岐） | 1576-1613 | 2262-2281 |

さらに**既に乖離している**: `infer` のみ feature-cache blending（1303-1350）と `stabilize_f0_boundaries`（1501）を適用し、streaming は別系統のキャッシュ（`_streaming_feat_cache` / `_streaming_audio_history` / `_streaming_f0_pre_filter_tail`）+ F0 フィルタ連続性のための履歴 prepend（2128-2169）を使う。どちらが「正」か判断できる場所がコード上に無い。

### 状態管理の問題

- **キャッシュが2系統×7属性**: バッチ系（`_feature_cache` / `_f0_cache` / `_f0_voiced_cache` / `_audio_cache` / `_stream_history`）とストリーミング系（`_streaming_feat_cache` / `_streaming_audio_history` / `_streaming_f0_pre_filter_tail`）。`_output_cache`（`inference.py:726-727`）は**定義・クリアされるだけで一切使われないデッド状態**。
- **ストリーミング系3属性は `__init__` で未定義**、`clear_cache()`（`inference.py:930-946`）でのみ生成。リアルタイム経路は `RealtimeVoiceChangerUnified.__init__` → `_run_warmup` → `infer` → `clear_cache()`（`realtime_unified.py:1181`）が先に走るため**偶然動いている**。`RVCPipeline` を直接生成して `infer_streaming()` を呼ぶと `inference.py:2227` で AttributeError。
- 型注釈 `self.hubert: Optional[HuBERTFeatureExtractor]`（`inference.py:692`）は削除済みクラスへのデッド参照（実体は `HuBERTLoader`）。

### インターフェース肥大

`infer_streaming()` は引数18個、`infer()` は22個。呼び出し側（`realtime_unified.py:935-955` 本番、`1219-1238` warmup）で全引数を1個ずつ転記しており、パラメータ追加1回につき最低4箇所（+GUI 側2箇所）の変更が要る。`StreamingParams` dataclass 1個に畳むのが費用対効果最大のリファクタ。

### ホットパスの性能

- F0 後処理チェーンで GPU→CPU→GPU 往復が最大4回/チャンク: `smooth_f0_spikes`（`inference.py:303`）、`lowpass_f0`（341）、`suppress_octave_flips`（507）、`limit_f0_slew`（539）が個別に `.cpu().numpy()` → `torch.from_numpy().to(device)`。後2者はフレーム単位の Python 二重ループ。1回降ろして4フィルタ通して1回戻す構造に統合可能。
- `pitch_shift_resample`（`inference.py:206-259`）も GPU→CPU→scipy→GPU の往復を毎チャンク実施（pre_hubert_pitch_ratio > 0 時）。
- `infer()`（バッチ経路）は per-chunk の `logger.info` が10箇所超（1170, 1219, 1281, 1362, 1529 等）。CLI バッチでは大量出力になる。streaming 側は静かで非対称。

### その他

- `load()` がグローバル副作用を持つ: `torch.manual_seed(0)`、cudnn deterministic 設定（`inference.py:734-741`）。ライブラリとしての行儀が悪く、同プロセス内の他の torch 利用に影響。
- `_set_fixed_harmonics()`（`inference.py:948-961`）は `synthesizer.model.dec.m_source.l_sin_gen` と4階層の内部構造に依存（Law of Demeter 違反）。`SynthesizerLoader` のメソッドにすべき。
- マジック定数の分散: `hubert_hop = 320` がファイル内外で再定義（`inference.py:1091, 1864` / `realtime_unified.py:228, 373, 888`）、`t_pad = 800`（`inference.py:1888`）、`zc = sample_rate // 100` の導出が複数箇所。

### 設計として良い点

- 固定サイズ HuBERT 入力による XPU カーネル再コンパイル回避（`inference.py:1940-1954`、コメントで理由まで明記）。
- `skip_head` / `return_length` による「TextEncoder は全コンテキスト・Decoder は出力領域のみ」の合成（`inference.py:2290-2331`）。SineGen 位相蓄積アーティファクト対策として設計意図がコメントされている。
- F0 フィルタのクロスチャンク連続性（フィルタ前テールを保持して次チャンクで prepend、`inference.py:2128-2169`）。

## config.py

- 素の dataclass 5種 + JSON 永続化。バリデーション無し（文字列 enum は typo が実行時までサイレント）。
- **マイグレーションの非対称性**: 未知キーのフィルタは `InferenceConfig` のみ（`config.py:176-182`）。`AudioConfig(**audio_data)`（185）とトップレベル `cls(**data)`（191）は未知キーで TypeError → **旧 config が残っていると起動失敗**。既知の旧キー2個だけ手動 pop（160-163）している場当たり対応。
- `AudioConfig.crossfade_sec`（26行）は `InferenceConfig.crossfade_sec`（97行）と重複した死にフィールド（保存は inference 側のみ）。
- `PostprocessConfig` の normalizer_* 5フィールド（58-62行）は `RealtimeConfig` に運ばれず（`realtime_unified.py:142-147, 247-253` に normalizer 無し）、ランタイムは `audio/postprocess.py` 側デフォルトを常用。**設定がランタイムに届かない死んだ配線**（現状は両者のデフォルト値が一致しているため挙動上は無害）。
