# rcwx/audio アーキテクチャレビュー

対象: `input.py`, `output.py`, `buffer.py`, `resample.py`, `sola.py`, `denoise.py`, `wav_input.py`, `stream_base.py`, `duplex.py`, `postprocess.py`（依存把握のため消費側 `pipeline/realtime_unified.py` も確認）

## 1. 各モジュールの責務と公開API

### `stream_base.py` (447行) — デバイスI/Oの基盤
- `AudioStreamBase(ABC)` (17): 入出力共通の基底。WASAPI→DirectSound→MME→default のAPIフォールバックを実装。
  - クラス変数 `STREAM_TYPE`/`STREAM_CLASS` (26-27) をサブクラスが設定
  - `actual_sample_rate` プロパティ (43-46)
  - `_audio_callback` 抽象メソッド (48-57)
  - `_get_api_preferences` (59-103)、`_find_device_on_api` (105-117)、`_get_device_sample_rates` (119-139)、`_get_blocksize_options` (141-147)、`_try_open_stream` (149-166)
  - `start` (197-251) / `stop` (253-262) / `is_active` (264-267)
- `AudioStreamError(Exception)` (270)
- モジュール関数: `list_devices` (276-325)、`get_default_device` (328-333)、`query_asio_channel_names` (336-392, ctypesでPortAudio DLL直叩き)、`get_asio_hostapi_index` (395-403)、`is_device_on_asio` (406-421)、`query_asio_native_sample_rate` (424-446)

### `input.py` (154行) — マイク入力
- `_auto_select_channel` (17-59)、`select_channel` (62-104): マルチch→モノ変換
- `AudioInputError(AudioStreamError)` (107-110)
- `AudioInput(AudioStreamBase)` (113): `STREAM_CLASS = sd.InputStream` (121)。`__init__` (123-134)、`_audio_callback` (136-149)
- `list_input_devices` (151-153)

### `output.py` (93行) — スピーカー出力
- `AudioOutputError(AudioStreamError)` (17-20)
- `AudioOutput(AudioStreamBase)` (23): `STREAM_CLASS = sd.OutputStream` (31)。`__init__` (33-53, `output_channel_selection`をindexペアにパース)、`_audio_callback` (55-88, mono→指定chルーティング)
- `list_output_devices` (90-92)

### `buffer.py` (158行) — リング出力バッファ
- `RingOutputBuffer` (12): `__init__` (20)、`add` (39-75, 上書きでオーバーフロー処理)、`get` (77-113, アンダーラン時フェード)、`_read` (115-128)、`clear` (130-137)、プロパティ `available`/`free`/`samples_dropped`/`underrun_count` (139-157)

### `resample.py` (160行) — リサンプリング
- `resample()` (12-49, poly/linear)
- `StatefulResampler` (52): `__init__` (67)、`resample_chunk` (103-155, overlap-save方式)、`reset` (157-159)

### `sola.py` (221行) — SOLAクロスフェード
- `SolaState` dataclass (26-48)
- `sola_crossfade` (51-138)、`sola_flush` (141-154)、`_find_best_offset` (157-220, 正規化相互相関)

### `denoise.py` (496行) — ノイズ抑制
- `DenoiseConfig` dataclass (23-47)
- `MLDenoiser` (54): `_load_model` (82-116, 遅延ロード+CPUフォールバック)、`process` (118-176)
- `SpectralGateDenoiser` (183): `learn_noise` (238)、`enable_auto_learn` (266)、`reset` (274-281)、`_compute_gain` (293)、`process_frame` (339)、`process` (375-419)
- モジュール: `_get_cached_ml_denoiser` (426-431, グローバルキャッシュ)、`denoise` (434-486)、`is_ml_denoiser_available` (489-495)

### `wav_input.py` (142行) — WAVループ入力（AudioInput互換）
- `WavFileInput` (17): `_load_wav` (52-79)、`actual_sample_rate` (81-83)、`start` (85-95, スレッド起動)、`stop` (97-101)、`_playback_thread` (103-141)

## 2. スレッド安全性

**関与するスレッド**: (a) メイン/GUIスレッド、(b) sounddeviceのオーディオコールバックスレッド、(c) 推論スレッド `RCWX-Inference-Unified`、(d) `WavFileInput` の `RCWX-WavInput`。

**audio層のどのモジュールにも `threading.Lock`/`Event` は存在しない**（`wav_input.py` が `threading.Thread` を使うのみ）。ロックが無くても安全にできているのは設計上の分離による:

- **RingOutputBuffer はロック無しだが単一スレッド専有**。`add()`/`get()`/`available` は全て消費側 `_on_audio_output`（出力コールバックスレッド）からのみ呼ばれる (`realtime_unified.py:771, 799, 820, 789`)。推論スレッドは `_output_queue`（thread-safeな `queue.Queue`）経由でのみ受け渡し、リングには直接触れない。生産者/消費者境界がQueueにあるため、リング自体はシングルスレッド。**妥当な設計**。

- **ロック無しで跨スレッド共有される可変状態（指摘事項）**:
  1. `RingOutputBuffer._count` を推論スレッドが `output_buffer.available` 経由で読む (`realtime_unified.py:1022`) 一方、出力コールバックスレッドが `add/get` で書き換える。int読み取りのみでレイテンシ表示用途なのでCPythonでは実害は小さいが、非同期アクセス。
  2. `start()` が `self.output_buffer` の参照を差し替え (`realtime_unified.py:418`, `_apply_runtime_sample_rates` 内、569から呼ばれる) た後、コールバックが動く可能性。ただし `_running=False`（581でTrueにする前）かつ `_on_audio_output` は `not self._running` で早期return (765-766) するためガードされている。安全マージンは `_running` フラグ依存。
  3. `AudioStreamBase.is_active` (266) は `self._stream.active` を読むが、`stop()` (253) が別スレッドから `self._stream=None` にできる。並行時 `NoneType.active` でAttributeError の余地（軽微）。

- **WavFileInput の同期は素の bool**: `self._running` をメインスレッドが `start/stop` で書き、`_playback_thread` が読む (86, 98, 109, 131)。`threading.Event` ではなく plain bool。`self._pos` は再生スレッドが書く。停止は `_running=False` → `join(timeout=2.0)` (97-100)。実害は小さいが、フラグは可視性保証の弱い共有変数。

- **コールバック内例外の扱いに不整合**: `duplex.py` の `_duplex_callback` は入力側・出力側を個別に `try/except` で包み、コールバック例外でストリームが `paAbort` で落ちないよう保護 (`duplex.py:172-210`)。一方 **`input.py` の `_audio_callback` (147-149) と `output.py` の `_audio_callback` (66-88) はユーザーコールバックを裸で呼んでおり**、`self._callback` が例外を投げるとストリームが即死する。同じコールバック契約なのに保護レベルが違う。

- **denoise のグローバルキャッシュ**: `_ml_denoiser_cache` (423) は無ロックで `_get_cached_ml_denoiser` (426-431) が初期化。初回同時呼び出しでモデル二重生成の余地（推測: 実際は推論スレッド単一からの呼び出しなので顕在化しにくい）。

## 3. モジュール間の依存関係

**audio層は完全な下位（leaf）レイヤーで、pipeline/gui への上向き依存はゼロ**（確認済み）。層内依存のみ:

- `input.py` → `stream_base.py` (12)
- `output.py` → `stream_base.py` (12)
- `duplex.py` → `input.py`(`select_channel`), `stream_base.py`(`query_asio_native_sample_rate`) (22-23)
- `wav_input.py` → `resample.py` (関数内import, 54)
- `buffer.py` / `resample.py` / `sola.py` / `denoise.py`: audio層内の他モジュールへの依存なし（純粋なユーティリティ）
- `__init__.py` が `buffer/denoise/input/output/resample` を再エクスポート

**消費側（上位層→audio層への依存）**:
- `pipeline/realtime_unified.py`: `RingOutputBuffer`(22), `AsioDuplexStream`(24), `AudioInput`(25), `AudioOutput`(26), `WavFileInput`(531), `get_default_device`(476/479)
- `gui/widgets/audio_settings.py`: `list_input_devices`, `select_channel`, `list_output_devices` (12-13)
- `cli.py`: `list_input_devices`, `list_output_devices` (104-105)

## 4. リソースのライフサイクル管理

- **`AudioStreamBase.start()` にストリームリーク**: `_try_open_stream` (157-166) は `self._stream = self.STREAM_CLASS(...)` で生成後に `.start()` を呼ぶ。`.start()` が例外を投げると、生成済み（未開始）のストリームオブジェクトが `self._stream` に残る。呼び出し元 (240-244) の except は `self._stream = None` に**上書きするだけで `.close()` を呼ばない**。生成に成功して開始で失敗したPortAudioストリームは明示クローズされず、GC/`__del__` 頼み。フォールバックで多数の組み合わせを試すため、失敗ごとに未クローズオブジェクトが積まれる。
- **`duplex.py` コンストラクタも同型**: レートループで `sd.Stream(...)` 生成後に例外時、次イテレーションで上書き（108-125）。生成済み失敗ストリームを close せず。
- **正常系の停止は良好**: `stream_base.stop()` (253-262) と `duplex.stop()` (143-154) は `try/finally` で `_stream=None`。`duplex` は `_stopped` フラグで二重close防止 (144-146)、`realtime_unified.stop()` はASIO duplex時に同一オブジェクトを一度だけ停止 (611-624)。
- **`AsioDuplexStream.start()`** (139-141): `self._stream.start()` が例外を投げた場合のハンドリング無し（そのまま伝播）。
- **`WavFileInput`**: `_playback_thread` 内で `self._callback` が例外を投げるとスレッドが静かに死に `_running` は True のまま残る (133-134 に保護なし)。`stop()` の `join(timeout=2.0)` は妥当。
- **`MLDenoiser`**: モデル解放（GPUメモリ）を明示するclose/`__del__` は無く、プロセス終了/GC任せ（推測: 常駐前提なので許容範囲）。

## 5. 設計上の問題

- **`SpectralGateDenoiser.reset()` は壊れている（デッド/バグ）**: 276-277行で `self.input_buffer.fill(0)` / `self.output_buffer.fill(0)` を呼ぶが、これらの属性は `__init__` (201-230) で**一切定義されていない**。`reset()` を呼ぶと `AttributeError`。`process()` はステートレスな overlap-add で内部バッファを持たないため、これらは実在しない属性。
- **出力ch ルーティングの重複コード**: `output.py` の `_audio_callback` (68-88) と `duplex.py` の `_duplex_callback` 出力側 (188-210) がほぼ完全に同一。さらに `output_channel_selection` のindexペア解析も `output.py` (46-53) と `duplex.py` (75-82) でコピペ。
- **resample_poly の gcd/up/down 算出が3箇所に重複**: `resample.py` (43-49) / `StatefulResampler.__init__` (85-89) / `denoise.py MLDenoiser.process` (142-145, 165-167)。`MLDenoiser.process` は `from scipy import signal` を関数内で2回import (141, 164)。
- **未使用import**: `get_default_device` が `input.py` (12) と `output.py` (12) でimportされるが両ファイル内で未使用。
- **denoise のデフォルト値不整合**: `denoise()` の引数デフォルト `threshold_db=6.0` (439) が `DenoiseConfig.threshold_db` のデフォルト `-20.0` (37) を上書きする (478)。`SpectralGateDenoiser` を直接使う場合(-20dB)と `denoise()` 経由(+6dB)でゲート閾値が大きく変わる（`threshold_mult` が 0.1x vs 2.0x）。入口によって挙動が異なる。
- **channel_selection デフォルトの不統一**: `AudioInput` は `"average"` (130)、`AudioOutput`/`AsioDuplexStream` は `"auto"` (40, 68)。
- **ハードコード値**: `fade_samples=256`（消費側 `realtime_unified.py:292/420` でリング生成時に固定）、サンプルレート候補 `[48000, 44100, 16000, 22050, 32000, 96000]` (`stream_base.py:135`)、blocksize候補 `[..., 2048, 4096, 1024, 512, 8192]` (146)、duplex のレート候補 `[48000, 44100, 96000]` (95)、`MLDenoiser` の `target_sr=16000` (137)、DNS64固定 (91)。
- **エイリアシングの懸念**: `resample()` は `orig_sr==target_sr` のとき入力配列をコピーせず返す (33)。`StatefulResampler.resample_chunk` も同様 (117)。呼び出し側が破壊的変更するとバグの元（推測）。
- **`StatefulResampler` の出力長がint切り捨て** (128, 148, 155): `int(len(chunk)*up/down)` の累積でサンプルドリフトの可能性（推測: 下流のSOLA/ドリフト制御が補償）。
- **ローカライズ文字列がコード内に直書き**: `stream_base._get_error_message` (182-195) に日本語エラーメッセージがハードコード。i18n層の分離なし。

## 6. RingOutputBuffer と AudioOutput の関係、アンダーラン/オーバーラン設計

**関係は「間接・コールバック経由」で、両者は直接参照しない**:
- `AudioOutput` は薄いデバイスラッパで、RingOutputBuffer への参照を一切持たない（`output.py` 全体）。リングは**pipeline側が保持**する。
- 結線: `realtime_unified.py:520-525` で `AudioOutput(callback=self._on_audio_output)`。デバイスコールバックスレッドで `AudioOutput._audio_callback` (55) → `self._callback(frames)` (67) → `_on_audio_output` (763) が `_output_queue` をリングへ `add` (771) し `output_buffer.get(frames)` (799/820) を返す。`output.py` はリングの存在を知らない。

**オーバーラン処理（2層）**:
1. ソフト層（pipeline先行）: `_on_audio_output` の適応ドリフト制御 (787-818)。`available > drain_target+frames` のとき余分に読み `np.interp` で時間圧縮（≤1.5%、最大 `frames//64` サンプル 794）。サンプル破棄によるクリックを避ける。
2. ハード層（リング内）: `RingOutputBuffer.add` がオーバーフロー時に**最古サンプルを破棄** (56-61)、`_read_pos` を進め `_samples_dropped` を加算。リング容量は `chunk*4` (`realtime_unified.py:291/419`)。
   - ただし `add()` の戻り値（drop数）は呼び出し側 771 で**無視**されており、pipeline統計には反映されずリング内 `samples_dropped` プロパティは誰も読んでいない（軽微）。

**アンダーラン処理**:
- `RingOutputBuffer.get` (100-113): 残量不足時、利用可能分にフェードアウトを掛け残りをゼロ埋め、`_last_was_underrun=True`・`_underrun_count++`。次回回復時に `get` 冒頭 (92-97) でフェードイン。クリック防止設計として妥当。
- ただしフェードアウトは `avail > self.fade_samples` の時のみ (107)。`avail` が `fade_samples`(256) 未満の部分アンダーランではフェードが掛からずクリックの余地（軽微）。
- プリバッファゲート: `_output_started` が `_prebuffer_chunks` 到達まで再生を抑止しゼロを返す (776-780)。

**冗長な防御コード**: `AudioOutput._audio_callback` (68-73) 自身にも「callbackが `frames` 未満を返したらゼロ埋め」する分岐があるが、`_on_audio_output` は常に `get` で厳密に `frames` サンプルを返すため、通常パイプライン下ではこの分岐は事実上デッド。`duplex.py:189-194` にも同じ冗長分岐が重複。

## 総評

audio層はレイヤー違反の無いクリーンなleaf層で、リングを単一スレッド専有にしてQueueで生産者/消費者を分離した点は堅実。主要な改善候補:

1. `input.py`/`output.py` コールバックの例外保護欠如（`duplex.py` との非対称）
2. `stream_base.start()`/`duplex` コンストラクタでの失敗ストリーム未クローズ
3. `SpectralGateDenoiser.reset()` の実在しない属性参照（バグ）
4. 出力chルーティングと resample gcd 算出の重複
