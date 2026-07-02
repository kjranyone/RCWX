# RCWX GUI アーキテクチャレビュー (`rcwx/gui/`)

調査対象ブランチ: `feature/improve202602`。行番号はすべて調査時点の実ファイル基準。

## 0. 未コミット変更の内容（`git diff rcwx/gui/app.py`）

変更は `_on_audio_settings_changed()` 内の1箇所のみ（`app.py:971-975` に5行追加）:

```python
# Push input gain to running voice changer
if self.realtime_controller.voice_changer:
    self.realtime_controller.voice_changer.config.input_gain_db = (
        self.audio_settings.input_gain_db
    )
```

- **何をしているか**: オーディオ設定（入力ゲインスライダー等）変更時に、稼働中の voice changer へ入力ゲインをライブ反映する。処理スレッドは `realtime_unified.py:868` (`if self.config.input_gain_db != 0.0`) で毎チャンク `config.input_gain_db` を読むため、GUIスレッドがこの1フィールドを書き換えるだけで即座に反映される。変更前はゲイン変更が config 保存のみで、稼働中VCには次回起動まで反映されなかった。
- **設計上の注目点**: 他のパラメータはすべて `voice_changer.set_pitch_shift()` 等の **setter 経由**（`realtime_unified.py:634-742`）だが、入力ゲインだけ `set_input_gain` 相当のメソッドが存在せず、`config` フィールドを**直接代入**している。setter パターンとの不整合。（推測）GUIメインスレッドから、処理スレッドが読む float を直接書くクロススレッド書き込みだが、単一float・GILのため実害は軽微。

## 1. 各モジュールの責務と規模／app.py の God Object 化

| ファイル | 行数 | 主責務 |
|---|---|---|
| `app.py` | **1145** | メインウィンドウ + 全体オーケストレーション（下記詳述） |
| `realtime_controller.py` | 337 | リアルタイムVCのライフサイクル制御・警告ダイアログ |
| `audio_test.py` | 327 | 3秒録音→変換→再生テスト |
| `file_converter.py` | 257 | WAVファイル変換・再生・保存 |
| `model_loader.py` | 149 | モデル非同期ロード |
| **widgets/** | | |
| `audio_settings.py` | **1037** | オーディオデバイス選択・レベルメーター・モニター/ループバック |
| `pitch_control.py` | 545 | ピッチ/F0/各種音質パラメータ |
| `model_selector.py` | 259 | モデル選択ドロップダウン |
| `latency_settings.py` | 244 | チャンクサイズ + 自動導出パラメータ |
| `latency_monitor.py` | 218 | ステータスバー（レイテンシ/GPU/Index表示） |
| `postprocess_settings.py` | 163 | 高域ブースト/リミッター |

### app.py の責務列挙（God Object 化している）

`RCWXApp` は `ctk.CTk` を継承し、以下 **約13の責務**を保持:

1. **ウィンドウ/アプリライフサイクル**（`__init__` 41-90, `_on_close` 1107-1126, `run/mainloop` 1128-1130）
2. **設定のロード/保存オーケストレーション**（`config.load` 57, `_save_config` 1005-1053 で約30フィールドを集約書き戻し, close時保存）
3. **3タブ + ステータスバーのUI構築**（`_setup_ui` 92, `_setup_main_tab` **144-485＝約340行**, `_setup_audio_tab` 487, `_setup_settings_tab` 571, `_setup_audio_test_section` 696）
4. **インラインウィジェットの直接保有**: Index検索枠(201-254)、ノイズキャンセル枠(257-305)、Voice Gate枠(308-377)、デバイス表示枠(384-417)、テスト枠(420-472)、WAVループ入力(445-472)、開始ボタン(475-485)——これらは widget クラスに抽出されず app.py に生でベタ書き
5. **pipeline 参照の共有可変状態としての保有**（`self.pipeline` 60、複数モジュールが `self.app.pipeline` を読み書き）
6. **デバイス情報算出**（`get_device`/`get_device_name` 67-68）
7. **イベントハンドラ・ハブ**: `_on_*` ハンドラ約25個（835-1003）が「config保存 + VC setter呼び出し」を仲介
8. **マネージャへの薄いラッパ委譲**: `_browse_test_file`/`_convert_test_file`/`_play_converted_audio`/`_stop_test_playback`/`_save_converted_audio`（784-802）等、1行パススルーが約10個
9. **config→widget の復元/同期ロジック**（audio_tab 復元 515-569、main_tab 復元 186-198）
10. **AudioSettingsFrame の private 内部への直接操作**（531-557 で `_input_hostapi_filter`/`_filter_devices_by_hostapi`/`_all_input_devices`/`input_dropdown.configure` を外から触る＝カプセル化破壊）
11. **Voice Gate 閾値スライダーの表示/非表示ロジック**（`_on_voice_gate_mode_changed` 925-939）
12. **WAV入力トグルのUI制御**（`_on_wav_input_toggled` 1089-1094）
13. **タブ切替時の auto-refresh 制御**（`_on_tab_changed` 1055-1066）

→ 1145行・約13責務で **God Object 化している**と判断。特に「メインタブの生ウィジェット構築(4)」と「イベントハンドラ・ハブ(7)」が肥大の主因。

## 2. GUI ⇔ RealtimeVoiceChangerUnified の結合

### パイプライン(`RVCPipeline`)を直接触るクラス
- **`ModelLoader`**（`model_loader.py:56-66`）: `RVCPipeline(...)` 生成・`.load()`/`.unload()`、`.has_f0`/`.synthesizer`/`.device`/`.faiss_index`/`.sample_rate` を読む
- **`app.py`**: `self.pipeline.faiss_index` を直接参照（`_on_index_changed` 901, `_on_index_ratio_changed` 922）
- **`AudioTestManager`**（`audio_test.py:193, 212`）: `self.app.pipeline.infer()`, `.sample_rate`
- **`FileConverter`**（`file_converter.py:103, 124`）: `self.app.pipeline.infer()`, `.sample_rate`
- **`RealtimeController`**（`realtime_controller.py:137-138`）: `self.app.pipeline` を VC コンストラクタへ渡す

→ pipeline は `app.pipeline` という共有可変状態で、**5クラスが直接アクセス**。カプセル化されていない。

### RealtimeVoiceChangerUnified を触るクラス
- **`RealtimeController`**: 生成(137)、`start()`(150)/`stop()`(204)、コールバック設定(142-143)、`RealtimeConfig` 組み立て(88-134)
- **`app.py`（重要）**: `self.realtime_controller.voice_changer.set_*(...)` を **約15箇所で2階層貫通アクセス**（`_on_pitch_changed` 838-839, `_on_f0_method_changed` 849-850, … `_on_energy_threshold_changed` 946-947, さらに diff の 972-975）

### realtime_controller.py の役割
`RealtimeController` は「VCライフサイクルの橋渡し」に特化:
- `toggle/start/stop`（46-212）
- 全 widget 状態を集約して `RealtimeConfig` を構築（`start()` 内 88-134、約50行のパラメータ集約）
- ウォームアップ進捗表示（`_on_warmup_progress` 158-161）
- stats→ステータスバー、error→UI のスレッド跨ぎ marshalling（214-242）
- バッファ underrun/overrun 警告のしきい値判定・ダイアログ（219-276）
- フィードバック（同一オーディオIF）検出（`_check_same_audio_interface` 278-306）
- 警告/エラーダイアログ生成（308-337）

**結合上の問題**: controller は**起動時の設定集約と停止のみを仲介**し、**稼働中のランタイム・パラメータ更新は仲介していない**。app.py が `realtime_controller.voice_changer.set_xxx()` と controller を貫通して VC を直接叩く（デメテルの法則違反）。controller にパススルー setter がないため、app と VC が2階層で密結合。

## 3. スレッドモデル

Tkinter は単一スレッド。本コードのモデル:

| 処理 | 実行スレッド | メインへの復帰方法 |
|---|---|---|
| 全UI/mainloop | メイン | — |
| **モデルロード** | バックグラウンド daemon（`model_loader.py:54-76`） | `self.app.after(0, ...)`（69, 73）✔ 正しい |
| **リアルタイムVC start/stop** | **メインスレッドで同期実行**（`realtime_controller.py:80-82` コメント: sounddeviceはWindowsでメインスレッド必須） | VC が内部で処理スレッドを spawn（`realtime_unified.py:582-587`）。stats/error は VC→`app.after(0,...)`（`realtime_controller.py:217, 242, 229, 237`）✔ |
| **オーディオテスト録音/再生** | **メインスレッドで同期ブロック**（`sd.rec`+`sd.wait` 93-100, `sd.play`+`sd.wait` 281-282） | — |
| **オーディオテスト変換** | バックグラウンド daemon（`audio_test.py:170`） | `_conversion_done` Event + `after(100,...)` 非ブロッキングポーリング（174, 218-223）✔ 良い実装 |
| **ファイル変換** | バックグラウンド daemon（`file_converter.py:135`） | `self.app.after(0, ...)`（128, 133）✔ |
| **レベルメーター monitor** | PortAudioコールバックスレッド（`audio_settings.py:621`） | `self.after(0, ...)`（646）✔ |
| **デバイス auto-refresh** | メイン（`after` ループ、`audio_settings.py:419-435`）✔ | — |

### GUIスレッド違反の可能性

**(A) バックグラウンド変換スレッドからの ctk 変数 `.get()` 読み取り（要注意・推測）**
- `audio_test.py:_convert_thread`（187-216）はワーカースレッドから `self.app.voice_gate_mode_var.get()`(198), `self.app.energy_threshold_slider.get()`(199), `self.app.use_denoise_var.get()`(207) を呼ぶ
- `file_converter.py:convert_thread`（85-136）も同様に `self.app.voice_gate_mode_var.get()`(109), `energy_threshold_slider.get()`(110), `use_denoise_var.get()`(118) をワーカーから読む

Tkinter/Tcl インタプリタは非スレッドセーフで、`Variable.get()` は Tcl インタプリタにアクセスする。**別スレッドからの `.get()` 呼び出しは Tk のスレッド安全性違反の可能性**（実害が出るかは Tcl のビルド次第で確率的）。純Python属性（`self.app.pitch_control.pitch` 等）の読みは安全。**この2箇所は明確なリスク**。

**(B) メインスレッドのブロッキング（設計スメル、違反ではない）**
- テスト録音（最大3秒 `sd.wait`）と VC 起動時ウォームアップがメインスレッドをブロックし mainloop を停止させる。`update_idletasks()`（`audio_test.py:66,159,271`, `realtime_controller.py:146,161`）で描画を部分的に回避しているが、イベントループは固まる。コメント上「sounddevice がメインスレッド必須」という制約に由来する意図的実装だが、UIフリーズを招く。

## 4. 設定 (config.py) の読み書きフロー

`config.py` は dataclass 4種（`AudioConfig`/`DenoiseConfig`/`PostprocessConfig`/`InferenceConfig`）+ `RCWXConfig`。保存先 `~/.config/rcwx/config.json`（`config.py:204-207`）。

- **ロード**: `RCWXConfig.load()`（`config.py:144-192`）を `app.__init__`（`app.py:57`）で1回。旧キー(`input_device`/`output_device` int)のマイグレーション(160-163)、未知キーのフィルタ(176-182)、ネスト `denoise`/`postprocess` の再構築(165-173)あり。
- **config→widget 配布**: `__init__` と `_setup_*` の復元ブロックで push。`pitch_control.set_*`（186-198）、`latency_settings.set_values`（963）、`audio_settings` 復元（515-569）。
- **保存**: `_save_config()`（1005-1053）が**各 widget から pull** → `self.config.*` に代入 → `config.save()` で全体JSON書き出し。`self._initializing` フラグで構築中の保存を抑止（1008）、`try/except` で握り潰し(1052)。close時にも保存(1123)。
- **同期方式**: 基本は**「保存時 pull（app が widget を読む）＋ロード時 push」**。ただし **PostprocessSettingsFrame は例外**で、`self.config.inference.postprocess` の参照を構築時に受け取り（`app.py:503`）in-place で書き換える（`postprocess_settings.py:133,139,145,151,157`）＝**共有参照ミューテーション**。他 widget と同期パターンが異なる。

### 設計上の問題
- **書き込みアンプリフィケーション**: ほぼ全ての `_on_*_changed` が `_save_config()` を呼び、その都度**config全体をJSONにフルシリアライズしてディスク書き込み**。スライダーをドラッグすると1ステップ毎に `_on_slider_change`→`_save_config`→全JSON dump（例: `_on_index_ratio_changed` 914-917, `_on_energy_threshold_changed` 941-944, pitch/moe/noise 各スライダー）。ドラッグ中に多数の disk write が発生する堅牢性・性能スメル。
- **オブザーバ/バインディング不在**: GUI状態とconfigは自動同期されず、`_save_config` が全フィールドを手作業で列挙(1012-1050)。フィールド追加時に3箇所（dataclass, restore, save）を手更新する必要があり、漏れやすい。

## 5. widgets/ と親アプリの結合方法

**2つの相反する結合スタイルが混在**:

### (a) widgets/ = コールバック注入（疎結合・良好）
- `ModelSelector`（`on_model_selected`）、`PitchControl`（**10個**のコールバック `on_pitch_changed`…`on_f0_slew_max_step_changed`、`pitch_control.py:17-43`）、`AudioSettingsFrame`（`on_settings_changed` 単一）、`LatencySettingsFrame`（`on_settings_changed`）、`PostprocessSettingsFrame`（`on_settings_changed` + config参照）
- これらは **app への逆参照を持たず**、コンストラクタ注入されたコールバックを呼ぶだけ。状態は `@property`（`pitch_control.py:497-545`）や `get_*()`（`audio_settings.get_channel_selection`, `latency_settings.get_settings`）で公開。**クリーンな依存方向**。
- `LatencyMonitor`（ステータスバー）はコールバックすら持たない純ビュー。app が `set_device`/`update_stats`/`set_running`/`set_loading`/`set_index_status` を push（`latency_monitor.py:121-214`）。

### (b) マネージャ(audio_test/file_converter/model_loader/realtime_controller) = app 逆参照（密結合）
- 全て `__init__(self, app)` で app 参照を保持し、**app のウィジェット/状態へ直接読み書き**。例: `RealtimeController.start()` は `app.pipeline`/`app._loading`/`app.use_wav_input_var`/`app.audio_settings.*`/`app.start_btn`/`app.status_bar`/`app.latency_settings`/`app.pitch_control.*`/`app.config.*`/`app.model_selector` 等 **20以上の app 属性**を読む。
- これらは「app から抽出したメソッド群だが、依然 app のウィジェットを直接操作する」中途半端な抽出。widgets/ とは逆に**双方向の密結合**。

→ **widgets/ は callback で疎結合、manager群は god-object 逆参照で密結合**、という一貫性の欠如が最大の構造的観察点。

## 6. 気になる設計上の問題

### 重複コード
- **`pipeline.infer()` パラメータ集約が3箇所で重複**: `audio_test.py:193-210`(約13 kwargs)、`file_converter.py:103-121`(ほぼ同一)、`realtime_controller.py:88-134`(同じ widget 状態を `RealtimeConfig` へ)。同じ「widget→推論パラメータ」変換が3重。
- **ループバック `output_callback` の重複**: `audio_settings.py:544-566`（`_on_loopback_toggle` 内）と `661-684`（`_start_monitor` 内）にほぼ同一のコールバック定義。
- **警告ダイアログ生成の重複**: `realtime_controller._show_warning`（308-331）と `model_loader._on_model_load_error`（120-134）が同型の CTkToplevel ダイアログを個別実装。

### 責務混在・レイヤー違反
- **View→Domain 依存**: `latency_monitor.py:9` が `rcwx.pipeline.realtime_unified.RealtimeStats` を import（ビュー widget がパイプライン型に依存）。`postprocess_settings.py:10` が `rcwx.config.PostprocessConfig` に依存し in-place 変更（widget が config dataclass を書き換え）。
- **カプセル化破壊**: `app.py:531-557` が `AudioSettingsFrame` の private（`_input_hostapi_filter`, `_all_input_devices`, `_filter_devices_by_hostapi`, `input_dropdown`）を外部から操作。
- **デメテル違反**: app.py が `self.realtime_controller.voice_changer.set_*()` を2階層貫通（15箇所、§2参照）。
- **メインタブUIの未抽出**: Index/Denoise/VoiceGate/Device/Test の各枠が widget クラス化されず app.py にベタ書き（`_setup_main_tab` 340行）。他機能は widget 化されているのに一貫性がない。

### デッドコード / 未使用
- `PitchControl.set_pre_hubert_pitch`（`pitch_control.py:437-439`, backward-compat bool setter）— 呼び出し元なし（grep で定義のみ）。
- `LatencyMonitor.update_index_rate`（`latency_monitor.py:216-218`）— 呼び出し元なし。
- `RealtimeVoiceChangerUnified.set_sola`（`realtime_unified.py:717`）と `set_f0_lowpass_cutoff_hz`（732）— GUIから呼ばれない。特に `_on_latency_settings_changed`（`app.py:989-994`）は `set_prebuffer_chunks/buffer_margin/overlap/crossfade/chunk_sec` は呼ぶが **`set_sola` を呼ばない**（`use_sola` 設定は起動時のみ反映）。
- `AudioConfig.crossfade_sec`（`config.py:26`）— `InferenceConfig.crossfade_sec`(97) と**重複**。`_save_config` は inference 側にのみ保存(1042)するため audio 側は死にフィールド。`AudioConfig.sample_rate`/`output_sample_rate`(23-24) も GUI 実行時は `audio_settings` の自動検出値を使うため未使用。
- `widgets/__init__.py` の `__all__`（9-15）に `PostprocessSettingsFrame` が欠落。かつ app.py は全 widget を**サブモジュール直 import**（`app.py:20-25`）しており、パッケージ `__init__` の export を一切使わない。`__init__.py` の export は実質未使用・不整合。

### その他
- **`_check_same_audio_interface`（`realtime_controller.py:278-306`）のヒューリスティック**: デバイス名に "realtek"/"hd audio" 等が含まれるかで同一IF判定。名前ベースで脆く、誤検知・見逃しの温床。（推測）
- **例外の握り潰し**: `_save_config`（1052）、`file_converter.stop_playback`（213）等で `except Exception: pass`/`log only`。デバッグ困難化。
- **監視コールバック内の大量 `logger.info`**（`audio_settings.py:546-565, 652-683`、`#3` まで出力するデバッグ痕跡）— 本番コードにデバッグログが残存。

## まとめ（要点）

- `app.py`（1145行/約13責務）は **God Object 化**。特にメインタブUIの生ベタ書きとイベントハンドラ・ハブが主因。
- **結合の一貫性欠如**が根本問題: `widgets/` はコールバック注入で疎結合な一方、`audio_test/file_converter/model_loader/realtime_controller` は `app` 逆参照で密結合。
- `realtime_controller` は**起動時のパラメータ集約と停止のみ**を仲介し、稼働中のランタイム更新は app が VC を貫通アクセス。
- スレッドモデルは概ね健全（`after()` marshalling）だが、**バックグラウンド変換スレッドが ctk `Variable.get()` を呼ぶ2箇所**（`audio_test.py:198-207`, `file_converter.py:109-118`）が Tk スレッド安全性違反の可能性。
- config は「保存時 pull + ロード時 push」だが**全変更で全JSONフルダンプ**（スライダードラッグ中の頻繁 disk write）。postprocess のみ共有参照ミューテーションで同期方式が不統一。
- 進行中の diff は入力ゲインのライブ反映追加。setter を介さず config フィールドを直接代入する点が既存パターンと不整合。
