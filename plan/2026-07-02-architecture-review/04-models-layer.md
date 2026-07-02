# RCWX アーキテクチャレビュー: models/ + device/config/cli/downloader/diagnose

対象: `rcwx/models/` (hubert_loader, rmvpe, fcpe, swiftf0, synthesizer, infer_pack/) と `rcwx/device.py`, `config.py`, `cli.py`, `downloader.py`, `diagnose.py`。事実は file:line 付き。推測は「推測」と明記。

## 1. 各モデルローダーの責務・公開API・共通インターフェース

### 各ローダーの形態と公開API

| モデル | 型 | 主なコンストラクタ | 主な公開メソッド |
|---|---|---|---|
| HuBERTLoader (`hubert_loader.py:125`) | 通常class（nn.Moduleでない） | `__init__(model_path, device, dtype)` (`:130`) | `extract(audio, output_layer=12, output_dim=768)` (`:234`), `forward()` (`:273`) |
| RMVPE (`rmvpe.py:426`) | **nn.Module** | `__init__(model_path, device, dtype, hop_length=160)` (`:429`) | `infer(audio, threshold=0.03)` (`:535`), `forward()` (`:547`), `mel2hidden`, `decode` |
| FCPE (`fcpe.py:33`) | 通常class（torchfcpeの薄いラッパ） | `__init__(device, dtype, hop_length=160)` (`:40`) | `infer(audio, threshold=0.006, f0_min, f0_max)` (`:72`), `to(device)` (`:155`) |
| SwiftF0 (`swiftf0.py:203`) | 通常class（swift_f0 ONNXラッパ） | `__init__(hop_length=160, confidence_threshold=0.35)` (`:214`) | `infer(audio, threshold=0.35, f0_min, f0_max)` (`:234`) |
| SynthesizerLoader (`synthesizer.py:129`) | 通常class | `__init__(model_path, device, dtype, use_compile)` (`:136`) | `load()` (`:163`), `infer(...)` (`:269`) + モジュール関数 `detect_model_type`(`:22`), `get_model_class`(`:85`), `load_index`(`:343`) |

責務の要点:
- **HuBERTLoader**: RVC の `hubert_base.pt`(fairseq形式) と transformers形式の両対応が主責務。偽fairseqモジュール注入(`:18`)＋fairseq→transformersのキー写像(`:46-122`)という重い変換ロジックを内包。
- **RMVPE**: アーキ全体(BiGRU/ConvBlockRes/DeepUnet/E2E/MelSpectrogram)を1ファイルに実装。RVC WebUI の rmvpe.py 完全移植を明記(`:1-5`)。ローダーというより「モデル本体＋前処理＋デコード」を全部持つ。
- **FCPE**: torchfcpe への薄いラッパ。NaN/Inf対策とサイレンス早期リターン(`:104-111`)、ログスロットリング(`:62-63,140-150`)が独自付加。
- **SwiftF0**: ONNXラッパだが、**ハーモニクス・オクターブ誤検出をViterbi/DPで補正する独自コードが約170行**(`swiftf0.py:33-200`)と、ラッパの域を超えた重いロジックを infer 経路に内包。
- **SynthesizerLoader**: チェックポイント種別検出(v1/v2, has_f0, 話者数)＋config配列パース＋モデル生成＋weight_norm除去＋infer振り分け。

### F0抽出器3種の共通抽象 → 「無い。アドホック（ダックタイピング）」

- 3種とも `infer(audio, threshold=...) -> [B, T_frames]`（無声=0）という**慣習的シグネチャは揃っている**が、**共通の基底クラス/Protocol/ABCは存在しない**。RMVPEだけ nn.Module、FCPE/SwiftF0 は素のclass。
- コンストラクタが不揃い: RMVPEは`model_path`必須、FCPEは`device/dtype`、**SwiftF0は`device`も`dtype`も取らない**(`swiftf0.py:214`)。
- 既定thresholdが各所でバラバラ: RMVPEクラス既定`0.03`(`rmvpe.py:535`) に対しパイプラインは定数`RMVPE_VOICING_THRESHOLD=0.015`(`inference.py:45`)を渡す。FCPE=0.006、SwiftF0=0.35 の定数も別途 `inference.py:44,46` に重複定義。**真の初期値の所在が不明瞭**。
- 実際のディスパッチは `pipeline/inference.py` の `if f0_method=="fcpe" ... elif "swiftf0" ... elif rmvpe` という **if/elif 連鎖**で、少なくとも4箇所に重複(`inference.py:1248-1262`, `1379-1399`, `2037付近`, `2098付近`)。
- `models/__init__.py` は `RMVPE` と `SynthesizerLoader` しか公開せず(`:6-10`)、FCPE/SwiftF0/HuBERTLoader はサブモジュール直import。**公開面が非一貫**。

## 2. デバイス管理 (device.py)

- 切替: `get_device(preferred)` の優先度は **XPU > CUDA > CPU**(`device.py:30-34`)。`preferred` 指定時も各ブランチ(`:22-28`)で可用性を確認。
- **サイレント・フォールバック（設計上の問題）**: `preferred="cuda"` だが未搭載の場合、内側if/elifのいずれにも一致せずブロックを抜け、そのまま auto 検出(`:30-34`)に落ちる。**警告/ログ無しで別デバイスになる**。`preferred="xpu"`未搭載時にCUDAが返る可能性も同様。
- dtype: `get_dtype(device, preferred)` は **CPUなら強制的にfloat32**(`:56-57`)、それ以外は`float16/float32/bfloat16`マップ(`:48-54`)。未知文字列はfloat16へフォールバック(`:54`)。
- **deviceは全ローダーに「引数」渡し。グローバル状態は無い**。`RVCPipeline.__init__` が `get_device`/`get_dtype` を1回だけ解決(`inference.py:675-676`)し、各ローダーのコンストラクタへ `device`(str)＋`dtype`(torch.dtype)を渡す(`inference.py:746-751, 763-767, 778-782, 792-795`)。
- 例外: **SwiftF0 は device を受け取らない**(ONNX/CPU固定)。パイプラインも device なしで生成(`inference.py:806-808`)。SwiftF0.infer は入力テンソルから device を読み、結果を同 device に戻す(`swiftf0.py:257,358`)。
- FCPE: `spawn_bundled_infer_model(device=device)`(`fcpe.py:68`)。ただし `self.dtype` は torchfcpe 本体には適用されず、**出力castとzeros生成にのみ使用**(`:110,131,153`)。`to()`(`:155`)も`self.device`更新のみでコメント上「torchfcpeが内部処理」。
- dtype特記: RMVPE は数値安定性のため **mel/STFT を常に float32** で実施(`rmvpe.py:539-541`)し、モデル推論のみ設定dtype。HuBERTは audio を dtype へcast(`hubert_loader.py:255`)。パイプラインは `torch.autocast(device_type=self.device, ...)` で抽出をラップ(`inference.py:1242,1249,1258`)——device文字列("xpu"/"cuda"/"cpu")をそのまま autocast の device_type に流用。

## 3. config.py の構造

- **標準dataclass（pydanticではない）**。`AudioConfig`(`:15`), `DenoiseConfig`(`:37`), `PostprocessConfig`(`:48`), `InferenceConfig`(`:65`), `RCWXConfig`(`:131`) の5つ。ネストは `field(default_factory=...)`(`:127-128,141-142`)。
- **バリデーションは皆無**。範囲チェックも列挙チェックも無し。文字列フィールド(`f0_method`, `method`, `voice_gate_mode`, `device`, `dtype`, `*_channel_selection` 等)は自由入力で、typoは実行時までサイレント。
- 後方互換/マイグレーション処理は**限定的**:
  - 旧キー削除: `input_device`/`output_device`(int時代)を pop(`config.py:160-163`)。
  - InferenceConfig の未知キーのみ `dataclasses.fields` で**フィルタしてTypeError回避**(`:176-182`)。
  - **非対称な堅牢性（問題）**: `AudioConfig(**audio_data)`(`:185`) と `cls(**data)`(`:191`) は未知キーを除去しない。**トップレベル/audioに未知キーがあると TypeError で読込失敗**。フィルタは inference のみ。
- save は `asdict` + `json.dump`(`:194-202`)。default_path は `~/.config/rcwx/config.json`(`:204-207`)、models_dir 既定は `~/.cache/rcwx/models`(`:11-12`)。
- 既定値の分散: `noise_scale` が config=0.45(`:115`)、CLI=0.4(`cli.py:414`)、SynthesizerLoader.infer=0.66666(`synthesizer.py:277`)、models.infer=0.66666(`models.py:703`)と**4箇所で異なる**。

## 4. infer_pack/ の移植度と独自コードの境界

- 全ファイルが docstring で **"ported from RVC WebUI"** を明記(`models.py:1`, `attentions.py:1`, `commons.py`, `modules.py:1`, `transforms.py:1`, `__init__.py:1`)。構造は標準的なVITS/RVC(TextEncoder/PosteriorEncoder/Flow/GeneratorNSF/Generator)。
- 手の入り具合:
  - **全面的な型注釈追加**（`from __future__ import annotations` ＋ 型付きシグネチャ）——主に体裁。
  - **SineGen の独自二重実装**: 新`_f02sine`(subframe位相累積, `models.py:194-241`) と旧`_f02sine_legacy`(補間, `:243-280`)を**環境変数 `RCWX_SINEGEN_MODE`**(既定"legacy", `:184,291-294`)で切替。
  - **`fixed_harmonics` フラグ**(`models.py:187,233-238,253-262`): ストリーミング連続性のためハーモニクス初期位相を0固定する独自機能。外部から `RVCPipeline._set_fixed_harmonics`(`inference.py:948-961`)で `dec.m_source.l_sin_gen.fixed_harmonics` を書換え。
  - infer に **streaming用 skip_head/return_length/return_length2** と flow文脈24フレーム(`models.py:718-743, 871-891`)、GeneratorNSF の n_res 補間(`:425-433`)、noise_scale パラメータ化。
  - 学習経路は削除され推論(infer)のみ。ただし `commons.py` には学習専用ヘルパ(`slice_segments`, `rand_slice_segments`, `generate_path`, `subsequent_mask`, `kl_divergence`)が**デッドコードとして残存**(`commons.py:24-124`)。
- **境界の明確さ**: infer_pack は torch/numpy と自パッケージ内(`attentions/commons/modules/transforms`)しか import せず自己完結(`models.py:16`)。RCWX側との境界は `synthesizer.py`(SynthesizerLoaderがこれらをラップ)で明快。**唯一の侵入点**は、移植ファイル内部に埋め込まれた `RCWX_SINEGEN_MODE` 環境変数と `fixed_harmonics` 属性で、これらだけが「移植コードにRCWX固有ロジックが染み出した箇所」。

## 5. ロード/アンロードのライフサイクル・キャッシュ・メモリ

- **遅延ロード**: 各コンポーネントは None 初期化(`inference.py:692-696`)、`load()`(`:729-826`)で生成。`_loaded` ガードで二重ロード防止(`:731-732`)。
- **ロード順と過剰ロード（問題）**: synthesizer を最初にロードして種別/sr/has_f0検出(`:746-756`)→ HuBERT(`:763`)→ has_f0 なら **RMVPE・FCPE・SwiftF0 を3種すべて eager ロード**(`:774-814`)。設定 `f0_method` が1種でも、他2種を無条件生成。**起動時間とメモリの無駄**（RMVPE本体＋torchfcpe＋ONNXセッション）。
- **アンロード**: `unload()`(`:912-928`)は全参照をNone、`clear_cache()`、cuda/xpuの`empty_cache()`。個別モデル単位のunloadは無く、パイプライン単位のみ。
- **クロスインスタンスのモデルキャッシュ/LRUは無い**。各 RVCPipeline が自前で保持。
- torch.compile は hubert.model / rmvpe.model / synthesizer.model に任意適用(`inference.py:769-771,783-785`, `synthesizer.py:223-225`)。FCPE/SwiftF0 は対象外。
- ストリーミング用キャッシュが多数（feature/f0/f0_voiced/audio/output/stream_history、`inference.py:707-727`）、`clear_cache()`(`:930-946`)で一括クリア。
- FAISS index は `reconstruct_n` で**全ベクトルをメモリ再構成**(`inference.py:838`)。

## 6. 気になる設計上の問題

1. **デッドかつ誤解を招く型注釈**: `inference.py:692` の `self.hubert: Optional[HuBERTFeatureExtractor]` が参照する `HuBERTFeatureExtractor` は**存在しないクラス**。旧 `rcwx/models/hubert.py` がコミット `83b9522` で削除され `HuBERTLoader` に置換済み（`__pycache__/hubert.cpython-312.pyc` だけ残存）。`from __future__ import annotations` で文字列注釈のため実行時例外にならず生き残っている。実体は `HuBERTLoader`。

2. **F0抽出器の共通抽象欠如による重複**: 前述の if/elif ディスパッチが4箇所に重複(`inference.py:1248-1262,1379-1399,2037,2098`)。Protocol/ABC を切れば集約可能。SwiftF0 のコンストラクタ非対称(device/dtypeなし)も統一を妨げる。

3. **threshold/noise_scale等の既定値分散**: F0 threshold がクラス既定とパイプライン定数で不一致（RMVPE 0.03 vs 0.015）。noise_scale が4箇所で異なる（§3,§4）。真の設定源が不明瞭。

4. **F0モデル3種の過剰eagerロード**（§5）: 使わない2種も常時ロード。

5. **HuBERT transformers経路の final_proj がランダム初期化**: チェックポイント無しの transformers ロード時、v1用 final_proj(768→256) を **Xavier乱数で初期化**(`hubert_loader.py:161-163`)。v1モデルで output_dim=256 を使うと乱数射影で無意味な特徴になり得るが警告無し。（推測: v2/768専用運用なら実害は限定的だが、フォールバック時のサイレント劣化リスク。）

6. **config マイグレーションの非対称性**（§3）: inference以外のセクションは未知キーで TypeError。

7. **推論ロジックの大規模重複**: `pipeline/inference.py` が107KB/2300行超で、`infer`(バッチ)と `infer_streaming`(複数F0ディスパッチ)を持ち、さらに `realtime_unified.py`(57KB)が並存。F0ディスパッチ・出力cast等の重複面が広い。

8. **SwiftF0 の重いViterbi/DPを infer 経路に内包**(`swiftf0.py:80-200`): Pythonネストループのオクターブ補正が推論ごとに走る。「モデルローダー」の責務としては重く、リアルタイム性への影響が懸念（推測: セグメント長次第で無視できない）。

9. **device.py のサイレント・フォールバック**（§2）: 要求デバイス未搭載時に警告無しで別デバイス。

10. **デッドコード**: `commons.py` の学習専用ヘルパ群（`slice_segments/slice_segments2/rand_slice_segments/generate_path/subsequent_mask/kl_divergence`）は推論専用移植では未使用。`SineGen._f02sine`(subframe)も既定"legacy"のため通常経路では不使用。

11. **ログ過多**: `detect_model_type` が毎ロードで全チェックポイントキーとconfigを INFO 出力(`synthesizer.py:38-39,56,60`)。

12. **循環依存**: 検出されず。`models → infer_pack` は一方向、infer_pack内も `models→attentions/commons/modules→commons/transforms` の非循環DAG。この点は健全。

## 補足（downloader.py / diagnose.py / cli.py）

- `downloader.py`: HuBERT/RMVPE のみDL対象(`:139-142`)。FCPE/SwiftF0 はpip依存でDL対象外＝ダウンロード対象とcheck_modelsが実際の必要モデルと非一致。`local_dir_use_symlinks=False`(`:57,89`)は新しめ huggingface_hub で deprecated（推測: バージョン次第で警告）。
- `diagnose.py`: 純出力ツール。device.py とは独立実装で、XPU/CUDA判定ロジックが `device.py` と**重複**(`diagnose.py:174-191` vs `device.py:101-117`)。
- `cli.py`: `cmd_run` の既定 `use_compile=not args.no_compile` = True 既定だが、config既定は `use_compile=False`(`config.py:77`, Windows XPU安定性のため)。**CLIとconfigで既定が逆**（`cli.py:192`）。`--index-rate` 既定0.0(`:396`)でCLIからはindex実質無効が既定。

## 要点

models/ は「F0抽出3種に共通抽象が無くダックタイピング＋if/elif分岐」「infer_packは概ねクリーンな移植で独自侵入はSineGenの2箇所のみ」「device引数渡しは一貫（SwiftF0のみ例外）」「configは素dataclass・無バリデーション・非対称マイグレーション」。最も実害寄りの具体的欠陥は `inference.py:692` のデッド型参照と、F0モデル3種の無条件eagerロード。
