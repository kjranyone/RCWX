# デコーダ出力クロスオーバーによる長音ブツ切り修正

## Context

持続母音 "a------" がチャンク境界でブツ切りになる。
原因: `skip_head`/`return_length` によりデコーダ（SineGen + Conv 層）が
各チャンクの出力領域のみを独立に合成し、チャンク間にクロスオーバーがない。
SineGen 位相がチャンクごとにリセットされ、Conv 層もコールドスタートする。

## 方針

`_sola_extra_model` にデコーダオーバーラップ分を加算することで、
`infer_streaming` の `skip_head` を自然に後退させる。
**モデル側 (`models.py`) の変更は不要**。既存の SOLA クロスフェードが
デコーダのオーバーラップ出力を消費し、チャンク間連続性を実現する。

### 数値例 (chunk=0.3s, overlap=0.2s, model_sr=40kHz)

| | 変更前 | 変更後 |
|---|---|---|
| sola_extra_model | 4000 (100ms) | 6000 (150ms) |
| trim_left | 6000 | 4000 |
| skip_head_feat | 15 | 10 |
| return_length_feat | 20 | 25 |
| decoder 処理量 | 20 frames | 25 frames (+25%) |

デコーダが 5 フレーム (50ms) 早くから合成:
- SineGen 位相が overlap 領域から蓄積 → 出力領域で連続
- Conv 受容野がウォーム → 境界の音色不連続が解消
- SOLA が model_sr→48kHz 変換後にクロスフェード

## 実装（1ファイル）

### `rcwx/pipeline/realtime_unified.py`

#### 1. `RealtimeConfig` にフィールド追加

```python
# Decoder overlap for cross-chunk continuity (feature frames, 1 frame = 10ms)
decoder_overlap_frames: int = 5
```

#### 2. `__init__` の `_sola_extra_model` 計算を修正

```python
sola_extra_out = crossfade_samples_out + 2 * search_samples_out
zc_model = self.pipeline.sample_rate // 100
sola_extra_raw = int(
    sola_extra_out * self.pipeline.sample_rate / self._runtime_output_sample_rate
)
# Decoder overlap: extra context frames so decoder Conv/SineGen
# have warm start, preventing cold-start discontinuity at chunk edges.
decoder_overlap_model = self.config.decoder_overlap_frames * zc_model
self._sola_extra_model = (
    (sola_extra_raw + decoder_overlap_model + zc_model - 1)
    // zc_model * zc_model
)
```

#### 3. `_recalculate_sizes()` にも同じ変更

`_recalculate_sizes` 内の `_sola_extra_model` 計算箇所（L380-385）に同様の
`decoder_overlap_model` 加算を追加。

## 影響範囲

- `inference.py`: 変更不要。`sola_extra_samples` が大きくなるだけで
  `trim_left` / `skip_head_feat` / `expected_output` は既存計算で自動調整
- `models/infer_pack/models.py`: 変更不要。`skip_head` / `return_length` は
  パイプラインから渡される値に従うだけ
- SOLA: 変更不要。入力が長くなるだけで `target_len=hop_samples_out` は不変
- GUI: 変更不要。`decoder_overlap_frames` はユーザー設定不要の内部パラメータ

## 検証

1. `uv run rcwx` で GUI 起動、持続母音 "a------" を入力
2. チャンク境界での音色不連続・ブツ切りが解消されていること
3. 推論時間の増加が chunk_sec の 80% 以内に収まること（ステータスバー確認）
4. 既存テスト: `uv run python tests/integration/test_realtime_chunk_boundary_gain.py`
