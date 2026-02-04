# RCWX リアルタイム音声変換 解析レポート

## 概要

RCWXのリアルタイム音声変換処理において、スムーズな変換が行われない問題を調査し、検証フレームワークを構築しました。

## 発見した問題

### 1. ChunkBuffer の重大なバグ（修正済み）

**問題箇所**: `rcwx/audio/buffer.py` の `get_chunk()` メソッド

**症状**: 第一チャンクから第二チャンクへの移行時に、コンテキストの重複がない

**詳細**:
```
修正前:
  Chunk 0: main = [0, 7199]
  Chunk 1: context = [7200, 11999]  ← 前のmainと重複なし！ギャップ発生

修正後:
  Chunk 0: main = [0, 7199]
  Chunk 1: context = [2400, 7199]   ← 正しく重複
```

**原因**: 第一チャンク処理後、入力バッファを `chunk_samples` だけ進めていたが、
正しくは `chunk_samples - context_samples` だけ進める必要があった。

**影響**:
- 音声データの欠落（context_samples分のギャップ）
- クロスフェードが非重複領域で実行される
- チャンク境界でのアーティファクト

**修正内容**:
```python
# 修正前
self._input_buffer = self._input_buffer[self.chunk_samples:]

# 修正後
if self._is_first_chunk:
    advance = max(0, self.chunk_samples - self.context_samples)
    self._input_buffer = self._input_buffer[advance:]
else:
    self._input_buffer = self._input_buffer[self.chunk_samples:]
```

### 2. SOLA RMS Matching の潜在的問題

**問題箇所**: `rcwx/audio/crossfade.py` の `_match_rms()` 関数

**症状**: 極端なゲイン（最大3.0倍）が適用される可能性

**詳細**:
- 診断テストで `gain = 2.80` が観測された
- 2倍を超えるゲインは歪みの原因になりうる

**推奨対応**: `max_gain` を 2.0 に下げることを検討

### 3. 特徴キャッシュのサイズ問題

**問題箇所**: `rcwx/pipeline/inference.py`

**症状**: キャッシュサイズがチャンクサイズを超える場合、品質劣化

**詳細**:
- `cache_len = 10` フレーム（200ms）がハードコード
- `chunk_sec = 0.15s`（150ms）の場合、`cache_ratio = 1.33 > 1.0`

**推奨対応**: キャッシュサイズをチャンクサイズに応じて動的に調整

## 構築した検証フレームワーク

### 1. `tests/test_realtime_analysis.py`

GUIと冪等な設定でリアルタイム処理をシミュレートし、バッチ処理と比較。

```bash
uv run python tests/test_realtime_analysis.py --test-file sample_data/kakita.wav --visualize
```

**機能**:
- バッチ vs ストリーミング比較
- 相関、RMSE、不連続性メトリクス
- 各チャンクの推論時間記録
- 可視化（matplotlib）

### 2. `tests/test_step_by_step.py`

各処理ステップを個別に実行し、中間結果を保存。

```bash
uv run python tests/test_step_by_step.py --chunk-idx 0 1 2 3 4
```

**機能**:
- ステップ別出力保存（input → resample → infer → sola）
- チャンク境界アーティファクト検出

### 3. `tests/test_diagnostic.py`

各コンポーネントを個別にテスト。

```bash
uv run python tests/test_diagnostic.py --component all
```

**テスト対象**:
- StatefulResampler精度
- SOLA crossfade品質
- チャンク境界処理
- レイテンシ蓄積

### 4. `tests/test_chunk_continuity_detail.py`

ChunkBufferのコンテキスト重複を詳細検証。

## 改善提案

### 短期的な改善

1. ✅ ChunkBuffer の第一チャンク進行量バグを修正
2. RMS matching の max_gain を 2.0 に下げる
3. 特徴キャッシュサイズを動的に調整

### 中期的な改善

1. SOLA相関計算の改善（wokadaモードでは常に0）
2. 出力サンプル数の累積誤差を補正
3. 無音区間での処理最適化

### 長期的な改善

1. チャンキング戦略の統一的なテストスイート
2. リアルタイム処理のプロファイリング
3. 適応的パラメータ調整の検証

## テスト実行結果サマリー

| テスト | 結果 | 備考 |
|--------|------|------|
| Resampler | ✓ PASS | 相関 > 0.999 |
| SOLA | ✓ PASS | 不連続性 0 |
| Chunk Boundary | ✓ PASS | 修正後 |
| Latency | ⚠ WARNING | 計算式要確認 |

## ファイル変更

### 修正されたファイル
- `rcwx/audio/buffer.py` - ChunkBuffer.get_chunk()

### 追加されたファイル
- `tests/test_realtime_analysis.py`
- `tests/test_step_by_step.py`
- `tests/test_diagnostic.py`
- `tests/test_chunk_continuity_detail.py`
- `docs/ANALYSIS_REPORT.md`

## 次のステップ

1. FCPEをインストールして低レイテンシモードでテスト
2. GUIでの実際の動作確認
3. 長時間ストリーミングでの安定性テスト
