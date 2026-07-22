# Moe Boost Feminization: Design Note

F0-only の萌え声スタイリング。実装は `rcwx/pipeline/inference.py` の `apply_moe_f0_style`。
ユーザー向け要約は [README.md の Moe Boost](../README.md#moe-boost)。

## Problem Statement

`moe_boost` は低音域の男声 F0 を明るいゲーム声風コンターへ寄せつつ、明瞭さとアーティファクト回避を両立する。

制約:
- 再学習なし
- リアルタイム互換
- 既存の RVC 合成・後段 F0 フィルタと共存

## Signal Model

有声 F0 を `f(t)` Hz、`l(t)=log2(f(t))` とする。

分解:
- phrase trend `m(t)`（低周波ベースライン）
- deviation `d(t)`（semitone）

`d(t) = 12 * (l(t) - m(t))`

これにより次を独立に制御できる:
- register shift（基底音域の引き上げ）
- accent shaping（上昇/下降の非対称ゲイン）
- floor control（胸声域の低すぎるディップ抑制）

## Transform（strength `s ∈ [0, 1]`）

実装係数（AS-IS）:

1. **短ギャップ補間** — 無声ギャップを最大 `round(2 + 4s)` frames（100fps）まで線形補間
2. **レジスタ目標** — `f_target = 165 + 55s` Hz、上方向のみ  
   `shift_st = clamp(12*log2(f_target / median), 0, 1.5 + 4.5s)`
3. **コンター整形**
   - trend 窓: `odd(max(7, 7 + 14s))`
   - 上昇ゲイン `1 + 0.45s` / 下降 `1 - 0.25s`
   - phrase bias `0.10 + 0.45s` st
   - soft sat: `d / (1 + (0.08 + 0.16s)*|d|)`
4. **再構成** — `l_out = m + (d + shift + bias) / 12`、`f_out = 2^l_out`
5. **フロア / シーリング**
   - relative: `f_target * (0.55 + 0.08s)`
   - absolute: `85 + 45s` Hz
   - applied floor: `max(rel, abs)` を `[85, 220]` にクリップ
   - 最終クリップ `[0, 940]` Hz

加えて実装では **voiced-ratio confidence** で styled と raw をブレンドする（strength とは独立軸）。有声が薄い窓でレジスタフロアへスナップしチャンク境界で段差が出るのを防ぐ。

後段の RCWX フィルタ（lowpass / octave-flip / slew / hole-fill）がスタイリング後コンターを安定化する。

## Limitations

- F0 のみ。フォルマント（声道共鳴）シフトは行わない
- 効果は入力音域依存（男声 100–160Hz 帯で効きやすい）
- 強度 1.0 は不自然になり得る。運用推奨はおおよそ 0.40–0.80

## Validation

- `tests/models/test_moe_f0_style.py` — identity / register lift / gap fill / high-voice bound
- `tests/integration/test_moe_f0_processing.py` — パイプライン通し
- `tests/integration/test_moe_clarity_scoring.py` — clarity 関連
- `tests/integration/test_realtime_drift_control.py` — ランタイムドリフト制御
