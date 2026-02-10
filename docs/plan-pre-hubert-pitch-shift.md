# Pre-HuBERT Pitch Shift: アプローチ検証と設計

## Context

### 問題

現在のRCWXでは、ピッチシフトはF0カーブに対してのみ適用される（`inference.py:852,1455`）。
HuBERT/ContentVecは常に元の話者の音声をそのまま処理するため、特徴量は元の話者のスペクトル特性（フォルマント位置、倍音構造）を反映する。

例: 男性→女性変換（+12半音）の場合
- HuBERTは男性の声（基本周波数 80-180Hz、低い倍音構造）を見る
- シンセサイザはこの「男性的」な特徴量に女性のピッチ（200-300Hz）を加算合成
- ContentVecの学習データ分布と入力の乖離 → 特徴空間で最適でない領域を使用

### 目標

HuBERTに渡す音声をピッチシフトして、ターゲット音声モデルの学習データ分布に近づける。

### 現在のパイプライン（関連部分）

```text
audio_16k → [history buffer 8960samples] → highpass → reflect padding
         → [audio_t tensor] → HuBERT (50fps) ─────────────┐
         → [audio_t tensor] → F0 (100fps) → pitch_shift → │ → synthesizer
                                                           │
         同じaudio_tを使用（完全なフレーム整合）           │
         ─────────────────────────────────────────────────┘
```

ポイント:
- HuBERTとF0は同一の`audio_t`テンソルを使用（`inference.py:1374`）
- 固定サイズパディングでXPU再コンパイル回避（`inference.py:1339-1346`）
- torchaudioは既に依存関係に含まれる（`pyproject.toml:9`）

---

## アプローチ比較

### A. `torchaudio.functional.pitch_shift`（STFT位相ボコーダ）

原理: STFT → 時間伸縮（位相ボコーダ） → ISTFT → リサンプル

```python
import torchaudio.functional as F
shifted = F.pitch_shift(audio_t, sample_rate=16000, n_steps=effective_shift)
# shifted.shape == audio_t.shape （サンプル数保存）
```

| 項目 | 評価 |
|---|---|
| サンプル数 | 保存される — フレーム整合が維持される |
| 品質 | STFT位相アーティファクト（金属的な響き）あり |
| レイテンシ | CPU: ~2-5ms / XPU: 未検証（STFT→カーネル互換性に要注意） |
| 依存関係 | torchaudio（既存） |
| 実装難易度 | 低 — 1行で完了、パイプライン変更最小 |
| フォルマント | ピッチと共にシフトされる（自然な変換） |

最大の利点: サンプル数が変わらないため、既存のトリム計算・固定サイズパディング・HuBERTフレームアライメントが全て無変更で動作する。

懸念点: XPUでのtorchaudio STFT互換性。CPU fallback必要かもしれない。

### B. リサンプルトリック（時間長変化あり）

原理: リサンプルで周波数を変更（時間長も変わる）

```python
r = 2 ** (effective_shift / 12)
target_len = int(len(audio_16k) / r)
shifted = scipy.signal.resample(audio_16k, target_len)
# サンプル数が変わる（+12半音 → 半分の長さ）
```

| 項目 | 評価 |
|---|---|
| サンプル数 | 変化する — フレーム整合が崩れる |
| 品質 | 高品質（FFTベース、位相アーティファクトなし） |
| レイテンシ | ~0.3ms（非常に高速） |
| 依存関係 | scipy（既存） |
| 実装難易度 | 高 — トリム計算、固定サイズパディング、オーバーラップすべて要修正 |
| フォルマント | ピッチと共にシフトされる |

重要: 「二重リサンプル」（L→L/r→L）はピッチシフトにならない。数学的にはローパスフィルタに等しい。正しくはリサンプル1回で長さが変わる。

課題:
- `fixed_hubert_input`パディング: 短い音声を固定長にパディング → HuBERTフレーム数は一定だが、「実コンテンツ」の占める割合が変わり、トリム位置がずれる
- `_streaming_audio_history`: 異なる長さの音声を蓄積する際の整合性
- `overlap_samples`: リサンプル後に320倍数の制約を維持する必要

### C. HuBERT専用シフト（F0は元音声で抽出）

原理: HuBERTへの入力のみピッチシフト、F0は元音声から抽出

```python
# audio_tは元音声（F0用）
# audio_t_shifted はシフト済み音声（HuBERT用）
features = hubert.extract(audio_t_shifted, ...)  # シフト済み
f0 = fcpe.infer(audio_t, ...)                     # 元音声
```

| 項目 | 評価 |
|---|---|
| F0精度 | 最高 — 元の音声からF0抽出（シフトの影響なし） |
| レイテンシ | シフト処理 + 2つの異なるテンソル管理 |
| 実装難易度 | 中 — 並列抽出の構造変更が必要 |
| フレーム整合 | シフト方式による（Aなら保存、Bなら要補間） |

利点: F0抽出がシフトの影響を受けないため、residual pitch shift計算が不要。

懸念: HuBERTの特徴量フレームとF0フレームの時間的対応が崩れる可能性（B方式の場合）。

### D. 部分シフト（比率制御）

原理: full `pitch_shift`の一部（50-100%）のみをプレシフトに適用

これは独立したアプローチではなく、A/B/Cの上に乗せるパラメータ。

```python
effective_shift = pitch_shift * pre_hubert_ratio  # ratio: 0.0-1.0
residual_shift = pitch_shift * (1.0 - pre_hubert_ratio)  # 残りはF0に適用
```

### E. 適応的シフト（F0ベース）

原理: 入力F0を検出し、ターゲット範囲との差分に基づいてシフト量を決定

問題: F0検出がHuBERTと並列実行される現行設計では、F0結果をHuBERTの前に使えない。F0を先に実行するとレイテンシ増加。

---

## 推奨アプローチ: A + C + D の組み合わせ

### 根拠

1. **方式A** (`torchaudio pitch_shift`) を基盤とする
   - サンプル数保存 → パイプライン変更最小
   - torchaudio既存依存
   - 位相アーティファクトはHuBERTの入力としては許容範囲（HuBERTはrobust — 軽度の音声劣化に耐性がある）
2. **方式C** (HuBERT専用シフト) を組み合わせる
   - F0は元音声から抽出（精度維持）
   - HuBERTのみシフト済み音声を処理
   - residual pitch shift計算が不要になり、ロジックがシンプルに
3. **方式D** (比率制御) でチューニング可能に
   - `pre_hubert_pitch_ratio: float` (0.0-1.0) パラメータ
   - 0.0 = 無効（現行動作）、1.0 = full pitch_shift分をプレシフト
   - デフォルト0.0で後方互換

### 処理フロー（推奨案）

```text
audio_16k
  ↓
[history buffer 蓄積] (元音声のまま)
  ↓
highpass → reflect padding → audio_padded
  ↓
audio_t = torch.from_numpy(audio_padded)  ← 元音声テンソル
  ↓
┌─────────────────────────────────────────────┐
│ if pre_hubert_pitch_ratio > 0:              │
│   effective = pitch_shift * ratio           │
│   audio_t_for_hubert = pitch_shift(audio_t) │
│ else:                                       │
│   audio_t_for_hubert = audio_t              │
└─────────────────────────────────────────────┘
  ↓                              ↓
HuBERT(audio_t_for_hubert)   F0(audio_t)  ← 並列実行
  ↓                              ↓
features (50fps)              f0 (100fps)
  ↓                              ↓
[FAISS index search]          pitch_shift適用（full amount、元音声のF0なので）
  ↓                              ↓
  └─────────────┬────────────────┘
                ↓
           synthesizer
```

### F0処理の簡素化

方式Cを採用するため、F0は元音声から抽出される。
したがって、F0への`pitch_shift`適用は現行のまま（full `pitch_shift`）でよい。
residual計算は不要。

```python
# 現行のまま変更なし
if pitch_shift != 0:
    f0 = torch.where(f0 > 0, f0 * (2 ** (pitch_shift / 12)), f0)
```

---

## 実装設計

### 変更ファイル

| ファイル | 変更内容 |
|---|---|
| `rcwx/config.py` | `InferenceConfig`に`pre_hubert_pitch_ratio`追加 |
| `rcwx/pipeline/inference.py` | `infer_streaming()`と`infer()`にプレシフト挿入 |
| `rcwx/pipeline/realtime_unified.py` | `RealtimeConfig`にフィールド追加、パラメータ透過 |
| `rcwx/gui/widgets/pitch_control.py` | トグル/スライダー追加 |
| `rcwx/gui/app.py` | コールバック接続 |
| `rcwx/cli.py` | `--pre-hubert-pitch`引数追加 |
| `tests/` | 新規テスト |

### Step 1: Config追加

```python
# rcwx/config.py InferenceConfig:
pre_hubert_pitch_ratio: float = 0.0  # 0.0-1.0

# rcwx/pipeline/realtime_unified.py RealtimeConfig:
pre_hubert_pitch_ratio: float = 0.0
```

### Step 2: コア実装 (inference.py)

`infer_streaming()` の変更（line ~1374付近）:

```python
# --- Pre-HuBERT pitch shift ---
audio_t_for_hubert = audio_t
if pre_hubert_pitch_ratio > 0 and pitch_shift != 0:
    effective_shift = pitch_shift * pre_hubert_pitch_ratio
    try:
        import torchaudio.functional as AF
        audio_t_for_hubert = AF.pitch_shift(
            audio_t, sample_rate=16000, n_steps=effective_shift
        )
    except Exception:
        logger.warning("[INFER] torchaudio pitch_shift failed, skipping pre-shift")
        audio_t_for_hubert = audio_t

# Parallel extraction: HuBERT uses shifted audio, F0 uses original
def extract_hubert():
    with torch.autocast(device_type=self.device, dtype=self.dtype):
        return self.hubert.extract(
            audio_t_for_hubert, output_layer=output_layer, output_dim=output_dim
        )

def extract_f0():
    # F0は元音声から（現行のまま）
    ...
```

同様の変更を `infer()` にも適用。

重要: F0の`pitch_shift`適用コード（line 1455-1456）は変更なし。F0は元音声から抽出されるため、full `pitch_shift`を適用する現行ロジックが正しい。

### Step 3: リアルタイムパイプライン透過

```python
# realtime_unified.py _inference_thread():
output_model = self.pipeline.infer_streaming(
    ...,
    pre_hubert_pitch_ratio=self.config.pre_hubert_pitch_ratio,
)
```

`set_pre_hubert_pitch_ratio()` ミューテータ追加。

### Step 4: GUI

`pitch_control.py` にチェックボックス追加:
- ラベル: "プレHuBERTシフト"
- チェック時 `ratio=1.0`、非チェック時 `ratio=0.0`
- 将来的にスライダーに変更可能

### Step 5: CLI

`cli.py` に `--pre-hubert-pitch` 引数追加（float, default 0.0）。

### Step 6: テスト

`tests/integration/test_pre_hubert_pitch.py`:
1. 同一性テスト: `ratio=0.0` → 現行と同一出力
2. サンプル数保存: pitch_shift後も`audio_t.shape`不変
3. 周波数シフト検証: FFTピーク比較で実際にシフトされていることを確認
4. 品質テスト: pre-shift ON vs OFF でHuBERT特徴量のコサイン類似度比較
5. パフォーマンス: pitch_shift処理が<5ms

---

## XPU互換性に関する注意

`torchaudio.functional.pitch_shift` は内部でSTFT/ISTFTを使用する。Intel XPUでのSTFTカーネル互換性は未検証。

フォールバック戦略:
```python
try:
    shifted = AF.pitch_shift(audio_t, ...)
except RuntimeError:
    # XPU非対応の場合、CPUで実行
    shifted = AF.pitch_shift(audio_t.cpu(), ...).to(audio_t.device)
```

CPU fallbackでも8960サンプルなら~3ms程度で許容範囲内。

---

## POC検証（先行実施）

`debuug_audio/01_input_raw.wav` を使って以下を検証:

1. `torchaudio.functional.pitch_shift` でピッチシフト（+5, +12, -5, -12半音）
2. シフト前後のスペクトログラム比較
3. シフト前後のHuBERT特徴量比較（コサイン類似度）
4. 処理時間計測
5. XPU/CPU互換性テスト

出力: `debuug_audio/poc_pre_hubert_shift/` に結果WAVとスペクトログラム画像を保存

## 検証手順（本実装後）

1. `uv run python tests/integration/test_pre_hubert_pitch.py` — 単体テスト
2. `uv run python tests/integration/test_infer_streaming.py` — 既存テストが壊れていないこと
3. GUI起動 → モデルロード → `pitch_shift=+12` → プレシフトON/OFF切替 → 音質比較
4. パフォーマンスモニタ: レイテンシ表示でpre-shift有効時の増加量確認
