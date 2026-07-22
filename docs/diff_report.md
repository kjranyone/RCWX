# RVC V2 オリジナル実装との差異報告

> **文書の位置づけ**: RVC 互換性を揃える過程の監査メモ。下記「修正完了」は当時の作業結果。  
> **現行コードとの乖離**は冒頭の [現行メモ](#現行メモ-as-is) を優先すること。セットアップ手順は [SETUP.md](SETUP.md)。

## 概要

RCWX の推論パイプラインと RVC V2 オリジナル実装（RVC-Project/Retrieval-based-Voice-Conversion-WebUI）を比較し、差異を報告する。

## 現行メモ (AS-IS)

コード確認時点の事実。本文中の古い記述と食い違う場合はこちらを正とする。

| 項目 | 現行 | 本文の旧記述 |
|------|------|-------------|
| 特徴量 50fps→100fps 補間 | `mode="linear"`（`inference.py`） | nearest に変更済み |
| FAISS `index_k` 既定 | **4**（品質寄りは 8 を GUI/設定で選択可） | k=8 固定 |
| RMVPE 呼び出し閾値 | ストリーミング経路は `RMVPE_VOICING_THRESHOLD=0.015` | モデル `infer` 既定 0.03 |
| FAISS 経路 | CPU FAISS + **Aggressive 時 XPU IVF**（L2 / nprobe=1 / list≤256） | CPU FAISS のみ |
| 既定 F0 | config 既定 `rmvpe`（FCPE / SwiftF0 も選択可） | （本文は手法別） |
| 対象デバイス | Intel Arc **XPU**（CUDA は未検証） | — |

---

## 1. HuBERT特徴量抽出 ✅ 修正完了

| 項目 | オリジナルRVC | RCWX実装 | 差異 |
|------|-------------|----------|------|
| モデル | fairseq HuBERT | transformers HubertModel | ライブラリは異なるが出力一致 |
| 出力レイヤー | v2: 12層目 | v2: 12層目 | 一致 |
| 出力次元 | v2: 768次元 | v2: 768次元 | 一致 |
| 入力正規化 | グループ正規化 | transformers内部で処理 | 実装依存 |
| 入力精度 | float32 | float32 | ✅ 一致 |
| layer_norm | fairseq: 常に適用 | v2: last_hidden_state使用 | ✅ 一致 |

### 修正内容

- transformers版 `lengyue233/content-vec-best` を使用（fairseq依存回避）
- **layer_norm 問題を修正**: HuggingFace の `hidden_states[N]` は layer_norm **前** の出力。
  fairseq は常に `encoder.layer_norm` を適用するため、分布不一致が生じていた。
  - v2 (output_layer=12): `outputs.last_hidden_state` を使用（layer_norm 込み）
  - v1 (output_layer<12): `hidden_states[N]` + `model.encoder.layer_norm()` を手動適用

### 関連ファイル

- `rcwx/models/hubert_loader.py`

---

## 2. RMVPE F0抽出 ✅ 修正完了

| 項目 | オリジナルRVC | RCWX実装 | 差異 |
|------|-------------|----------|------|
| ホップサイズ | 160 | 160 | 一致 |
| 閾値 | 0.03 | 0.03 | 一致 |
| 閾値判定対象 | 全360ビンの最大salience | 全360ビンの最大salience | **一致** |
| mel抽出精度 | float32 | float32 | **一致** |
| モデルアーキテクチャ | DeepUnet + E2E | DeepUnet + E2E | **一致** |
| Encoder構造 | BatchNorm + ResEncoderBlock | BatchNorm + ResEncoderBlock | **一致** |
| GRU層数 | 1層 (n_gru=1) | 1層 (n_gru=1) | **一致** |

### 修正内容

1. **閾値判定**: 9ビン窓ではなく、全360ビンの最大salienceで判定するよう修正
2. **数値精度**: mel抽出をfloat32で実行（float16のSTFT不安定性を回避）
3. **モデルアーキテクチャ完全一致**:
   - `Encoder`クラス: 初期BatchNorm (`self.bn`) を追加
   - `ResEncoderBlock`: kernel_size=None時はpoolingなしで単一tensor返却
   - `ResDecoderBlock`: stride-based upsampling、出力paddingの計算修正
   - `Intermediate`: ResEncoderBlock with kernel_size=None使用
   - `E2E`: n_gru=1 (チェックポイントに合わせて修正)
4. **サイズミスマッチ対応**: ResDecoderBlockで奇数サイズ入力時のパディング処理
5. **mel filterbank**: librosa互換のHTKスケール mel filterbank（エリア正規化）
   - numbaがNumPy 2.4非対応のためlibrosaのインポートが失敗する環境でも、手動実装で同一出力を実現
   - 手動実装とlibrosaの出力は完全一致（sum=8.1892）
6. **入力形式**: `mel.transpose(-1, -2).unsqueeze(1)` でオリジナルRVCと同じ向きに修正
   - 修正前: `[B, 1, mel_bins, frames]`
   - 修正後: `[B, 1, frames, mel_bins]` ← オリジナルRVC互換

```python
# Encoder: 初期BatchNormを追加
class Encoder(nn.Module):
    def __init__(self, in_channels, in_size, n_encoders, ...):
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList([...])

    def forward(self, x):
        x = self.bn(x)  # 初期正規化
        for layer in self.layers:
            skip, x = layer(x)
            concat_tensors.append(skip)
        return x, concat_tensors

# ResDecoderBlock: サイズミスマッチ対応
def forward(self, x, concat_tensor):
    x = self.conv1(x)
    if x.shape[2] != concat_tensor.shape[2] or x.shape[3] != concat_tensor.shape[3]:
        diff_h = concat_tensor.shape[2] - x.shape[2]
        diff_w = concat_tensor.shape[3] - x.shape[3]
        x = F.pad(x, (0, diff_w, 0, diff_h))
    x = torch.cat((x, concat_tensor), dim=1)
    ...
```

### 関連ファイル

- `rcwx/models/rmvpe.py`

---

## 3. F0量子化 ✅ 修正完了

| 項目 | オリジナルRVC | RCWX実装 | 差異 |
|------|-------------|----------|------|
| スケール | ログスケール線形 | melスケール | 方式は同等 |
| ビン範囲 | 1-255 (0は未使用) | 1-255 | 一致 |
| unvoiced処理 | `f0_mel <= 1` → 1 | clamp(1,255) | **一致** |

### 修正内容

オリジナルRVCのロジックに完全一致するよう修正済み：

```python
# Convert F0 to mel scale (f0=0 -> f0_mel=0)
f0_mel = 1127 * torch.log(1 + f0 / 700)

# Only normalize voiced frames (f0_mel > 0)
voiced_mask = f0_mel > 0
f0_mel_normalized = torch.where(
    voiced_mask,
    (f0_mel - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1,
    f0_mel  # Keep 0 for unvoiced
)

# Clamp to valid range and set low values to 1
pitch = torch.clamp(f0_mel_normalized, 1, 255).round().long()
```

### 関連ファイル

- `rcwx/pipeline/inference.py`

---

## 4. 特徴量アライメント ✅ 修正完了

| 項目 | オリジナルRVC | RCWX実装 | 差異 |
|------|-------------|----------|------|
| 補間方法 | 固定2x upscale | 固定2x upscale | **一致** |
| 補間モード | nearest（PyTorch 既定） | **linear**（現行） | 意図的差分の可能性。要再確認 |

### 現行コード

```python
features = torch.nn.functional.interpolate(
    features.permute(0, 2, 1),  # [B, T, C] -> [B, C, T]
    scale_factor=2,
    mode="linear",
    align_corners=False,
).permute(0, 2, 1)
```

> 過去の監査では nearest への変更を記録していたが、現行 `inference.py` は `linear`。

### 関連ファイル

- `rcwx/pipeline/inference.py`

---

## 5. 入力前処理 ✅ ほぼ一致

| 項目 | オリジナルRVC | RCWX実装 | 差異 |
|------|-------------|----------|------|
| ハイパスフィルタ | 48Hz, 5次Butterworth | 48Hz, 5次Butterworth | ✅ 一致 |
| パディング | reflect padding | reflect padding | ✅ 一致 |
| パディング量 | 設定可能 (x_pad) | 固定50ms | 軽微な差異 |

### 備考

- パディング量50msはオリジナルRVCのデフォルト設定に近い
- 必要であれば将来的にパラメータ化可能

### 関連ファイル

- `rcwx/pipeline/inference.py`

---

## 6. Index検索（FAISS）✅ 実装完了

| 項目 | オリジナルRVC | RCWX実装 | 差異 |
|------|-------------|----------|------|
| FAISS検索 | あり (index_rate) | あり (index_rate) | **一致** |
| 特徴融合 | `入力*(1-rate) + 検索*rate` | `入力*(1-rate) + 検索*rate` | **一致** |

### 実装内容

オリジナルRVCと同等のFAISS検索機能を実装：

```python
def _search_index(self, features, index_rate=0.5, k=4):
    # Convert to numpy for FAISS search
    npy = features[0].cpu().numpy()

    # Search for k nearest neighbors (default 4)
    score, ix = self.faiss_index.search(npy, k=k)

    # Compute inverse squared distance weights
    weight = np.square(1 / (score + 1e-6))
    weight /= weight.sum(axis=1, keepdims=True)

    # Weighted average of retrieved features
    retrieved = np.sum(self.index_features[ix] * np.expand_dims(weight, axis=2), axis=1)

    # Blend with original features
    blended = index_rate * retrieved_tensor + (1 - index_rate) * features
    return blended
```

### 機能（現行）

- .index ファイルの自動検出（モデルと同ディレクトリ）
- `index_rate` で融合率を制御（0=無効、1=index 完全使用）。config 既定 ratio=0.15
- `index_k` 既定 **4**（8=品質寄り）。逆二乗距離重み付け
- Aggressive では対応 IVF を XPU 常駐化（`accelerator_index.py`）。未対応は CPU FAISS

### 関連ファイル

- `rcwx/pipeline/inference.py`

---

## 7. Synthesizer推論 ✅ 修正完了

| 項目 | オリジナルRVC | RCWX実装 | 差異 |
|------|-------------|----------|------|
| TextEncoder | nn.Linear (emb_phone) | nn.Linear (emb_phone) | **一致** |
| 入力形式 | [B, T, C] | [B, T, C] | **一致** |
| infer引数順序 | (phone, phone_len, pitch, pitchf, sid) | 同じ | 一致 |
| スケーリング | sqrt(hidden_channels) | sqrt(hidden_channels) | **一致** |
| 活性化関数 | LeakyReLU(0.1) | LeakyReLU(0.1) | **一致** |

### 修正内容

TextEncoderをオリジナルRVCと完全一致するよう修正：

```python
class TextEncoder(nn.Module):
    def __init__(self, ...):
        # Original RVC uses Linear embedding, not Conv1d
        self.emb_phone = nn.Linear(in_channels, hidden_channels)
        self.lrelu = nn.LeakyReLU(0.1)

    def forward(self, phone, phone_lengths, pitch=None):
        # Embed phone features: [B, T, C] -> [B, T, hidden]
        x = self.emb_phone(phone)
        if pitch is not None:
            x = x + self.emb_pitch(pitch)
        x = x * math.sqrt(self.hidden_channels)
        x = self.lrelu(x)
        x = x.transpose(1, 2)  # -> [B, hidden, T]
        ...
```

### 関連ファイル

- `rcwx/models/synthesizer.py`
- `rcwx/models/infer_pack/models.py`

---

## まとめ

### 修正状況

| 優先度 | 項目 | 状態 | 説明 |
|--------|------|------|------|
| **高** | TextEncoder | ✅ 完了 | nn.Conv1d→nn.Linear、入力形式修正 |
| **高** | 特徴量アライメント | ⚠ 要確認 | 2x 固定は一致。モードは現行 `linear`（旧メモは nearest） |
| **高** | FAISS Index検索 | ✅ 完了 | index_rate 対応、k 可変（既定 4）、Aggressive で XPU IVF |
| **中** | F0量子化 | ✅ 完了 | オリジナルRVCのロジックに一致 |
| **中** | HuBERTモデル | ✅ 完了 | transformers版使用、layer_norm問題修正済み |
| **高** | RMVPEアーキテクチャ | ✅ 完了 | オリジナルRVC完全一致、重み正常ロード |
| **低** | RMVPE閾値判定 | ✅ 完了 | 全360ビンの最大salience使用 |

### 完了した修正

1. **TextEncoder** - `nn.Conv1d`→`nn.Linear`（emb_phone）、入力形式`[B,T,C]`に統一
2. **特徴量補間** - 2x 固定 upscale（現行 `mode="linear"`。nearest への変更は現行コードに残っていない）
3. **FAISS Index** - `index_rate`、自動 index 検出、k 可変（既定 4）
4. **F0量子化** - オリジナルRVCと同一ロジック
5. **RMVPE** - 閾値判定を全360ビンの最大salience使用に修正
6. **RMVPEアーキテクチャ** - オリジナルRVCと完全一致するよう再実装:
   - Encoder: 初期BatchNorm追加
   - ResEncoderBlock: kernel_size=None対応
   - ResDecoderBlock: stride-based upsampling、サイズミスマッチ対応
   - E2E: n_gru=1（チェックポイント互換）
   - mel filterbank: librosa htk=True対応（エリア正規化）
   - 入力形式: transpose修正 `[B,1,frames,mel_bins]`

---

## 参考資料

- [RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- [Understanding RVC - gudgud96's Blog](https://gudgud96.github.io/2024/09/26/annotated-rvc/)
- [RMVPE Paper](https://arxiv.org/pdf/2306.15412)
