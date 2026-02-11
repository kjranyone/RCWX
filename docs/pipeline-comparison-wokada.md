# RCWX vs w-okada/voice-changer 推論パイプライン比較

RVCリアルタイムボイスチェンジャーの2つの実装を、推論パイプラインの観点から詳細に比較する。

- **RCWX**: Intel XPU向けに最適化されたネイティブデスクトップ実装
- **w-okada/voice-changer**: マルチモデル対応のサーバー・クライアント型フレームワーク

---

## 1. 全体アーキテクチャ

### 1.1 RCWX

```mermaid
graph TD
    MIC[Microphone 48kHz] -->|sounddevice callback| ACC[Input Accumulator]
    ACC -->|hop単位| IQ[Input Queue]
    IQ --> IT[Inference Thread]

    subgraph IT[Inference Thread]
        direction TB
        G[Input Gain] --> RS1[StatefulResampler 48k->16k]
        RS1 --> DN[Optional Denoise]
        DN --> OVL[Overlap Assembly]
        OVL --> INF[infer_streaming]
        INF --> RS2[StatefulResampler 40k->48k]
        RS2 --> SC[Soft Clip]
        SC --> SOLA[SOLA Crossfade]
        SOLA --> FB[Feedback Detection]
    end

    IT -->|output chunk| OQ[Output Queue]
    OQ --> RB[RingOutputBuffer]
    RB -->|sounddevice callback| SPK[Speaker 48kHz]

    style IT fill:#e8f4fd,stroke:#2196F3
    style IQ fill:#fff3e0,stroke:#FF9800
    style OQ fill:#fff3e0,stroke:#FF9800
```

- 推論は専用スレッドで非同期実行
- 入出力は `sounddevice` コールバック駆動
- Queue によるバッファリングで I/O と推論を分離

### 1.2 w-okada/voice-changer

```mermaid
graph TD
    MIC[Microphone 48kHz] -->|sounddevice callback| PROC[_processDataWithTime]

    subgraph PROC[Audio Input Callback - 同期処理]
        direction TB
        GI[generate_input] --> INF[RVC.inference]
        INF --> SOLA[SOLA Crossfade]
        SOLA --> PAD[pad_array]
    end

    PROC -->|put| OQ[outQueue]
    OQ -->|sounddevice callback| SPK[Speaker 48kHz]

    WEB[Web UI / Socket.IO] -.->|設定変更| PROC

    style PROC fill:#fce4ec,stroke:#E91E63
    style OQ fill:#fff3e0,stroke:#FF9800
```

- 推論はオーディオ入力コールバック内で**同期実行**
- 推論がチャンク時間を超えるとフレームドロップが直接発生
- Web UI からの設定変更はリアルタイム反映

### 1.3 設計思想の比較

| 観点 | RCWX | w-okada |
|---|---|---|
| アプリ形態 | ネイティブ GUI (tkinter) | Web サーバー + ブラウザ UI |
| スレッドモデル | 推論専用スレッド (非同期) | コールバック内同期処理 |
| 対応モデル | RVC v2 専用 | RVC, MMVC, DDSP-SVC, Beatrice 等 8+ |
| 対象デバイス | Intel XPU (Arc) | CUDA / CPU / DirectML |
| 過負荷対策 | 自動品質退避 (F0/Index無効化) | フレームドロップ (暗黙) |

---

## 2. 推論パイプライン詳細フロー

### 2.1 RCWX: `infer_streaming()` フロー

```mermaid
flowchart TB
    subgraph input[入力準備]
        A1[audio_16k<br/>overlap + new_hop] --> A2[Audio History 蓄積<br/>最大 ~8960 samples]
        A2 --> A3[High-pass Filter<br/>48Hz Butterworth 5次]
        A3 --> A4[Reflection Padding<br/>800 samples 両端]
        A4 --> A5[固定サイズ Padding<br/>XPU再コンパイル回避]
    end

    subgraph extract[特徴抽出 - 並列]
        direction LR
        B1[HuBERT Extract<br/>layer 12, 768d<br/>50fps]
        B2[F0 Extract<br/>FCPE / RMVPE<br/>100fps]
    end

    subgraph process[後処理]
        C1[FAISS Index Search<br/>optional] --> C2[Feature Interpolation<br/>2x nearest 50fps->100fps]
        C2 --> C3[F0 Post-processing<br/>median + lowpass + octave flip + slew limit]
        C3 --> C4[Pitch Quantization<br/>mel scale, 256 bins]
    end

    subgraph synth[合成]
        D1[Feature Cache Prepend<br/>短チャンク補完] --> D2[Synthesizer Inference<br/>RVC decoder]
        D2 --> D3[Voice Gate<br/>off/strict/expand/energy]
        D3 --> D4[Output Trim<br/>overlap + padding除去]
    end

    input --> extract
    extract --> process
    process --> synth

    style extract fill:#e8f5e9,stroke:#4CAF50
    style synth fill:#f3e5f5,stroke:#9C27B0
```

**特徴的な点**:
- HuBERT と F0 を `ThreadPoolExecutor` で**並列抽出** (~10-15% 高速化)
- Audio History を蓄積し HuBERT に豊富なコンテキストを提供
- 固定サイズパディングにより XPU カーネル再コンパイルを回避
- Feature Cache により短チャンク（初回2チャンク）でも実特徴量でデコーダーを補完

### 2.2 w-okada: `Pipeline.exec()` フロー

```mermaid
flowchart TB
    subgraph input[入力準備]
        A1[audio_buffer<br/>convertSize分の蓄積音声] --> A2[torchaudio.resample<br/>model_sr -> 16kHz]
        A2 --> A3[Reflection Padding<br/>quality mode時]
    end

    subgraph extract[特徴抽出 - シーケンシャル]
        B1[F0 Extract<br/>RMVPE / Crepe等] --> B2[HuBERT Extract<br/>ContentVec]
    end

    subgraph process[後処理]
        C1[FAISS Index Search<br/>+ feature blending] --> C2[Feature Interpolation<br/>2x upscale]
        C2 --> C3[Pitch Protection<br/>無声子音保護]
    end

    subgraph synth[合成]
        D1[Synthesizer Inference<br/>out_size指定] --> D2[Padding Trim<br/>reflection除去]
        D2 --> D3[Volume Scaling<br/>× sqrt vol]
    end

    input --> extract
    extract --> process
    process --> synth

    style extract fill:#fff3e0,stroke:#FF9800
    style synth fill:#f3e5f5,stroke:#9C27B0
```

**特徴的な点**:
- HuBERT と F0 は**シーケンシャル実行** (並列化なし)
- `extraConvertSize` により大量のコンテキスト音声を投入
- `out_size` パラメータでシンセサイザー出力長を直接制御
- Pitch Protection で無声子音 (破裂音等) のindex変換を抑制

---

## 3. チャンクバッファレイアウト

推論に渡される音声データの構成が根本的に異なる。

### 3.1 w-okada のバッファ構成

```mermaid
block-beta
    columns 4

    block:extra["extraConvertSize\n(既定 4096 samples)\nコンテキスト専用"]
        space
    end
    block:sola["solaSearchFrame\n(12ms)\nSOLA探索窓"]
        space
    end
    block:cf["crossfadeFrame\n(既定 4096 samples)\nクロスフェード"]
        space
    end
    block:blk["blockFrame\n新規出力"]
        space
    end

    style extra fill:#e3f2fd,stroke:#1565C0
    style sola fill:#fff9c4,stroke:#F9A825
    style cf fill:#fce4ec,stroke:#C62828
    style blk fill:#e8f5e9,stroke:#2E7D32
```

```
|<-- extraConvertSize -->|<-- solaSearch -->|<-- crossfade -->|<-- block -->|
|     context only       |   SOLA search   |  crossfade zone |  new output |
|<-------------------------- convertSize (synth入力) ---------------------->|
                         |<------------------ outSize (synth出力) --------->|
```

- `convertSize` 全体をシンセサイザーに入力し、`outSize` だけ出力させる
- `extraConvertSize` はユーザーが手動調整可能 (4096 ~ 131072 samples)
- 128-sample 境界にアラインメント (シンセサイザーの hop size)

### 3.2 RCWX のバッファ構成

```mermaid
block-beta
    columns 3

    block:pad1["Reflection Pad\n800 samples (50ms)"]
        space
    end
    block:main["overlap + new_hop\n(HuBERT hop=320 アライン)"]
        space
    end
    block:pad2["Reflection Pad\n800 + extra"]
        space
    end

    style pad1 fill:#fff9c4,stroke:#F9A825
    style main fill:#e8f5e9,stroke:#2E7D32
    style pad2 fill:#fff9c4,stroke:#F9A825
```

```
|<-- pad (800) -->|<-- overlap -->|<-- new_hop -->|<-- pad (800+extra) -->|
|   reflection    | 前チャンク尾  |   新規音声    |      reflection       |
```

さらに、HuBERT には別途蓄積した Audio History を連結:

```
|<-- history (最大 ~8960 samples) -->|<-- overlap -->|<-- new_hop -->|
|       _streaming_audio_history     | 前チャンク尾  |   新規音声    |
```

- 全て 320-sample (HuBERT hop) 境界にアラインメント
- HuBERT 入力は固定長にパディング (XPU 最適化)
- 出力トリムで overlap + padding + history 分をカット

### 3.3 コンテキスト戦略の比較

```mermaid
graph LR
    subgraph wokada[w-okada: brute force型]
        E1[extraConvertSize<br/>~85ms at 48kHz] --> E2[大量コンテキスト<br/>一括処理]
        E2 --> E3[ユーザー手動調整]
    end

    subgraph rcwx[RCWX: 効率型]
        R1[overlap_sec<br/>100ms 自動導出] --> R2[Audio History<br/>~560ms 自動蓄積]
        R2 --> R3[多段後処理<br/>F0 smooth等]
    end

    style wokada fill:#fff3e0,stroke:#FF9800
    style rcwx fill:#e8f4fd,stroke:#2196F3
```

| 観点 | RCWX | w-okada |
|---|---|---|
| コンテキスト供給 | `overlap_sec` + Audio History (自動) | `extraConvertSize` (ユーザー設定) |
| 最大コンテキスト | ~560ms (自動上限) | ~3秒 (ユーザー設定次第) |
| 品質担保手段 | 多段後処理 (F0 smooth, feature blend) | 大量コンテキスト投入 |
| レイテンシ影響 | コンテキスト増でも推論時間ほぼ一定 | コンテキスト増で推論時間が線形増加 |

---

## 4. 特徴量抽出

### 4.1 HuBERT

```mermaid
flowchart LR
    subgraph rcwx[RCWX]
        direction TB
        RA[Audio History 蓄積<br/>最大 ~8960 samples] --> RB[固定サイズ Pad<br/>XPU最適化]
        RB --> RC[HuBERT forward<br/>layer 12 / 768d]
        RC --> RD[2x interpolation<br/>50fps -> 100fps]
    end

    subgraph wokada[w-okada]
        direction TB
        WA[convertSize全体<br/>可変長] --> WB[Reflection Pad<br/>quality mode時]
        WB --> WC[HuBERT forward<br/>layer 9 or 12]
        WC --> WD[2x interpolation<br/>50fps -> 100fps]
    end

    style rcwx fill:#e8f4fd,stroke:#2196F3
    style wokada fill:#fff3e0,stroke:#FF9800
```

| 観点 | RCWX | w-okada |
|---|---|---|
| 入力 SR | 16kHz (事前リサンプル済み) | 16kHz (Pipeline.exec 内でリサンプル) |
| コンテキスト | `_streaming_audio_history` (最大 ~560ms) | `audio_buffer` 全体 (convertSize分) |
| パディング | **固定サイズ** (XPU カーネル再コンパイル回避) | 可変サイズ (チャンク毎に変動) |
| 出力 layer | v2: layer 12 / 768d | 設定可 (v1: layer 9/256d, v2: layer 12/768d) |
| 補間 | `F.interpolate(scale_factor=2, mode="nearest")` | 同一 |

**RCWX の固定サイズパディング**は Intel XPU 特有の最適化。XPU (oneAPI) はチャンク毎に入力サイズが変わるとカーネルを再コンパイルするため、初回チャンクで1回だけコンパイルし以降は再利用する設計:

```python
# rcwx/pipeline/inference.py (infer_streaming)
fixed_hubert_input = (
    (min_audio_for_full_features + 2 * t_pad + hubert_hop - 1)
    // hubert_hop * hubert_hop
)
if len(audio_padded) < fixed_hubert_input:
    end_pad = fixed_hubert_input - len(audio_padded)
    audio_padded = np.pad(audio_padded, (0, end_pad), mode="reflect")
```

### 4.2 F0 (ピッチ抽出)

```mermaid
flowchart TB
    subgraph rcwx[RCWX の F0 パイプライン]
        direction TB
        R1[FCPE or RMVPE<br/>パディング含む全音声] --> R3[Median Filter<br/>window=3, スパイク除去]
        R3 --> R4[Lowpass Filter<br/>configurable cutoff Butterworth]
        R4 --> R4b[Octave Flip Suppress<br/>1オクターブ飛び補正]
        R4b --> R4c[Slew Rate Limiter<br/>フレーム間最大ステップ制限]
        R4c --> R5[Pitch Shift<br/>semitone単位]
        R5 --> R6[Length Interpolation<br/>feature長に合わせる]
        R6 --> R7[Mel Quantization<br/>50-1100Hz -> 1-255]
    end

    subgraph wokada[w-okada の F0 パイプライン]
        direction TB
        W1[RMVPE / Crepe / DIO / Harvest<br/>パディング含む全音声] --> W2[Pitch Shift<br/>semitone単位]
        W2 --> W3[pitchf_buffer に書き込み<br/>負インデックスで上書き]
        W3 --> W4[Quantization]
    end

    style rcwx fill:#e8f4fd,stroke:#2196F3
    style wokada fill:#fff3e0,stroke:#FF9800
```

| 観点 | RCWX | w-okada |
|---|---|---|
| 既定手法 | FCPE (低レイテンシ) | RMVPE |
| 選択肢 | FCPE / RMVPE | RMVPE / Crepe / DIO / Harvest / FCPE |
| 並列化 | HuBERT と ThreadPoolExecutor で並列 | シーケンシャル |
| 後処理 | median + lowpass (configurable) + octave flip + slew limit | **なし** (生 F0 をそのまま使用) |
| キャッシュ | `_f0_cache` でチャンク境界値を保持 | `pitchf_buffer` で前チャンク値を保持 |
| 過負荷時 | `f0_method="none"` に自動退避 | なし |

RCWX の F0 後処理（median + lowpass + octave flip suppress + slew limit）により、チャンク境界でのピッチジャンプやフレーム間ジッターが軽減される。w-okada は `extraConvertSize` による大量コンテキストでF0境界品質を間接的に担保する。

---

## 5. SOLA クロスフェード

両実装とも SOLA (Synchronized Overlap-Add) を採用しているが、適用位置・窓関数・探索方法が異なる。

### 5.1 SOLA フロー比較

```mermaid
sequenceDiagram
    participant P as Previous Chunk
    participant S as SOLA
    participant C as Current Chunk

    Note over P,C: RCWX: Hold-back Design (48kHz 出力側)
    P->>S: 末尾 cf samples を hold-back (buffer)
    C->>S: 先頭 cf+search samples を探索領域に
    S->>S: 正規化相互相関 (cumsum最適化)
    S->>S: Hann窓クロスフェード
    S-->>C: target_len に強制出力

    Note over P,C: w-okada: Pre-multiply Design (model SR)
    P->>S: 末尾 cf samples × fade_out を保持
    C->>S: 先頭 cf+search samples を探索領域に
    S->>S: 正規化畳み込み (np.convolve)
    S->>S: cos² 窓クロスフェード
    S-->>C: block_frame samples を出力
```

### 5.2 実装の詳細比較

| 観点 | RCWX | w-okada |
|---|---|---|
| 適用位置 | 出力 SR (48kHz) | model SR (32k/40k/48k) |
| 探索窓 | `sola_search_ms` (既定 10ms) | 12ms 固定 |
| クロスフェード長 | `crossfade_sec` (既定 50ms) | `crossFadeOverlapSize` (既定 4096 samples) |
| 窓関数 | **Hann** (対称) | **cos²** + offset/end margin (0.1/0.9) |
| 相関計算 | 正規化相互相関 (cumsum O(N)) | 正規化畳み込み (`np.convolve`) |
| バッファ保持 | 生波形を hold-back | fade-out 事前適用済み |
| 出力長制御 | `target_len` で厳密制御 | block_frame 固定 |
| ドリフト防止 | あり (`target_len` 強制) | なし |

### 5.3 窓関数の違い

```mermaid
graph LR
    subgraph rcwx[RCWX: Hann窓]
        direction TB
        H1["fade_in: hann[0:cf] (0→1)"]
        H2["fade_out: hann[cf:2cf] (1→0)"]
        H3["prev × fade_out + curr × fade_in"]
    end

    subgraph wokada["w-okada: cos² + margin"]
        direction TB
        C1["10% flat (prev=1.0)"]
        C2["80% cos² transition"]
        C3["10% flat (curr=1.0)"]
        C4["prev × cos²_out + curr × cos²_in"]
    end

    style rcwx fill:#e8f4fd,stroke:#2196F3
    style wokada fill:#fff3e0,stroke:#FF9800
```

w-okada の cos² + margin 方式は、クロスフェード領域の両端にフラット区間 (10%) を設けることで、遷移境界でのアーティファクトを軽減する。RCWX の Hann 窓は標準的だが、cos² と比較して遷移の滑らかさはほぼ同等。

### 5.4 SOLA 相関計算の計算量

**RCWX** (`_find_best_offset`):

```python
# O(N) cumsum ベースの正規化相互相関
dots = np.correlate(region, pt_centered, mode="valid")  # C実装
# per-window norms via cumulative sums (大きな中間配列なし)
cumsum = np.cumsum(x)
cumsum_sq = np.cumsum(x * x)
norms = sqrt(window_sq_sums - window_sums² / cf)
corrs = dots / (pt_norm * norms)
```

**w-okada**:

```python
# np.convolve ベース
cor_nom = np.convolve(audio[:cf + search], np.flip(sola_buffer), "valid")
cor_den = np.sqrt(np.convolve(audio[:cf + search]**2, np.ones(cf), "valid") + 1e-3)
sola_offset = argmax(cor_nom / cor_den)
```

両者とも計算量は `O(cf × search)` だが、RCWX は中間配列の生成を抑えたメモリ効率の良い実装。

---

## 6. リサンプリング

### 6.1 アプローチの違い

```mermaid
flowchart LR
    subgraph rcwx[RCWX: StatefulResampler]
        direction TB
        R1[チャンクN] --> R2[overlap_buffer 連結]
        R2 --> R3[scipy resample_poly]
        R3 --> R4[overlap分トリム]
        R4 --> R5[tail保存 -> 次チャンクへ]
        R5 -.->|状態保持| R2
    end

    subgraph wokada[w-okada: Stateless]
        direction TB
        W1[チャンクN] --> W2[torchaudio.resample<br/>rolloff=0.99]
        W2 --> W3[出力]
    end

    style rcwx fill:#e8f4fd,stroke:#2196F3
    style wokada fill:#fff3e0,stroke:#FF9800
```

| 観点 | RCWX | w-okada |
|---|---|---|
| 方式 | `StatefulResampler` (overlap-save) | `torchaudio.functional.resample` |
| 状態保持 | あり (フィルタ位相連続) | なし (チャンク独立) |
| 品質 | バッチ比 ~0.98+ 相関 | チャンク境界にフィルタ過渡応答 |
| 入力パス | mic 48k -> 16k | model_sr -> 16k |
| 出力パス | model_sr 40k -> 48k | 不要 (model_sr = 出力SR の場合が多い) |

**StatefulResampler の動作原理**:

```
チャンク 1:
  [chunk1 + zero_pad] → resample → output1
  tail保存: chunk1[-overlap:]

チャンク 2:
  [saved_tail | chunk2] → resample → output2
  overlap分をトリム → phase-aligned output
  tail保存: chunk2[-overlap:]
```

これにより、チャンク境界でのフィルタ過渡応答 (transient) が除去され、バッチ処理とほぼ同等の品質を実現する。w-okada は各チャンクを独立にリサンプルするため、境界にわずかなアーティファクトが生じるが、SOLA クロスフェードで隠蔽される。

---

## 7. Voice Gate / サイレンス処理

### 7.1 処理フロー比較

```mermaid
flowchart TB
    subgraph rcwx[RCWX: 多モード Voice Gate]
        direction TB
        R1{voice_gate_mode?}
        R1 -->|off| R2[全パス]
        R1 -->|strict| R3[F0ベースのみ]
        R1 -->|expand| R4[有声区間を<br/>±30ms拡張<br/>max_pool1d]
        R1 -->|energy| R5[F0 + エネルギー<br/>閾値判定]
        R3 --> R6[5ms smoothing<br/>attack/release]
        R4 --> R6
        R5 --> R6
        R6 --> R7[output × gate_mask]
    end

    subgraph wokada[w-okada: Volume Gate]
        direction TB
        W1{vol < silentThreshold?}
        W1 -->|Yes| W2[ゼロ出力<br/>推論スキップ]
        W1 -->|No| W3[通常推論]
    end

    style rcwx fill:#e8f4fd,stroke:#2196F3
    style wokada fill:#fff3e0,stroke:#FF9800
```

| 観点 | RCWX | w-okada |
|---|---|---|
| 判定基準 | F0 voiced mask + energy | 全体 volume RMS |
| モード数 | 4 (off/strict/expand/energy) | 1 (threshold) |
| 破裂音保護 | expand モードで有声区間を拡張 | なし |
| attack/release | 5ms smoothing convolution | なし |
| 推論スキップ | なし (常に推論実行) | 閾値以下で推論完全スキップ |

w-okada の方式はシンプルだが、推論スキップによる計算節約が可能。RCWX は推論後にゲートを適用するため計算は削減されないが、きめ細かい無声区間制御が可能。

---

## 8. Synthesizer 最小入力長の処理

RVC シンセサイザーのデコーダーは最小入力長を要求する。この制約への対処が異なる。

```mermaid
flowchart TB
    subgraph rcwx[RCWX: Feature Cache + Reflect Pad]
        R1{features < 64 frames?}
        R1 -->|Yes, cache有| R2[前チャンクの実特徴量を prepend<br/>_streaming_feat_cache]
        R1 -->|Yes, cache無| R3[Reflect / Replicate Pad]
        R1 -->|No| R4[そのまま合成]
        R2 --> R5[合成後 prepend分をトリム]
    end

    subgraph wokada[w-okada: extraConvertSize]
        W1[convertSize = block + crossfade<br/>+ solaSearch + extra]
        W1 --> W2{convertSize 十分?}
        W2 -->|Yes| W3[そのまま合成<br/>out_size で出力長制御]
        W2 -->|No| W4[Zero Pad]
    end

    style rcwx fill:#e8f4fd,stroke:#2196F3
    style wokada fill:#fff3e0,stroke:#FF9800
```

| 観点 | RCWX | w-okada |
|---|---|---|
| 最小フレーム | `MIN_SYNTH_FEATURE_FRAMES = 64` (100fps) | 128-sample アラインメントのみ |
| 不足時の対処 | Feature cache prepend → reflect pad fallback | `extraConvertSize` で通常は不足しない |
| 初回チャンク | Reflect pad (キャッシュ未生成) | Zero pad |
| 品質 | 実特徴量による補完 (高品質) | ゼロ埋めまたは既定コンテキスト |

RCWX の `_streaming_feat_cache` は前チャンクから実際の特徴量 (HuBERT features + pitch + pitchf) を保存し、短チャンク時にデコーダーへの入力を補完する。reflect pad よりもデコーダーが自然な出力を生成できる。

---

## 9. 過負荷保護とフィードバック検出

### 9.1 RCWX 固有の仕組み

```mermaid
flowchart TB
    subgraph overload[過負荷保護]
        O1[Queue full 検出] --> O2{1秒間に 3回以上?}
        O2 -->|Yes| O3[Overload Mode ON<br/>2秒間]
        O3 --> O4[f0_method = none<br/>index_rate = 0.0]
        O4 --> O5[推論負荷を大幅削減]
        O2 -->|No| O6[通常品質維持]
        O5 -->|2秒後| O7[自動復帰<br/>Overload Mode OFF]
    end

    subgraph feedback[フィードバック検出]
        F1[出力履歴保存<br/>1秒分リングバッファ] --> F2[10チャンク毎に<br/>入力と相関チェック]
        F2 --> F3{corr > 0.3?}
        F3 -->|Yes| F4[警告表示<br/>フィードバック検出]
        F3 -->|No| F5[正常]
    end

    style overload fill:#fff3e0,stroke:#FF9800
    style feedback fill:#fce4ec,stroke:#E91E63
```

w-okada にはこれらの仕組みが存在しない。推論が遅い場合は sounddevice コールバック内でブロックが発生し、OS レベルでオーディオフレームがドロップされる。

---

## 10. エンドツーエンド レイテンシ構成

### 10.1 レイテンシ内訳

```mermaid
gantt
    title レイテンシ内訳 (典型的な 150ms チャンク設定)
    dateFormat X
    axisFormat %L ms

    section RCWX
    Input Accumulation    :a1, 0, 150
    Inference (async)     :a2, 150, 100
    Output Buffer         :a3, 250, 50

    section w-okada
    Input Capture         :b1, 0, 150
    Inference (sync)      :b2, 150, 120
    Output Queue          :b3, 270, 30
```

| 構成要素 | RCWX | w-okada |
|---|---|---|
| 入力バッファリング | chunk_sec (150ms) | block_frame (設定次第) |
| 推論時間 | 50-100ms (並列抽出) | 80-150ms (直列抽出) |
| 出力バッファ | RingOutputBuffer + prebuffer | outQueue |
| SOLA holdback | crossfade_sec (50ms) | crossfadeFrame (可変) |
| **合計** | **~300ms** | **~350ms** |

RCWX は並列特徴抽出と非同期推論により、同等のチャンクサイズでレイテンシが若干低い。

---

## 11. まとめ

### 11.1 設計哲学の対比

```mermaid
mindmap
    root((RVC Real-time<br/>Voice Changer))
        RCWX
            効率型
                少コンテキスト + 多段後処理
                StatefulResampler
                Feature Cache prepend
            XPU最適化
                固定サイズ HuBERT 入力
                カーネル再コンパイル回避
            自動化
                パラメータ自動導出
                過負荷自動退避
                フィードバック検出
        w-okada
            brute force型
                大量コンテキスト投入
                extraConvertSize
                品質を物量で担保
            汎用性
                8+ モデル対応
                Web UI
                マルチプラットフォーム
            手動調整
                chunk / extra 手動チューニング
                品質とレイテンシのトレードオフをユーザーに委任
```

### 11.2 各実装の強み

**RCWX が優れている点**:
- `StatefulResampler` によるチャンク境界のリサンプリング品質
- HuBERT + F0 並列抽出による推論高速化
- F0 後処理 (median + lowpass + octave flip suppress + slew limit) によるピッチ安定性
- 過負荷自動退避とフィードバック検出
- `target_len` による SOLA ドリフト防止
- XPU 向け固定サイズパディング最適化

**w-okada が優れている点**:
- `extraConvertSize` によるユーザー可変コンテキスト (品質チューニングの自由度)
- Pitch Protection (無声子音の index 変換抑制)
- 推論スキップによる計算節約 (silentThreshold)
- cos² + margin 方式の SOLA 窓関数
- Web UI によるリモート操作・複数クライアント対応
- 多モデルフレームワーク対応

### 11.3 RCWX に取り入れる価値のある要素

| 要素                      | 概要                                 | 難易度 |
| ----------------------- | ---------------------------------- | --- |
| **Pitch Protection**    | 無声子音で index features を元特徴量にブレンドバック | 低   |
| **Silent Threshold**    | 音量閾値以下で推論スキップ (計算節約)               | 低   |
| **cos² + margin 窓**     | SOLA 窓関数のフラット区間追加                  | 低   |
| **ユーザー可変コンテキスト**        | `extraConvertSize` 相当のパラメータ公開      | 中   |
| **Quality Repeat Mode** | HuBERT reflection padding の増量オプション | 中   |
