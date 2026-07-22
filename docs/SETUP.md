# RCWX 環境構築ガイド

本プロジェクトは **Intel Arc (XPU)** 向けです。CUDA / NVIDIA 環境は未検証のためサポート対象外です。

## 前提条件

### ハードウェア
- **GPU**: Intel Arc (A770, A750, B580, B570, etc.)
- **Resizable BAR**: BIOS/UEFI で有効化推奨

### OS
- Windows 10/11 (64-bit)

### ドライバ
- Intel Arc GPU ドライバ（最新版推奨）
- https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html

### ソフトウェア
- [uv](https://docs.astral.sh/uv/) パッケージマネージャ
- Python 3.11 or 3.12（`uv` がプロジェクト用に自動で用意）

---

## 推奨: `rcwx.ps1` による自動セットアップ

同梱の `rcwx.ps1` が依存関係の同期・XPU 確認・モデル有無のチェック・対話メニューを一括で行います。

### 1. リポジトリを取得

```powershell
git clone https://github.com/grand2-products/rcwx.git
cd rcwx
```

### 2. uv をインストール（未導入の場合）

```powershell
# PowerShell（管理者権限不要）
irm https://astral.sh/uv/install.ps1 | iex

# または winget
winget install astral-sh.uv

# 確認
uv --version
```

### 3. ランチャーを起動

```powershell
.\rcwx.ps1
```

実行ポリシーで弾かれる場合:

```powershell
powershell -ExecutionPolicy Bypass -File .\rcwx.ps1
```

初回起動時に次を自動実行します。

1. **`uv sync`** — PyTorch XPU 版を含む依存関係のインストール / 同期  
   （`pyproject.toml` に XPU インデックスが設定済み）
2. **PyTorch / XPU の確認** — `torch` バージョンと `torch.xpu.is_available()`
3. **必須モデルの有無** — HuBERT / RMVPE（未取得ならメニューから DL）
4. **ML デノイザ（任意）** — 未導入なら対話で有効化を選択可能

その後、対話メニューが表示されます。

| 番号 | 操作 |
|------|------|
| 1 | GUI を起動 |
| 2 | オーディオデバイス一覧 |
| 3 | 必須モデル (HuBERT / RMVPE) をダウンロード |
| 4 | フィードバック診断（XPU / Accelerator Graph 含む） |
| 5 | 最新ログを表示 (tail 50) |
| 6 | 環境チェックを再実行 |
| 7 | 依存関係を同期 (`uv sync`) |
| 8 | 任意の `rcwx` コマンドを入力 |
| 9 | ML デノイザを有効化 (`--extra ml-denoise`) |
| 0 | 終了 |

### 直接起動（メニューを介さない）

```powershell
.\rcwx.ps1 gui                # GUI
.\rcwx.ps1 -Denoise gui       # ML デノイザ有効化で GUI
.\rcwx.ps1 download           # 必須モデル DL
.\rcwx.ps1 diagnose           # 診断
.\rcwx.ps1 download --force   # 引数はそのまま rcwx へ転送
```

> ML デノイザ（任意依存 `ml-denoise` extra、CC BY-NC 4.0）は既定で未インストールです。  
> `uv run` / `uv sync` を extra なしで実行すると prune されるため、ランチャーは有効時に全ての `uv run` を `--extra ml-denoise` 付きにします。

---

## 手動セットアップ（開発者向け）

ランチャーを使わずに手順を踏む場合の参考です。

```powershell
# 依存関係（PyTorch XPU / torchfcpe / swift-f0 を含む）
uv sync

# XPU 確認
uv run python -c "import torch; print(torch.__version__); print(f'XPU Available: {torch.xpu.is_available()}')"

# 必須モデル (HuBERT / RMVPE)
uv run rcwx download

# 診断（XPU / Accelerator Graph など）
uv run rcwx diagnose

# GUI
uv run rcwx

# (オプション) ML Denoiser
uv sync --extra ml-denoise
```

**期待される XPU 確認出力例:**

```
2.13.0+xpu
XPU Available: True
```

GPU 名の確認:

```powershell
uv run python -c "import torch; print(torch.xpu.get_device_name(0))"
# 例: Intel(R) Arc(TM) A770 Graphics
```

---

## XPU Accelerator Graph

RCWX は PyTorch 2.13 の Accelerator Graph を **XPU** で自動使用します（GUI 設定不要）。対象:

- HuBERT 特徴抽出
- 定常状態の RVC Synthesizer
- Aggressive モードの XPU IVF（対応 FAISS index 時）

```powershell
.\rcwx.ps1 diagnose
# または
uv run rcwx diagnose
```

次の行が表示されれば利用可能です。

```text
[OK] XPU Accelerator Graph: True
```

変換開始前のウォームアップで固定 shape の Graph を capture します。音声履歴とリサンプラ状態は実ストリーム開始前にリセットし、capture 済み Graph は保持します。capture 失敗時は eager 推論へ自動フォールバックします。`use_compile=true` の Synthesizer は Graph を併用しません。
---

## トラブルシューティング

### `torch.xpu.is_available()` が False

1. **ドライバ確認**
   ```powershell
   Get-PnpDevice | Where-Object { $_.FriendlyName -like "*Arc*" }
   ```

2. **PyTorch 再インストール**
   ```powershell
   del uv.lock
   uv sync
   # または .\rcwx.ps1 でメニュー 7) 依存関係を同期
   ```

3. **Python バージョン確認**  
   プロジェクトは Python 3.11 / 3.12 を要求します（`uv` が解決します）。

### メモリ不足エラー

```python
print(torch.xpu.memory_allocated() / 1024**3, "GB")
torch.xpu.empty_cache()
```

### torch.compile が遅い・エラー

- 初回はコンパイルに時間がかかります（キャッシュされます）
- 問題切り分け時は GUI / 設定で `use_compile` を無効化して確認してください

### Accelerator Graph を無効化する

```powershell
$env:RCWX_ACCELERATOR_GRAPH = "0"
.\rcwx.ps1 gui
# または: uv run rcwx
```

自動判定へ戻す:

```powershell
Remove-Item Env:RCWX_ACCELERATOR_GRAPH
```

### ML デノイザ関連

- config の `denoise.method='ml'` のまま denoiser 未導入だと実行時エラーになります
- 回避: `.\rcwx.ps1` のメニュー 9) で有効化、GUI で spectral / off に変更、または `.\rcwx.ps1 -Denoise gui`

---

## 開発環境（オプション）

```powershell
# 開発用依存関係
uv sync --extra dev

# Lint / Format
ruff check .
ruff format .

# テスト
uv run pytest
```

---

## 配置されるモデル

| ファイル | 既定パス |
|----------|----------|
| `hubert_base.pt` | `~/.cache/rcwx/models/hubert/` |
| `rmvpe.pt` | `~/.cache/rcwx/models/rmvpe/` |
| RVC `.pth` | 任意（GUI の RVC モデルディレクトリで指定） |

必須モデルはメニュー 3) または `.\rcwx.ps1 download` / `uv run rcwx download` で取得できます。

---

## 関連ドキュメント

| 文書 | 内容 |
|------|------|
| [README.md](../README.md) | ユーザー向け概要・Latency・設定 |
| [AGENTS.md](../AGENTS.md) / [CLAUDE.md](../CLAUDE.md) | 開発者向け現行アーキテクチャ |
| [docs/README.md](README.md) | docs 配下の索引 |
