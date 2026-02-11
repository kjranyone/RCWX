# RCWX 環境構築ガイド

## 前提条件

### ハードウェア
- **GPU**: Intel Arc (A770, A750, B580, etc.)
- **Resizable BAR**: BIOS/UEFIで有効化推奨

### OS
- Windows 10/11 (64-bit)

### ドライバ
- Intel Arc GPU ドライバ（最新版推奨）
- https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html

---

## Step 1: uv のインストール

### PowerShell（管理者権限不要）
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

### または winget
```powershell
winget install astral-sh.uv
```

### 確認
```powershell
uv --version
```

---

## Step 2: プロジェクトのクローン

```powershell
git clone https://github.com/your-org/rcwx.git
cd rcwx
```

---

## Step 3: 依存関係のインストール

```powershell
# PyTorch XPU版を含む全依存関係をインストール
uv sync

# (推奨) 低レイテンシF0抽出を追加
uv sync --extra lowlatency
```

> `pyproject.toml` で PyTorch XPU インデックスが設定済みのため、`uv sync` だけで XPU 版がインストールされます。

---

## Step 4: 動作確認

### XPU が認識されているか確認
```powershell
python -c "import torch; print(f'XPU Available: {torch.xpu.is_available()}')"
```

**期待される出力:**
```
XPU Available: True
```

### GPU 情報の確認
```powershell
python -c "import torch; print(torch.xpu.get_device_name(0))"
```

**出力例:**
```
Intel(R) Arc(TM) A770 Graphics
```

### 簡単な演算テスト
```powershell
python -c "import torch; x = torch.randn(1000, 1000, device='xpu'); print(f'Tensor on XPU: {x.device}')"
```

---

## Step 5: torch.compile の確認（オプション）

```powershell
python -c "
import torch

@torch.compile
def test_fn(x):
    return x * 2 + 1

x = torch.randn(100, device='xpu')
y = test_fn(x)
print(f'torch.compile OK: {y.device}')
"
```

初回実行時はコンパイルに時間がかかります（数十秒〜数分）。

---

## トラブルシューティング

### `torch.xpu.is_available()` が False

1. **ドライバ確認**
   ```powershell
   # デバイスマネージャーでIntel Arcが認識されているか確認
   Get-PnpDevice | Where-Object { $_.FriendlyName -like "*Arc*" }
   ```

2. **PyTorch再インストール**
   ```powershell
   # uv.lock を再生成して再インストール
   del uv.lock
   uv sync
   ```

3. **Python バージョン確認**
   ```powershell
   python --version  # 3.11 または 3.12 であること
   ```

### CUDA関連エラーが出る

PyTorch XPU版では `cuda` ではなく `xpu` を使用します。
```python
# NG
tensor.to("cuda")

# OK
tensor.to("xpu")
```

### メモリ不足エラー

```python
# GPUメモリ使用量を確認
print(torch.xpu.memory_allocated() / 1024**3, "GB")

# キャッシュクリア
torch.xpu.empty_cache()
```

### torch.compile が遅い・エラー

- 初回はコンパイル時間がかかる（キャッシュされる）
- エラー時は `torch.compile` を無効化して動作確認
  ```python
  # torch.compile を使わない場合
  model = model  # torch.compile(model) の代わりに
  ```

---

## 開発環境（オプション）

### 開発用依存関係
```powershell
uv sync --extra dev
```

### Ruff（リンター）
```powershell
ruff check .
ruff format .
```

### テスト
```powershell
pytest
```

---

## 次のステップ

環境構築が完了したら、[CLAUDE.md](../CLAUDE.md) の Quick Start に従ってモデルをダウンロードしてください。

```powershell
# 必須モデル（HuBERT / RMVPE）を自動ダウンロード
uv run rcwx download
```

必要なモデルファイル:
- `hubert_base.pt` — `~/.cache/rcwx/models/hubert/` に自動配置
- `rmvpe.pt` — `~/.cache/rcwx/models/rmvpe/` に自動配置
- RVCv2モデル (`.pth`) — 任意のディレクトリ
