#Requires -Version 5.1
<#
.SYNOPSIS
    RCWX ランチャー / 環境チェック用の対話式スクリプト。

.DESCRIPTION
    uv 環境を確認し、対話メニューから GUI 起動・デバイス一覧・モデルダウンロード
    などを実行する。引数を渡した場合はメニューを介さず、そのまま `uv run rcwx`
    のサブコマンドへ転送する（配布・自動化向け）。

.EXAMPLE
    .\rcwx.ps1
    引数なし → 環境チェック後に対話メニューを表示。

.EXAMPLE
    .\rcwx.ps1 gui
    メニューを介さず GUI を直接起動。

.EXAMPLE
    .\rcwx.ps1 download --force
    渡した引数はそのまま `uv run rcwx` へ転送される。

.NOTES
    実行ポリシーで弾かれる場合:
        powershell -ExecutionPolicy Bypass -File .\rcwx.ps1
#>
[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RcwxArgs
)

$ErrorActionPreference = 'Stop'

# コンソール出力を UTF-8 に（日本語の文字化け対策）
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}

# スクリプトのある場所（= リポジトリルート）へ移動
Set-Location -Path $PSScriptRoot

# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

function Test-Uv {
    <# uv の有無を確認。無ければ導入方法を案内して $false を返す。 #>
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        return $true
    }
    Write-Host "[NG] uv が見つかりません。" -ForegroundColor Red
    Write-Host "     インストール手順: https://docs.astral.sh/uv/getting-started/installation/"
    Write-Host '     PowerShell 例:    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"'
    return $false
}

function Invoke-EnvCheck {
    <# uv / PyTorch(XPU) / ML Denoiser / 必須モデルの配置状況を確認。 #>
    Write-Host ""
    Write-Host "=============== 環境チェック ===============" -ForegroundColor Cyan

    $uvVer = (& uv --version)
    Write-Host "[OK] uv: $uvVer" -ForegroundColor Green

    # 依存関係の同期（uv run は lock との差分を自動同期するが、明示的に一度走らせる）
    Write-Host "-- 依存関係の同期 (uv sync) --"
    & uv sync
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] 依存関係は最新です。" -ForegroundColor Green
    } else {
        Write-Host "[!] uv sync が失敗しました。ネットワーク / pyproject.toml を確認してください。" -ForegroundColor Yellow
    }

    # PyTorch / XPU
    Write-Host "-- PyTorch / XPU --"
    & uv run python -c "import torch; print('     torch', torch.__version__, '| XPU available:', torch.xpu.is_available())"

    # ML Denoiser
    Write-Host "-- ML Denoiser --"
    & uv run python -c "from rcwx.audio.denoise import is_ml_denoiser_available as f; print('     ML denoiser available:', f())"

    # 必須モデル (既定の配置先を確認)
    Write-Host "-- 必須モデル (HuBERT / RMVPE) --"
    $modelsDir = Join-Path $HOME ".cache\rcwx\models"
    $hubert = Join-Path $modelsDir "hubert\hubert_base.pt"
    $rmvpe  = Join-Path $modelsDir "rmvpe\rmvpe.pt"
    if ((Test-Path $hubert) -and (Test-Path $rmvpe)) {
        Write-Host "[OK] 配置済み: $modelsDir" -ForegroundColor Green
    } else {
        Write-Host "[!] 未取得。メニュー 3) または 'uv run rcwx download' を実行してください。" -ForegroundColor Yellow
        Write-Host "     (既定の探索先: $modelsDir)"
    }
    Write-Host "===========================================" -ForegroundColor Cyan
}

function Show-Menu {
    Write-Host ""
    Write-Host "================== RCWX ==================" -ForegroundColor Cyan
    Write-Host "  1) GUI を起動"
    Write-Host "  2) オーディオデバイス一覧"
    Write-Host "  3) 必須モデルをダウンロード"
    Write-Host "  4) フィードバック診断"
    Write-Host "  5) 最新ログを表示 (tail 50)"
    Write-Host "  6) 環境チェックを再実行"
    Write-Host "  7) 依存関係を同期 (uv sync)"
    Write-Host "  8) 任意の rcwx コマンドを入力"
    Write-Host "  0) 終了"
    Write-Host "=========================================" -ForegroundColor Cyan
}

# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

if (-not (Test-Uv)) { exit 1 }

# 引数があればメニューを介さず rcwx へ直接転送
if ($RcwxArgs -and $RcwxArgs.Count -gt 0) {
    & uv run rcwx @RcwxArgs
    exit $LASTEXITCODE
}

# 初回に環境チェック
Invoke-EnvCheck

$running = $true
while ($running) {
    Show-Menu
    $choice = Read-Host "選択"
    switch ($choice) {
        '1' { & uv run rcwx gui }
        '2' { & uv run rcwx devices }
        '3' { & uv run rcwx download }
        '4' { & uv run rcwx diagnose }
        '5' { & uv run rcwx logs --tail 50 }
        '6' { Invoke-EnvCheck }
        '7' { & uv sync }
        '8' {
            $custom = Read-Host "rcwx への引数 (例: run in.wav model.pth -o out.wav -p 5)"
            if ($custom) {
                $parts = $custom -split '\s+' | Where-Object { $_ -ne '' }
                & uv run rcwx @parts
            }
        }
        '0' { $running = $false }
        default { Write-Host "無効な選択です: $choice" -ForegroundColor Yellow }
    }
}

Write-Host "終了しました。"
