#Requires -Version 5.1
<#
.SYNOPSIS
    RCWX ランチャー / 環境チェック用の対話式スクリプト。

.DESCRIPTION
    uv 環境を確認し、対話メニューから GUI 起動・デバイス一覧・モデルダウンロード
    などを実行する。引数を渡した場合はメニューを介さず、そのまま `uv run rcwx`
    のサブコマンドへ転送する（配布・自動化向け）。

    ML デノイザ (Facebook denoiser, 任意依存 extra "ml-denoise", CC BY-NC 4.0) は
    既定では未インストール。有効化を対話（メニュー / 環境チェック）または -Denoise
    スイッチで選択できる。uv は extra 無しの実行で extra 依存を prune するため、
    有効時は全ての `uv run` を `--extra ml-denoise` 付きで実行する。

.PARAMETER Denoise
    ML デノイザ(denoiser)を有効化して起動する（uv run --extra ml-denoise ...）。

.EXAMPLE
    .\rcwx.ps1
    引数なし → 環境チェック後に対話メニューを表示。

.EXAMPLE
    .\rcwx.ps1 gui
    メニューを介さず GUI を直接起動。

.EXAMPLE
    .\rcwx.ps1 -Denoise gui
    ML デノイザを有効化した状態で GUI を起動。

.EXAMPLE
    .\rcwx.ps1 download --force
    渡した引数はそのまま `uv run rcwx` へ転送される。

.NOTES
    実行ポリシーで弾かれる場合:
        powershell -ExecutionPolicy Bypass -File .\rcwx.ps1
#>
[CmdletBinding()]
param(
    [switch]$Denoise,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RcwxArgs
)

$ErrorActionPreference = 'Stop'

# コンソール出力を UTF-8 に（日本語の文字化け対策）
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}

# スクリプトのある場所（= リポジトリルート）へ移動
Set-Location -Path $PSScriptRoot

# ML デノイザを有効にするか（-Denoise 指定 / 既存インストール検出 / 対話で決定）
$script:UseDenoiser = [bool]$Denoise

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

function Test-DenoiserInstalled {
    <# 現在の venv に denoiser が入っているか（uv pip list で確認）。 #>
    $list = & uv pip list
    return [bool]($list | Select-String -Pattern '(?i)^denoiser\s' -Quiet)
}

function Get-ExtraArgs {
    <# ML デノイザ有効時は uv の extra 引数を返す。 #>
    if ($script:UseDenoiser) { return @('--extra', 'ml-denoise') }
    return @()
}

function Invoke-Rcwx {
    <# uv run [--extra ml-denoise] rcwx <args> を実行。 #>
    param([string[]]$CmdArgs)
    $extra = Get-ExtraArgs
    & uv run @extra rcwx @CmdArgs
}

function Enable-Denoiser {
    <# ML デノイザ(denoiser)を有効化し、依存をインストールする。 #>
    Write-Host ""
    Write-Host "ML Denoiser (Facebook denoiser, CC BY-NC 4.0) をインストールします..." -ForegroundColor Cyan
    Write-Host "  実行: uv sync --extra ml-denoise"
    & uv sync --extra ml-denoise
    if ($LASTEXITCODE -eq 0) {
        $script:UseDenoiser = $true
        Write-Host "[OK] 有効化しました。以降の起動は --extra ml-denoise 付きで実行します。" -ForegroundColor Green
    } else {
        Write-Host "[NG] インストールに失敗しました。ネットワーク等を確認してください。" -ForegroundColor Red
    }
}

function Invoke-EnvCheck {
    <# uv / PyTorch(XPU) / 必須モデル / ML デノイザの状態を確認。 #>
    Write-Host ""
    Write-Host "=============== 環境チェック ===============" -ForegroundColor Cyan

    $uvVer = & uv --version
    Write-Host "[OK] uv: $uvVer" -ForegroundColor Green

    # 既存の denoiser を検出したら有効扱い（base sync による prune を防ぐ）
    if (-not $script:UseDenoiser -and (Test-DenoiserInstalled)) {
        $script:UseDenoiser = $true
        Write-Host "[i] 既存の denoiser を検出。ML デノイザを有効として扱います。"
    }

    # 依存関係の同期（denoiser 有効時は extra 付きで prune を回避）
    $extra = Get-ExtraArgs
    Write-Host "-- 依存関係の同期 (uv sync $($extra -join ' ')) --"
    & uv sync @extra
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] 依存関係は最新です。" -ForegroundColor Green
    } else {
        Write-Host "[!] uv sync が失敗しました。ネットワーク / pyproject.toml を確認してください。" -ForegroundColor Yellow
    }

    # PyTorch / XPU
    Write-Host "-- PyTorch / XPU --"
    & uv run @extra python -c "import torch; print('     torch', torch.__version__, '| XPU available:', torch.xpu.is_available())"

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

    # ML Denoiser (任意)
    Write-Host "-- ML Denoiser (任意, CC BY-NC 4.0) --"
    if ($script:UseDenoiser) {
        Write-Host "[OK] 有効（--extra ml-denoise 付きで起動）。" -ForegroundColor Green
    } else {
        Write-Host "[!] 無効。config の denoise.method='ml' のままだと実行時に ImportError になります。" -ForegroundColor Yellow
        Write-Host "     回避策: GUI でデノイズ方式を spectral / off にする、または今すぐ有効化。"
        $ans = Read-Host "     ML デノイザを今すぐ有効化(インストール)しますか? (y/N)"
        if ($ans -match '^(y|Y|yes|YES)$') { Enable-Denoiser }
    }
    Write-Host "===========================================" -ForegroundColor Cyan
}

function Show-Menu {
    if ($script:UseDenoiser) { $dn = "有効" } else { $dn = "無効" }
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
    Write-Host "  9) ML デノイザを有効化 (denoiser インストール)  [現在: $dn]"
    Write-Host "  0) 終了"
    Write-Host "=========================================" -ForegroundColor Cyan
}

# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

if (-not (Test-Uv)) { exit 1 }

# 既存の denoiser を検出したら有効扱い（-Denoise 未指定でも prune を防ぐ）
if (-not $script:UseDenoiser -and (Test-DenoiserInstalled)) {
    $script:UseDenoiser = $true
}

# 引数があればメニューを介さず rcwx へ直接転送（extra は現在の設定に従う）
if ($RcwxArgs -and $RcwxArgs.Count -gt 0) {
    $extra = Get-ExtraArgs
    & uv run @extra rcwx @RcwxArgs
    exit $LASTEXITCODE
}

# 初回に環境チェック
Invoke-EnvCheck

$running = $true
while ($running) {
    Show-Menu
    $choice = Read-Host "選択"
    switch ($choice) {
        '1' { Invoke-Rcwx @('gui') }
        '2' { Invoke-Rcwx @('devices') }
        '3' { Invoke-Rcwx @('download') }
        '4' { Invoke-Rcwx @('diagnose') }
        '5' { Invoke-Rcwx @('logs', '--tail', '50') }
        '6' { Invoke-EnvCheck }
        '7' { $e = Get-ExtraArgs; & uv sync @e }
        '8' {
            $custom = Read-Host "rcwx への引数 (例: run in.wav model.pth -o out.wav -p 5)"
            if ($custom) {
                $parts = $custom -split '\s+' | Where-Object { $_ -ne '' }
                Invoke-Rcwx $parts
            }
        }
        '9' { Enable-Denoiser }
        '0' { $running = $false }
        default { Write-Host "無効な選択です: $choice" -ForegroundColor Yellow }
    }
}

Write-Host "終了しました。"
