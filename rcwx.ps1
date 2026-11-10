#Requires -Version 5.1
<#
.SYNOPSIS
    RCWX ランチャー (Rust 単一バイナリ版)。

.DESCRIPTION
    Rust 実装 (rcwx-gui / rcwx-cli) を起動する。このランチャーと製品の実行経路に
    Python 依存は一切ない。

    - cargo があれば cargo build --release (インクリメンタル) を実行してから起動
    - 無ければ既存バイナリ (target/release/*.exe) を直接使用
    - モデルバンドル (~/.cache/rcwx/models/rust) を確認し、未変換なら
      開発ランチャー (.\dev.ps1 golden) での生成を案内

    開発・検証 (モデルバンドル生成、旧 Python 実装とのパリティ比較) は
    dev.ps1 を使用。

.EXAMPLE
    .\rcwx.ps1
    引数なし → 環境チェック後に対話メニュー。

.EXAMPLE
    .\rcwx.ps1 gui
    GUI を直接起動。

.EXAMPLE
    .\rcwx.ps1 rt --seconds 10 --chunk 0.02
    リアルタイム変換を 10 秒だけ実行 (実機確認)。

.EXAMPLE
    .\rcwx.ps1 run input.wav -o out.wav --pitch 5
    オフライン変換 (-o 等はそのまま rcwx-cli へ転送)。

.NOTES
    実行ポリシーで弾かれる場合:
        powershell -ExecutionPolicy Bypass -File .\rcwx.ps1
#>

$ErrorActionPreference = 'Stop'

# 引数は param で束ねず $args で生受け取り (-o 等が共通パラメータと衝突するのを避ける)
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
Set-Location -Path $PSScriptRoot

# ---------------------------------------------------------------------------
# パス
# ---------------------------------------------------------------------------
$ModelsRust = Join-Path $HOME '.cache\rcwx\models\rust'

# 配布アーカイブ (exe がスクリプト横) か開発ワークスペース (target\release) か
$DistLayout = Test-Path (Join-Path $PSScriptRoot 'rcwx-cli.exe')
$CliExe     = if ($DistLayout) { Join-Path $PSScriptRoot 'rcwx-cli.exe' }
              else { Join-Path $PSScriptRoot 'target\release\rcwx-cli.exe' }
$GuiExe     = if ($DistLayout) { Join-Path $PSScriptRoot 'rcwx-gui.exe' }
              else { Join-Path $PSScriptRoot 'target\release\rcwx-gui.exe' }

# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------
function Test-Cargo {
    return [bool](Get-Command cargo -ErrorAction SilentlyContinue)
}

function Invoke-RustBuild {
    if ($DistLayout) {
        # 配布アーカイブ: ビルド不要 (Cargo.toml も無いので cargo build は不可)
        return $true
    }
    if (Test-Cargo) {
        # --quiet だと初回の GPUI release 最適化が無言のまま数分続く。
        # 製品バイナリだけを対象にし、進捗は cargo の表示に任せる。
        & cargo build --release -p rcwx-cli -p rcwx-gui
        if ($LASTEXITCODE -ne 0) {
            Write-Host '[NG] cargo build に失敗。' -ForegroundColor Red
            return $false
        }
        return $true
    }
    if (-not (Test-Path $CliExe)) {
        Write-Host "[NG] バイナリがありません: $CliExe" -ForegroundColor Red
        Write-Host '     rustup を導入 (https://rustup.rs/) して再実行するか、'
        Write-Host '     ビルド済みバイナリを target/release/ に配置してください。'
        return $false
    }
    return $true
}

function Get-MissingWeights {
    $need = @(
        'hubert.safetensors',
        'rmvpe_folded.safetensors',
        'swiftf0.safetensors',
        'synth.safetensors',
        'synth.json'
    )
    $missing = @()
    foreach ($w in $need) {
        if (-not (Test-Path (Join-Path $ModelsRust $w))) { $missing += $w }
    }
    return $missing
}

function Invoke-EnvCheck {
    Write-Host ''
    Write-Host '========= 環境チェック (Rust) =========' -ForegroundColor Cyan

    if (Test-Cargo) {
        $rustVer = & rustc --version
        Write-Host "[OK] $rustVer" -ForegroundColor Green
        Write-Host '-- ビルド (cargo build --release) --'
        Write-Host '   初回は GPUI の release 最適化で数分かかります。進捗は下に出ます。' -ForegroundColor DarkGray
        if (-not (Invoke-RustBuild)) { return $false }
    } else {
        Write-Host '[i] cargo なし — 既存バイナリを使用。' -ForegroundColor DarkGray
        if (-not (Test-Path $CliExe)) {
            Write-Host "[NG] バイナリがありません: $CliExe" -ForegroundColor Red
            return $false
        }
    }
    Write-Host "[OK] rcwx-cli: $CliExe" -ForegroundColor Green

    Write-Host "-- モデルバンドル ($ModelsRust) --"
    $mw = Get-MissingWeights
    if ($mw.Count -eq 0) {
        Write-Host '[OK] モデルバンドル配置済み。' -ForegroundColor Green
    } else {
        Write-Host "[!] 不足: $($mw -join ', ')" -ForegroundColor Yellow
        Write-Host '     モデルバンドルが未生成です。開発ランチャーで生成してください:'
        Write-Host '       .\dev.ps1 golden'
        Write-Host '     (生成は開発環境での一回限りの作業です)'
    }
    Write-Host '=======================================' -ForegroundColor Cyan
    return $true
}

function Invoke-Cli {
    param([string[]]$CmdArgs)
    if (-not (Invoke-RustBuild)) { return }
    & $CliExe @CmdArgs
}

# ---------------------------------------------------------------------------
# 引数転送 (メニューを介さない)
# ---------------------------------------------------------------------------
$forward = @()
foreach ($a in @($args)) { $forward += "$a" }

if ($forward -and $forward.Count -gt 0) {
    # gui だけは rcwx-gui を起動
    if ($forward[0] -ieq 'gui') {
        if (-not (Invoke-RustBuild)) { exit 1 }
        if (Test-Path $GuiExe) {
            & $GuiExe
            exit $LASTEXITCODE
        }
        Write-Host '[i] rcwx-gui.exe が無いため cargo run で起動...' -ForegroundColor DarkGray
        & cargo run --release -p rcwx-gui
        exit $LASTEXITCODE
    }
    Invoke-Cli $forward
    exit $LASTEXITCODE
}

# ---------------------------------------------------------------------------
# 対話メニュー
# ---------------------------------------------------------------------------
if (-not (Invoke-EnvCheck)) { exit 1 }

$running = $true
while ($running) {
    Write-Host ''
    Write-Host '==== RCWX (Rust 単一バイナリ) ====' -ForegroundColor Cyan
    Write-Host '  1) GUI を起動'
    Write-Host '  2) オーディオデバイス一覧'
    Write-Host '  3) リアルタイム変換 (rt) — 引数入力'
    Write-Host '  4) オフライン変換 (run) — 引数入力'
    Write-Host '  5) モデル情報 (info)'
    Write-Host '  6) 最新ログを表示 (tail 50)'
    Write-Host '  7) 環境チェックを再実行'
    Write-Host '  0) 終了'
    Write-Host '==================================' -ForegroundColor Cyan
    $choice = Read-Host '選択'
    switch ($choice) {
        '1' {
            if (-not (Invoke-RustBuild)) { break }
            if (Test-Path $GuiExe) { & $GuiExe }
            else { & cargo run --release -p rcwx-gui }
        }
        '2' { Invoke-Cli @('devices') }
        '3' {
            $a = Read-Host 'rt への引数 (例: --seconds 10 --chunk 0.02 / 空で既定)'
            $parts = @('rt')
            if ($a) { $parts += ($a -split '\s+' | Where-Object { $_ -ne '' }) }
            Invoke-Cli $parts
        }
        '4' {
            $c = Read-Host 'run への引数 (例: input.wav -o out.wav --pitch 5)'
            if ($c) {
                $parts = @('run') + ($c -split '\s+' | Where-Object { $_ -ne '' })
                Invoke-Cli $parts
            }
        }
        '5' {
            $m = Read-Host 'モデル .pth のパス (空で model/kurumi/kurumi.pth)'
            if (-not $m) { $m = 'model/kurumi/kurumi.pth' }
            Invoke-Cli @('info', $m)
        }
        '6' { Invoke-Cli @('logs', '--tail', '50') }
        '7' { Invoke-EnvCheck | Out-Null }
        '0' { $running = $false }
        default { Write-Host "無効な選択です: $choice" -ForegroundColor Yellow }
    }
}
Write-Host '終了しました。'
