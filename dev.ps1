#Requires -Version 5.1
<#
.SYNOPSIS
    RCWX Rust 単一バイナリ (crates/*) の開発ランチャー。

.DESCRIPTION
    Rust 側 (rcwx-nn / rcwx-dsp / rcwx-rt / rcwx-cli / rcwx-gui) のビルド・
    テスト・ベンチ・パリティ検証・実行をまとめた対話/直両用スクリプト。

    - 重み変換 (tools/convert_*.py) と golden 生成は Python (uv) を使用
    - テスト・ベンチは golden / bench30s.wav が無い場合に案内
    - 引数を渡した場合はメニューを介さず cargo / rcwx-cli へ転送

    主なサブコマンド:
        check    環境チェック (cargo / 重み / golden)
        build    cargo build --release
        test     パリティテスト一式 (golden 必要)
        golden   重み変換 + golden 再生成 (uv 必要・数分)
        bench    30 秒持続ベンチ (batch 1 / 4)
        e2e      オフライン E2E を Python 版と corr 比較 (uv + モデル必要)
        rt ...   リアルタイム CLI へ転送 (例: rt --seconds 10)
        run ...  オフライン変換へ転送
        gui      GPUI GUI を起動
        fmt      cargo fmt
        clippy   cargo clippy
        clean    cargo clean

.EXAMPLE
    .\dev.ps1
    引数なし → 環境チェック後に対話メニュー。

.EXAMPLE
    .\dev.ps1 bench
    持続ベンチ (batch 1 と 4) を実行。

.EXAMPLE
    .\dev.ps1 rt --seconds 10 --chunk 0.02
    実デバイスで 10 秒の動作確認。

.NOTES
    実行ポリシーで弾かれる場合:
        powershell -ExecutionPolicy Bypass -File .\dev.ps1
#>
# 引数は param で束ねず $args で生受け取り (-o 等が共通パラメータと
# 衝突するのを避ける)。先頭要素をサブコマンド、残りを転送引数として扱う。
$ErrorActionPreference = 'Stop'

try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
Set-Location -Path $PSScriptRoot

# ---------------------------------------------------------------------------
# パス定数
# ---------------------------------------------------------------------------
$ModelsRust = Join-Path $HOME ".cache\rcwx\models\rust"
$GoldenDir  = Join-Path $PSScriptRoot "target\golden"
$BenchWav   = Join-Path $GoldenDir "bench30s.wav"
$ParityWav  = Join-Path $GoldenDir "test_input.wav"
$CliExe     = Join-Path $PSScriptRoot "target\release\rcwx-cli.exe"

# テストが必要とする golden (tools/*.py が生成)。
# モデル golden は重みと同じ場所へ、DSP/F0 golden は target\golden へ出力される。
$GoldenInWeights = @(
    'hubert_golden.safetensors',
    'synth_golden.safetensors',
    'rmvpe_golden.safetensors'
)
$GoldenInTarget = @(
    'swiftf0_golden.npz',
    'dsp_golden.npz',
    'f0_ref.safetensors'
)
# ivf_golden.npz はモデル (.index) がある場合のみ

# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------
function Test-Cargo {
    if (Get-Command cargo -ErrorAction SilentlyContinue) { return $true }
    Write-Host "[NG] cargo が見つかりません。" -ForegroundColor Red
    Write-Host "     rustup を導入: https://rustup.rs/"
    return $false
}

function Test-Uv {
    if (Get-Command uv -ErrorAction SilentlyContinue) { return $true }
    Write-Host "[NG] uv が見つかりません (golden 生成・e2e 比較に必要)。" -ForegroundColor Red
    Write-Host "     powershell -c `"irm https://astral.sh/uv/install.ps1 | iex`""
    return $false
}

function Invoke-CargoBuild {
    Write-Host "-- cargo build --release --workspace --" -ForegroundColor Cyan
    & cargo build --release --workspace
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[NG] ビルド失敗。" -ForegroundColor Red
        return $false
    }
    Write-Host "[OK] ビルド完了。" -ForegroundColor Green
    return $true
}

function Get-MissingGoldens {
    $missing = @()
    foreach ($g in $GoldenInWeights) {
        if (-not (Test-Path (Join-Path $ModelsRust $g))) { $missing += $g }
    }
    foreach ($g in $GoldenInTarget) {
        if (-not (Test-Path (Join-Path $GoldenDir $g))) { $missing += $g }
    }
    return $missing
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

function Ensure-BenchWav {
    <# 30 秒ベンチ入力がなければ parity wav 20 連結で生成 (uv 必要)。 #>
    if (Test-Path $BenchWav) { return $true }
    if (-not (Test-Path $ParityWav)) {
        Write-Host "[NG] $ParityWav がありません。先に '.\dev.ps1 golden' を実行してください。" -ForegroundColor Red
        return $false
    }
    if (-not (Test-Uv)) { return $false }
    Write-Host "-- bench30s.wav を生成 --"
    & uv run --no-sync python -c "import numpy as np; from scipy.io import wavfile; _, a = wavfile.read(r'$ParityWav'); a = a.astype(np.float32)/32767.0; rng = np.random.RandomState(1); out = np.concatenate([a*(0.8+0.2*rng.rand()) for _ in range(20)]); wavfile.write(r'$BenchWav', 16000, (out*32767).astype(np.int16))"
    if ($LASTEXITCODE -eq 0 -and (Test-Path $BenchWav)) {
        Write-Host "[OK] 生成しました: $BenchWav" -ForegroundColor Green
        return $true
    }
    Write-Host "[NG] 生成に失敗。" -ForegroundColor Red
    return $false
}

# ---------------------------------------------------------------------------
# サブコマンド
# ---------------------------------------------------------------------------
function Invoke-Check {
    Write-Host ""
    Write-Host "=========== 環境チェック (Rust) ===========" -ForegroundColor Cyan

    if (-not (Test-Cargo)) { return }
    $rustVer = & rustc --version
    Write-Host "[OK] $rustVer" -ForegroundColor Green

    # 重み (変換済みバンドル)
    Write-Host "-- 変換済み重み ($ModelsRust) --"
    $mw = Get-MissingWeights
    if ($mw.Count -eq 0) {
        Write-Host "[OK] 全重みが配置済み。" -ForegroundColor Green
    } else {
        Write-Host "[!] 不足: $($mw -join ', ')" -ForegroundColor Yellow
        Write-Host "     '.\dev.ps1 golden' で変換してください (uv + 元モデル必要)。"
    }

    # golden (パリティテスト基準)
    Write-Host "-- テスト用 golden ($GoldenDir) --"
    $mg = Get-MissingGoldens
    if ($mg.Count -eq 0) {
        Write-Host "[OK] 全 golden が配置済み。" -ForegroundColor Green
    } else {
        Write-Host "[!] 不足: $($mg -join ', ')" -ForegroundColor Yellow
        Write-Host "     '.\dev.ps1 golden' で生成してください。"
    }

    # IVF (任意: モデル .index がある場合)
    $ivf = Join-Path $ModelsRust 'ivf.safetensors'
    if (Test-Path $ivf) {
        Write-Host "[OK] ivf.safetensors あり (index 検証有効)。" -ForegroundColor Green
    } else {
        Write-Host "[i] ivf.safetensors なし (index 系テストはスキップ)。" -ForegroundColor DarkGray
    }

    # ベンチ入力
    if (Test-Path $BenchWav) {
        Write-Host "[OK] bench30s.wav あり。" -ForegroundColor Green
    } else {
        Write-Host "[i] bench30s.wav なし (bench 実行時に自動生成)。" -ForegroundColor DarkGray
    }
    Write-Host "==========================================" -ForegroundColor Cyan
}

function Invoke-Golden {
    <# 重み変換 + golden 生成 (数分・GPU/CPU それなりに使う)。 #>
    if (-not (Test-Uv)) { return }
    Write-Host ""
    Write-Host "-- 重み変換 + golden 生成 (tools/*.py) --" -ForegroundColor Cyan
    $steps = @(
        @{ Name = 'HuBERT 変換';       Cmd = @('convert_weights.py') },
        @{ Name = 'Synthesizer 変換';  Cmd = @('convert_synth.py') },
        @{ Name = 'RMVPE 変換 (BN fold)'; Cmd = @('convert_rmvpe.py') },
        @{ Name = 'SwiftF0 変換';      Cmd = @('convert_swiftf0.py') },
        @{ Name = 'F0 golden';         Cmd = @('dump_f0_ref.py') },
        @{ Name = 'DSP golden';        Cmd = @('dump_dsp_golden.py') }
    )
    $indexFile = Join-Path $PSScriptRoot 'model\kurumi\kurumi.index'
    if (Test-Path $indexFile) {
        $steps += @{ Name = 'IVF 変換 (kurumi.index)'; Cmd = @('convert_ivf.py') }
    } else {
        Write-Host "[i] model/kurumi/kurumi.index が無いため IVF 変換はスキップ。" -ForegroundColor DarkGray
    }
    $failed = $false
    foreach ($s in $steps) {
        Write-Host "-- $($s.Name): tools\$($s.Cmd[0]) --"
        & uv run --no-sync python (Join-Path 'tools' $s.Cmd[0])
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[NG] $($s.Name) に失敗。" -ForegroundColor Red
            $failed = $true
            break
        }
    }
    if (-not $failed) {
        Write-Host "[OK] golden 生成完了。" -ForegroundColor Green
        Invoke-Check
    }
}

function Invoke-Test {
    if (-not (Test-Cargo)) { return }
    $mg = Get-MissingGoldens
    if ($mg.Count -gt 0) {
        Write-Host "[!] golden 不足: $($mg -join ', ')" -ForegroundColor Yellow
        Write-Host "     '.\dev.ps1 golden' を先に実行してください (テストは skip で通ります)。"
    }
    Write-Host "-- cargo test --release (rcwx-nn / rcwx-dsp) --" -ForegroundColor Cyan
    & cargo test --release -p rcwx-nn -p rcwx-dsp
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] 全テスト合格。" -ForegroundColor Green
    } else {
        Write-Host "[NG] テスト失敗。" -ForegroundColor Red
    }
}

function Invoke-Bench {
    param([string[]]$ExtraArgs)
    if (-not (Test-Cargo)) { return }
    if (-not (Invoke-CargoBuild)) { return }
    if (-not (Ensure-BenchWav)) { return }
    $chunk = '0.02'
    # --chunk <v> が渡されていればそれを使う
    for ($i = 0; $i -lt $ExtraArgs.Count; $i++) {
        if ($ExtraArgs[$i] -eq '--chunk' -and $i + 1 -lt $ExtraArgs.Count) {
            $chunk = $ExtraArgs[$i + 1]
        }
    }
    $out = Join-Path $GoldenDir 'bench_out.wav'
    foreach ($b in @(1, 4)) {
        Write-Host ""
        Write-Host "-- bench: chunk ${chunk}s / batch $b (30 秒持続) --" -ForegroundColor Cyan
        & $CliExe rt --virtual $BenchWav -o $out --chunk $chunk --batch $b @ExtraArgs
    }
    Write-Host ""
    Write-Host "判定: budget 行に対して hops over budget 0 かつ p99 < budget が合格。" -ForegroundColor Cyan
}

function Invoke-E2E {
    <# オフライン E2E: Rust CLI と Python rcwx run を同条件で実行し corr 比較。 #>
    if (-not (Test-Cargo)) { return }
    if (-not (Test-Uv)) { return }
    $model = Join-Path $PSScriptRoot 'model\kurumi\kurumi.pth'
    if (-not (Test-Path $model)) {
        Write-Host "[NG] モデルがありません: $model" -ForegroundColor Red
        return
    }
    if (-not (Test-Path $ParityWav)) {
        Write-Host "[NG] $ParityWav がありません。'.\dev.ps1 golden' を先に実行。" -ForegroundColor Red
        return
    }
    if (-not (Invoke-CargoBuild)) { return }
    $rustOut = Join-Path $GoldenDir 'e2e_rust.wav'
    $pyOut   = Join-Path $GoldenDir 'e2e_py.wav'
    Write-Host "-- Python 基準出力 --" -ForegroundColor Cyan
    & uv run --no-sync python -m rcwx.cli run $ParityWav $model -o $pyOut --pitch 0 --f0-method rmvpe --no-compile
    if ($LASTEXITCODE -ne 0) { Write-Host "[NG] Python 実行失敗。" -ForegroundColor Red; return }
    Write-Host "-- Rust 出力 --" -ForegroundColor Cyan
    & $CliExe run $ParityWav -o $rustOut
    if ($LASTEXITCODE -ne 0) { Write-Host "[NG] Rust 実行失敗。" -ForegroundColor Red; return }
    Write-Host "-- corr 比較 --" -ForegroundColor Cyan
    & uv run --no-sync python -c "import numpy as np; from scipy.io import wavfile; rd = lambda p: wavfile.read(p)[1].astype(np.float64).ravel(); a, b = rd(r'$pyOut'), rd(r'$rustOut'); n = min(len(a), len(b)); c = np.dot(a[:n], b[:n])/(np.linalg.norm(a[:n])*np.linalg.norm(b[:n])); print(f'E2E corr = {c:.5f}  (gate: > 0.95)'); import sys; sys.exit(0 if c > 0.95 else 1)"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] E2E パリティ合格。" -ForegroundColor Green
    } else {
        Write-Host "[NG] corr が 0.95 未満。" -ForegroundColor Red
    }
}

function Invoke-Gui {
    if (-not (Test-Cargo)) { return }
    if (-not (Invoke-CargoBuild)) { return }
    Write-Host "-- rcwx-gui 起動 --" -ForegroundColor Cyan
    & cargo run --release -p rcwx-gui --quiet
}

function Invoke-CliForward {
    <# cargo build 後に rcwx-cli へ引数を転送。 #>
    param([string[]]$CmdArgs)
    if (-not (Invoke-CargoBuild)) { return }
    & $CliExe @CmdArgs
}

# ---------------------------------------------------------------------------
# メニュー
# ---------------------------------------------------------------------------
function Show-Menu {
    Write-Host ""
    Write-Host "======= RCWX dev (Rust workspace) =======" -ForegroundColor Cyan
    Write-Host "  1) 環境チェック"
    Write-Host "  2) ビルド (release)"
    Write-Host "  3) テスト (パリティ一式)"
    Write-Host "  4) golden 再生成 (重み変換・uv)"
    Write-Host "  5) 持続ベンチ (30s, batch 1/4)"
    Write-Host "  6) E2E パリティ (Python と corr 比較)"
    Write-Host "  7) GUI 起動"
    Write-Host "  8) リアルタイム CLI (rt) — 追加引数を入力"
    Write-Host "  9) 任意の rcwx-cli コマンドを入力"
    Write-Host " 10) fmt / clippy"
    Write-Host " 11) clean"
    Write-Host "  0) 終了"
    Write-Host "========================================" -ForegroundColor Cyan
}

function Invoke-Lint {
    Write-Host "-- cargo fmt --" -ForegroundColor Cyan
    & cargo fmt --all
    Write-Host "-- cargo clippy (workspace) --" -ForegroundColor Cyan
    & cargo clippy --release --workspace 2>&1 | Select-String -Pattern '^(warning|error)' | Select-Object -First 20
    Write-Host "[i] 警告は既知の未使用変数等が残ります (rcwx-nn の整理は今後課題)。" -ForegroundColor DarkGray
}

# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------
# 外側の @() で必ず配列化 (単一引数時にスカラー化して [0] が文字になるのを防ぐ)
$DevArgs = @(@($args) | Where-Object { $_ -ne $null } | ForEach-Object { "$_" })
if ($DevArgs -and $DevArgs.Count -gt 0) {
    $sub = ("$($DevArgs[0])").ToLower()
    $rest = @()
    if ($DevArgs.Count -gt 1) { $rest = $DevArgs[1..($DevArgs.Count - 1)] }
    switch ($sub) {
        'check'  { Invoke-Check }
        'build'  { if (Test-Cargo) { Invoke-CargoBuild | Out-Null } }
        'test'   { Invoke-Test }
        'golden' { Invoke-Golden }
        'bench'  { Invoke-Bench $rest }
        'e2e'    { Invoke-E2E }
        'gui'    { Invoke-Gui }
        'fmt'    { if (Test-Cargo) { & cargo fmt --all } }
        'clippy' { Invoke-Lint }
        'clean'  { if (Test-Cargo) { & cargo clean } }
        'rt'     { Invoke-CliForward $DevArgs }   # rt ... は cli へそのまま (サブコマンド含む)
        default  { Invoke-CliForward $DevArgs }   # run / info / devices / logs / download ...
    }
    exit $LASTEXITCODE
}

if (-not (Test-Cargo)) { exit 1 }
Invoke-Check

$running = $true
while ($running) {
    Show-Menu
    $choice = Read-Host "選択"
    switch ($choice) {
        '1'  { Invoke-Check }
        '2'  { Invoke-CargoBuild | Out-Null }
        '3'  { Invoke-Test }
        '4'  { Invoke-Golden }
        '5'  { Invoke-Bench @() }
        '6'  { Invoke-E2E }
        '7'  { Invoke-Gui }
        '8'  {
            $a = Read-Host "rt への引数 (例: --seconds 10 --chunk 0.02 / 空で既定)"
            $parts = @()
            if ($a) { $parts = $a -split '\s+' | Where-Object { $_ -ne '' } }
            Invoke-CliForward (@('rt') + $parts)
        }
        '9'  {
            $c = Read-Host "rcwx-cli への引数 (例: run in.wav -o out.wav --pitch 5)"
            if ($c) {
                $parts = $c -split '\s+' | Where-Object { $_ -ne '' }
                Invoke-CliForward $parts
            }
        }
        '10' { Invoke-Lint }
        '11' { & cargo clean }
        '0'  { $running = $false }
        default { Write-Host "無効な選択です: $choice" -ForegroundColor Yellow }
    }
}
Write-Host "終了しました。"
