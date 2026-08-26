# Paper-style eval suite (~8h on RTX 3060): base + lora/all, 12 ep × 12 env
#   .\eval\run_eval_suite.ps1
#
# Requires: pip install -e .

param(
    [int]$Episodes = 12,
    [string]$CheckpointDir = "lora",
    [string]$Variants = "base,all",
    [switch]$Paper              # 3B paper model (lora_3b/)
)

$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
Set-Location $Root

$VenvPy = Join-Path $Root "venv\Scripts\python.exe"
if (-not (Test-Path $VenvPy)) {
    $VenvPy = "python"
}

$EstHours = [math]::Round(3.75 * ($Variants -split ",").Count, 1)
Write-Host "Output: eval/runs/<timestamp>/result.md + tables/metrics/logs" -ForegroundColor Cyan
Write-Host "Root: $Root"
Write-Host "Checkpoint: $CheckpointDir | Variants: $Variants | Episodes: $Episodes"
Write-Host "Estimated wall time: ~${EstHours}h (RTX 3060, 0.5B 4-bit)" -ForegroundColor Yellow

$Args = @(
    "eval/run_eval_suite.py",
    "--checkpoint-dir", $CheckpointDir,
    "--episodes", "$Episodes",
    "--variants", $Variants
)

if ($Paper) {
    $Args += "--paper"
    if ($CheckpointDir -eq "lora") {
        $Args[2] = "lora_3b"
        Write-Host "Using lora_3b for --paper"
    }
}

Write-Host "`nCommand: $VenvPy $($Args -join ' ')`n"
& $VenvPy @Args
exit $LASTEXITCODE
