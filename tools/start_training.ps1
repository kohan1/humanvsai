<#
.SYNOPSIS
    Start a training run on this PC, detached, and return immediately.

.DESCRIPTION
    Intended to be invoked over SSH from the Mac:

        ssh <user>@<training-pc> "powershell -File <repo-path>\tools\start_training.ps1 -Game snake"

    The job is launched detached, so it keeps running after the SSH session
    closes. A plain `python train.py` over SSH does NOT survive logout — Windows
    tears down the process tree with the session, which is the whole reason this
    script exists.

    Training happens here, not on the Mac, because this box has the RTX 4060 Ti.
    The Mac has no CUDA; a batch-2048 update measured ~19x faster on this GPU.
    Use the Mac to drive and to watch, not to train.

.PARAMETER Game
    snake | watermelon | tetris

.PARAMETER Pipeline
    Run the full pretrain -> evaluate -> train -> evaluate chain instead of just
    train.py. Only meaningful for snake, which has run_v2_pipeline.sh.

.PARAMETER Status
    Do not start anything; just report whether a run is active and show the
    tail of its log.

.EXAMPLE
    .\start_training.ps1 -Game snake
    .\start_training.ps1 -Game watermelon -Pipeline
    .\start_training.ps1 -Status
#>
param(
    [ValidateSet("snake", "watermelon", "tetris")]
    [string]$Game = "snake",

    [switch]$Pipeline,
    [switch]$Status
)

$ErrorActionPreference = "Stop"
# This script lives in tools/, so the project root is one level up.
$root = Split-Path $PSScriptRoot -Parent
$trainingDir = Join-Path $root "$Game\training"
$logPath = Join-Path $trainingDir "remote_training.log"

function Get-TrainingProcesses {
    Get-CimInstance Win32_Process -Filter "name like '%python%'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and ($_.CommandLine -match "train\.py|pretrain\.py|run_v2_pipeline") }
}

if ($Status) {
    $procs = @(Get-TrainingProcesses)
    if ($procs.Count -gt 0) {
        Write-Output "RUNNING - $($procs.Count) python process(es)"
        $procs | Select-Object -First 3 | ForEach-Object {
            "  pid $($_.ProcessId)  cpu $([math]::Round($_.UserModeTime/6e8,1)) min"
        }
    } else {
        Write-Output "IDLE - no training process found"
    }
    if (Test-Path $logPath) {
        Write-Output "--- last 15 log lines ($logPath) ---"
        Get-Content $logPath -Tail 15
    } else {
        Write-Output "(no log at $logPath yet)"
    }
    exit 0
}

if (-not (Test-Path $trainingDir)) {
    Write-Error "No training folder at $trainingDir"
    exit 1
}

$existing = @(Get-TrainingProcesses)
if ($existing.Count -gt 0) {
    Write-Error "Training already running ($($existing.Count) process(es)). Use -Status, or stop it first."
    exit 1
}

# PYTHONIOENCODING: torch.onnx and SB3 print glyphs the default cp1252 console
# cannot encode, which crashes the run part-way through.
# -u: stdout is redirected to a file here, and buffered output means no visible
# progress until the process exits.
$env:PYTHONIOENCODING = "utf-8"

if ($Pipeline) {
    $pipelineScript = Join-Path $trainingDir "run_v2_pipeline.sh"
    if (-not (Test-Path $pipelineScript)) {
        Write-Error "No run_v2_pipeline.sh in $trainingDir"
        exit 1
    }
    $exe = "bash"
    $argList = @("run_v2_pipeline.sh")
} else {
    $exe = "python"
    $argList = @("-u", "train.py")
}

Write-Output "Starting: $exe $($argList -join ' ')"
Write-Output "Working dir: $trainingDir"
Write-Output "Log: $logPath"

$proc = Start-Process -FilePath $exe `
    -ArgumentList $argList `
    -WorkingDirectory $trainingDir `
    -RedirectStandardOutput $logPath `
    -RedirectStandardError (Join-Path $trainingDir "remote_training.err.log") `
    -WindowStyle Hidden `
    -PassThru

Start-Sleep -Seconds 3
if ($proc.HasExited) {
    Write-Output "FAILED - process exited immediately (code $($proc.ExitCode))"
    if (Test-Path $logPath) { Get-Content $logPath -Tail 20 }
    exit 1
}

Write-Output "Started, pid $($proc.Id). Safe to close the SSH session."
Write-Output "Check with: .\start_training.ps1 -Status"
