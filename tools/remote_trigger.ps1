<#
.SYNOPSIS
    Watch the GitHub repo for training requests and act on them.

.DESCRIPTION
    Lets you start training on this PC from anywhere — including school
    networks that block Tailscale, VPNs and SSH.

    Why this works when a VPN does not: nothing ever connects *in* to this PC.
    This script polls github.com outbound over HTTPS, which is the same thing a
    browser does. From school you only need to load a normal web page.

    HOW YOU USE IT
      1. This script runs in the background on the PC (see -Install).
      2. From anywhere — laptop, phone, library computer — open an issue on
         github.com/kohan1/humanvsai with a title starting "train:", e.g.
             train: snake
             train: watermelon
             status
      3. Within a minute the PC picks it up, starts training, comments with
         the result, and closes the issue.

    Because it replies in the issue thread, you also get a permanent log of
    every run and its outcome.

    SAFETY: only issues opened by the repo owner are acted on. Anything else
    is ignored and closed with a note. A private repo already limits this, but
    an unauthenticated trigger is not something to leave lying around.

.PARAMETER PollSeconds
    How often to check GitHub. Default 60. Do not go below ~20; the GitHub API
    is rate limited and there is no benefit.

.PARAMETER Once
    Check a single time and exit. Useful for testing, or for running from
    Task Scheduler on a timer instead of as a long-lived process.

.PARAMETER Install
    Register this script as a scheduled task that starts at logon, so the
    watcher is always running without a console window.

.EXAMPLE
    .\remote_trigger.ps1 -Once          # test it
    .\remote_trigger.ps1                # run the watcher now
    .\remote_trigger.ps1 -Install       # run automatically at logon
#>
param(
    [int]$PollSeconds = 60,
    [switch]$Once,
    [switch]$Install
)

$ErrorActionPreference = "Stop"
# This script lives in tools/, so the project root is one level up.
$root = Split-Path $PSScriptRoot -Parent
$repo = "kohan1/humanvsai"
$gh = Join-Path $env:ProgramFiles "GitHub CLI\gh.exe"
$logPath = Join-Path $PSScriptRoot "remote_trigger.log"

function Write-Log($msg) {
    $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Write-Output $line
    Add-Content -Path $logPath -Value $line -Encoding utf8
}

if ($Install) {
    $action = New-ScheduledTaskAction -Execute "powershell.exe" `
        -Argument "-NoProfile -WindowStyle Hidden -File `"$PSCommandPath`"" `
        -WorkingDirectory $root
    $trigger = New-ScheduledTaskTrigger -AtLogOn
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit ([TimeSpan]::Zero)
    try {
        Register-ScheduledTask -TaskName "humanvsai-remote-trigger" -Action $action `
            -Trigger $trigger -Settings $settings -Force -ErrorAction Stop | Out-Null
    } catch {
        Write-Output "FAILED to register the scheduled task: $($_.Exception.Message)"
        Write-Output "Registering a task needs an ADMIN PowerShell. Right-click"
        Write-Output "PowerShell -> Run as administrator, then run this again."
        exit 1
    }
    # Confirm it actually exists rather than trusting the call not to have
    # failed quietly - this previously reported success after being denied.
    if (-not (Get-ScheduledTask -TaskName "humanvsai-remote-trigger" -ErrorAction SilentlyContinue)) {
        Write-Output "FAILED: task did not register. Run this from an admin PowerShell."
        exit 1
    }
    Write-Output "Installed scheduled task 'humanvsai-remote-trigger' (starts at logon)."
    Write-Output "Start it now with:  Start-ScheduledTask -TaskName humanvsai-remote-trigger"
    exit 0
}

if (-not (Test-Path $gh)) {
    Write-Log "gh.exe not found at $gh - install GitHub CLI first"
    exit 1
}

# Repo owner, resolved once. Only this account may trigger a run.
$owner = (& $gh api "repos/$repo" | ConvertFrom-Json).owner.login
Write-Log "watching $repo (owner: $owner), polling every ${PollSeconds}s"

function Get-TrainingProcesses {
    Get-CimInstance Win32_Process -Filter "name like '%python%'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and ($_.CommandLine -match "train\.py|pretrain\.py|run_v2_pipeline") }
}

function Reply-AndClose($number, $body) {
    & $gh issue comment $number --repo $repo --body $body | Out-Null
    & $gh issue close $number --repo $repo | Out-Null
}

function Handle-Issue($issue) {
    $num = $issue.number
    $title = $issue.title.Trim()
    $author = $issue.author.login

    if ($author -ne $owner) {
        Write-Log "issue #$num from '$author' ignored (not the owner)"
        Reply-AndClose $num "Ignored: only @$owner can trigger training."
        return
    }

    Write-Log "issue #${num}: '$title'"

    # "status" — report what is happening, start nothing.
    $fence = '```'
    if ($title -match '^\s*status\s*$') {
        $procs = @(Get-TrainingProcesses)
        if ($procs.Count -gt 0) {
            $cpu = [math]::Round(($procs | Measure-Object -Property UserModeTime -Sum).Sum / 6e8, 1)
            $body = "**Training is RUNNING** - $($procs.Count) process(es), $cpu CPU-minutes so far.`n`n"
        } else {
            $body = "**Idle** - nothing is training.`n`n"
        }
        foreach ($g in @("snake", "watermelon", "tetris")) {
            $log = Join-Path $root "$g\training\v2_pipeline.log"
            if (-not (Test-Path $log)) { $log = Join-Path $root "$g\training\remote_training.log" }
            if (Test-Path $log) {
                $tail = (Get-Content $log -Tail 12) -join "`n"
                $body += "<details><summary>$g log</summary>`n`n$fence`n$tail`n$fence`n</details>`n`n"
            }
        }
        Reply-AndClose $num $body
        return
    }

    # "train: <game>" — start a run.
    if ($title -match '^\s*train:\s*(\w+)') {
        $game = $matches[1].ToLower()
        if ($game -notin @("snake", "watermelon", "tetris")) {
            Reply-AndClose $num "Unknown game '$game'. Use snake, watermelon or tetris."
            return
        }

        $running = @(Get-TrainingProcesses)
        if ($running.Count -gt 0) {
            Reply-AndClose $num "Already training ($($running.Count) process(es)). Open a **status** issue to check on it, or stop it on the PC first."
            return
        }

        $starter = Join-Path $PSScriptRoot "start_training.ps1"   # sibling in tools/
        $usePipeline = Test-Path (Join-Path $root "$game\training\run_v2_pipeline.sh")
        $psArgs = @("-NoProfile", "-File", $starter, "-Game", $game)
        if ($usePipeline) { $psArgs += "-Pipeline" }

        Write-Log "starting $game (pipeline: $usePipeline)"
        $out = & powershell.exe @psArgs | Out-String

        Reply-AndClose $num "Started **$game** training on the PC.`n`n$fence`n$($out.Trim())`n$fence`n`nOpen an issue titled **status** to check progress."
        return
    }

    Write-Log "issue #${num}: title not recognised, ignoring"
    Reply-AndClose $num "Not recognised. Use a title like ``train: snake`` or ``status``."
}

function Poll-Once {
    try {
        $issues = & $gh issue list --repo $repo --state open --json number,title,author --limit 20 | ConvertFrom-Json
    } catch {
        Write-Log "GitHub unreachable: $($_.Exception.Message)"
        return
    }
    foreach ($issue in @($issues)) { Handle-Issue $issue }
}

if ($Once) {
    Poll-Once
    exit 0
}

while ($true) {
    Poll-Once
    Start-Sleep -Seconds $PollSeconds
}
