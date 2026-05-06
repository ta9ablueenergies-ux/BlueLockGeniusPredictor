# Install (or update) the daily football predictions sweep as a Windows Scheduled Task.
# Run once as Administrator:  .\install_daily_sweep.ps1
# To remove:                  Unregister-ScheduledTask -TaskName "FootballPredictionsSweep" -Confirm:$false

$TaskName    = "FootballPredictionsSweep"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BatFile     = Join-Path $ProjectRoot "daily_sweep.bat"
$LogDir      = Join-Path $ProjectRoot "web\data\logs"

# Ensure log dir exists
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

# Remove stale task if it exists
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed existing task."
}

# Action: run the batch file
$Action = New-ScheduledTaskAction `
    -Execute "cmd.exe" `
    -Argument "/c `"$BatFile`"" `
    -WorkingDirectory $ProjectRoot

# Trigger: daily at 06:30 AM (before markets open, after overnight results)
$Trigger = New-ScheduledTaskTrigger -Daily -At "06:30AM"

# Principal: run as current user, only when logged in
$Principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Limited

# Settings: allow running even on battery, 30-min timeout
$Settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30) `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable `
    -MultipleInstances IgnoreNew `
    -DontStopIfGoingOnBatteries

Register-ScheduledTask `
    -TaskName   $TaskName `
    -Action     $Action `
    -Trigger    $Trigger `
    -Principal  $Principal `
    -Settings   $Settings `
    -Description "Daily football match prediction sweep (platform_orchestrator.py)" `
    -Force

Write-Host ""
Write-Host "Task '$TaskName' registered successfully."
Write-Host "  Runs: daily at 06:30 AM"
Write-Host "  Script: $BatFile"
Write-Host "  Logs: $LogDir"
Write-Host ""
Write-Host "To run now for testing:"
Write-Host "  Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "To check last run status:"
Write-Host "  Get-ScheduledTaskInfo -TaskName '$TaskName' | Select LastRunTime, LastTaskResult"
