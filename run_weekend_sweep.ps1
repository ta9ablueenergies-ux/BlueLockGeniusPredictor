$env:SWEEP_MODE="hybrid"
# Load FDO_TOKEN from .env if not already set
if (-not $env:FDO_TOKEN) {
    $envFile = Join-Path $PSScriptRoot ".env"
    if (Test-Path $envFile) {
        Get-Content $envFile | Where-Object { $_ -match "^FDO_TOKEN=" } | ForEach-Object {
            $env:FDO_TOKEN = ($_ -split "=", 2)[1].Trim()
        }
    }
}
$env:ENABLE_BROWSER_SCRAPING="1"
$env:ENABLE_TRAINING_DATA_GUARD="0"
$env:ENABLE_MARKET_COUNT_MODELS="0"
$env:ENABLE_FREE_SOURCE_BACKFILL="0"
$env:ENABLE_FLASHSCORE_STATS_BACKFILL="0"

$dates = @("2026-05-08", "2026-05-09", "2026-05-10", "2026-05-11")
$root = "C:\Users\U033IAT\Documents\antigravity world\football_predictions"

foreach ($d in $dates) {
    $env:PIPELINE_START_DATE = $d
    Write-Host "=== Sweeping $d ===" -ForegroundColor Cyan
    python "$root\scripts\platform_orchestrator.py" 2>&1
    Write-Host "=== Done $d ===" -ForegroundColor Green
}

Write-Host "Weekend sweep complete." -ForegroundColor Yellow
