@echo off
REM Daily football predictions sweep launcher.
REM Run by Windows Task Scheduler — do NOT run manually unless testing.

setlocal

set PROJECT_ROOT=%~dp0
set PYTHON=C:\Users\U033IAT\AppData\Local\Programs\Python\Python311\python.exe
set SCRIPT=%PROJECT_ROOT%scripts\platform_orchestrator.py
set LOG_DIR=%PROJECT_ROOT%web\data\logs
set LOG_FILE=%LOG_DIR%\sweep_%DATE:~10,4%%DATE:~4,2%%DATE:~7,2%.log

REM Create log directory if missing
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM Sweep configuration
set SWEEP_MODE=hybrid
set ENABLE_BROWSER_SCRAPING=1
set ENABLE_FLASHSCORE_STATS_BACKFILL=1
set ENABLE_MARKET_COUNT_MODELS=1
set ENABLE_TRAINING_DATA_GUARD=1
set ENABLE_FREE_SOURCE_BACKFILL=0
set FLASHSCORE_STATS_LIMIT=150
set BROWSER_AUTO_PREFER=seleniumbase

echo [%DATE% %TIME%] Starting daily sweep >> "%LOG_FILE%"
"%PYTHON%" "%SCRIPT%" >> "%LOG_FILE%" 2>&1
echo [%DATE% %TIME%] Sweep complete (exit=%ERRORLEVEL%) >> "%LOG_FILE%"

endlocal
