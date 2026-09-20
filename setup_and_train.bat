@echo off
setlocal
cd /d "%~dp0"

python -u setup_repositories.py
if errorlevel 1 (
    echo Source setup failed. Training was not started.
    pause
    exit /b 1
)

python -u main.py --mode train --resume 05 --only 02 03 04 05 06 07 08 09 --continue-on-error
set "runExit=%errorlevel%"
pause
exit /b %runExit%
