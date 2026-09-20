@echo off
setlocal
cd /d "%~dp0"
python -u "%~dp0setup_repositories.py" %*
set "setupExit=%errorlevel%"
echo.
if not "%setupExit%"=="0" echo Source setup has errors. See the messages above.
pause
exit /b %setupExit%
