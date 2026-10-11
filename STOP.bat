@echo off
setlocal

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

echo Stopping PortableAI model processes from:
echo %ROOT%

powershell -NoProfile -ExecutionPolicy Bypass -Command "$root = '%ROOT%'; Get-Process | Where-Object { $_.Path -like ($root + '\models\Qwen3.5-0.8B-Q8_0.exe') } | Stop-Process -Force"

echo Done.
pause
