@echo off
setlocal

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

set "PYTHON=%ROOT%\runtime\venv\Scripts\python.exe"
set "MODEL=%ROOT%\models\Qwen3.5-0.8B-Q8_0.exe"
set "LOGDIR=%ROOT%\logs"
set "MODEL_LOG=%LOGDIR%\llamafile.log"
set "MODEL_OUT=%LOGDIR%\llamafile.out.log"
set "MODEL_ERR=%LOGDIR%\llamafile.err.log"
set "MODEL_PORT=8080"
set "MODEL_URL=http://127.0.0.1:%MODEL_PORT%"

if not exist "%LOGDIR%" mkdir "%LOGDIR%"

if not exist "%PYTHON%" (
  echo Portable Python venv not found:
  echo %PYTHON%
  pause
  exit /b 1
)

if not exist "%MODEL%" (
  echo Model executable not found:
  echo %MODEL%
  pause
  exit /b 1
)

echo Checking local model server...
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $r = Invoke-RestMethod -Uri '%MODEL_URL%/health' -TimeoutSec 2; if ($r.status -eq 'ok') { exit 0 } } catch { }; exit 1"
if errorlevel 1 (
  echo Starting llamafile model server...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "$model = '%MODEL%'; $root = '%ROOT%'; $out = '%MODEL_OUT%'; $err = '%MODEL_ERR%'; Start-Process -FilePath $model -ArgumentList @('--server','--host','127.0.0.1','--port','%MODEL_PORT%','--nologo','--no-webui') -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $out -RedirectStandardError $err"
) else (
  echo Model server is already running.
)

echo Waiting for model API...
for /L %%I in (1,1,90) do (
  powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $r = Invoke-RestMethod -Uri '%MODEL_URL%/health' -TimeoutSec 2; if ($r.status -eq 'ok') { exit 0 } } catch { }; exit 1"
  if not errorlevel 1 goto ready
  powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Sleep -Seconds 1"
)

echo Model server did not become ready in time.
echo Check log:
echo %MODEL_LOG%
pause
exit /b 1

:ready
echo Model API is ready.
echo.
"%PYTHON%" "%ROOT%\agent\main.py"

echo.
echo Portable AI stopped. The model server may still be running.
echo Use STOP.bat to stop the USB model server.
if not "%PORTABLEAI_NO_PAUSE%"=="1" pause
