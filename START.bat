@echo off
setlocal EnableExtensions

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

set "VENV=%ROOT%\runtime\venv"
set "VENV_CFG=%VENV%\pyvenv.cfg"
set "PYTHON=%VENV%\Scripts\python.exe"
set "UV=%ROOT%\tools\uv.exe"
set "MODEL=%ROOT%\models\Qwen3.5-0.8B-Q8_0.exe"
set "LOGDIR=%ROOT%\logs"
set "MODEL_OUT=%LOGDIR%\llamafile.out.log"
set "MODEL_ERR=%LOGDIR%\llamafile.err.log"
set "MODEL_PORT=8080"
set "MODEL_URL=http://127.0.0.1:%MODEL_PORT%"

rem Keep the portable runtime isolated from any Python installed on the host PC.
set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONNOUSERSITE=1"

rem Keep uv's cache and downloads on the USB drive. exFAT has no hard links, so copy files.
set "UV_CACHE_DIR=%ROOT%\cache\uv"
set "UV_PYTHON_INSTALL_DIR=%ROOT%\runtime\python"
set "UV_LINK_MODE=copy"

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

rem Locate the bundled base interpreter that the venv runs on.
set "BASE_PY="
for /d %%D in ("%ROOT%\runtime\python\cpython-3.12*-windows-x86_64-none") do (
  if exist "%%~fD\python.exe" set "BASE_PY=%%~fD"
)
if not defined BASE_PY (
  echo Bundled base Python not found under:
  echo %ROOT%\runtime\python
  pause
  exit /b 1
)

rem The venv stores the base interpreter as an absolute path, including the drive letter.
rem Windows assigns USB drive letters per PC, so repoint it at this drive on every launch.
powershell -NoProfile -ExecutionPolicy Bypass -Command "$line = 'home = ' + $env:BASE_PY; $lines = [IO.File]::ReadAllLines($env:VENV_CFG); if ($lines -notcontains $line) { $new = [string[]]@($lines | ForEach-Object { if ($_ -match '^\s*home\s*=') { $line } else { $_ } }); [IO.File]::WriteAllLines($env:VENV_CFG, $new, (New-Object Text.UTF8Encoding $false)) }"

"%PYTHON%" -c "import sys" >nul 2>&1
if errorlevel 1 (
  echo Portable Python could not start:
  "%PYTHON%" -c "import sys"
  echo.
  echo Check %VENV_CFG%
  pause
  exit /b 1
)

"%PYTHON%" -c "import langchain, langchain_openai, langgraph, mcp, httpx, psutil, win32api" >nul 2>&1
if errorlevel 1 (
  echo Required Python packages are missing from the portable runtime.
  echo Installing them into the USB drive. This needs an internet connection...
  "%UV%" pip install -r "%ROOT%\agent\requirements.txt" --python "%PYTHON%"
  "%PYTHON%" -c "import langchain, langchain_openai, langgraph, mcp, httpx, psutil, win32api"
  if errorlevel 1 (
    echo.
    echo Package installation failed. Connect to the internet and run START.bat again.
    pause
    exit /b 1
  )
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
echo Port %MODEL_PORT% may be in use by another program, or antivirus may have blocked the model.
echo Check logs:
echo %MODEL_ERR%
echo %MODEL_OUT%
pause
exit /b 1

:ready
echo Model API is ready.
echo.
"%PYTHON%" "%ROOT%\agent\main.py"
set "AGENT_EXIT=%ERRORLEVEL%"

if "%AGENT_EXIT%"=="20" exit /b 0

echo.
echo Portable AI stopped. The model server may still be running.
echo Use STOP.bat to stop the USB model server.
if not "%PORTABLEAI_NO_PAUSE%"=="1" pause
