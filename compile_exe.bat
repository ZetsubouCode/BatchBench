@echo off
setlocal ENABLEEXTENSIONS ENABLEDELAYEDEXPANSION
cd /d "%~dp0"

echo.
echo ==== BatchBench EXE Builder ====

set "PYTHON_CMD="
where py >nul 2>nul
if not errorlevel 1 set "PYTHON_CMD=py -3"
if not defined PYTHON_CMD (
    where python >nul 2>nul
    if not errorlevel 1 set "PYTHON_CMD=python"
)
if not defined PYTHON_CMD (
    echo [ERROR] Python launcher not found. Install Python 3.11+ and run this file again.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment .venv ...
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create .venv.
        pause
        exit /b 1
    )
)

set "VENV_PY=%CD%\.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo [ERROR] Venv Python not found: %VENV_PY%
    pause
    exit /b 1
)

echo Upgrading pip ...
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip.
    pause
    exit /b 1
)

echo Installing build requirements ...
"%VENV_PY%" -m pip install -r requirements-dev.txt
if errorlevel 1 (
    echo [ERROR] Failed to install build requirements.
    pause
    exit /b 1
)

if not exist ".env" (
    if exist ".env.example" (
        echo Creating .env from .env.example ...
        copy /Y ".env.example" ".env" >nul
    )
)

echo Building Windows executable ...
set "BATCHBENCH_BUILD_PYTHON=%VENV_PY%"
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\build_windows.ps1"
if errorlevel 1 (
    echo.
    echo [ERROR] EXE build failed.
    pause
    exit /b 1
)

echo.
echo Build complete.
echo EXE: %CD%\dist\BatchBench\BatchBench.exe
if exist "%CD%\dist\BatchBench" start "" "%CD%\dist\BatchBench"
pause
