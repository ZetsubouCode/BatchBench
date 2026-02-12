@echo off
setlocal ENABLEEXTENSIONS ENABLEDELAYEDEXECUTION
cd /d "%~dp0"

echo.
echo ==== BatchBench Setup (Windows) ====

set "PYTHON_CMD="
where py >nul 2>nul
if not errorlevel 1 set "PYTHON_CMD=py -3"
if not defined PYTHON_CMD (
    where python >nul 2>nul
    if not errorlevel 1 set "PYTHON_CMD=python"
)
if not defined PYTHON_CMD (
    echo [ERROR] Python launcher not found. Install Python 3.11+ and re-run setup.bat.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment .venv ...
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create venv.
        pause
        exit /b 1
    )
)

set "VENV_PY=.venv\Scripts\python.exe"
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

echo Installing requirements ...
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] pip install failed.
    pause
    exit /b 1
)

if not exist ".env" (
    echo Creating .env from .env.example ...
    copy /Y .env.example .env >nul
)

echo.
echo Setup complete.
echo Next: run run.bat
pause
