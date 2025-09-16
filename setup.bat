\
    @echo off
    setlocal ENABLEEXTENSIONS ENABLEDELAYEDEXECUTION

    echo.
    echo ==== BatchBench Setup (Windows) ====
    where python >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] Python not found. Please install Python 3.11+ from python.org and re-run setup.bat
        pause
        exit /b 1
    )

    if not exist .venv (
        echo Creating virtual environment .venv ...
        python -m venv .venv
        if errorlevel 1 (
            echo [ERROR] Failed to create venv.
            pause
            exit /b 1
        )
    )

    echo Activating venv ...
    call .venv\Scripts\activate

    echo Upgrading pip ...
    python -m pip install --upgrade pip

    echo Installing requirements ...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] pip install failed.
        pause
        exit /b 1
    )

    if not exist .env (
        echo Creating .env from .env.example ...
        copy /Y .env.example .env >nul
    )

    echo.
    echo Setup complete!
    echo Next: double-click run.bat
    pause
