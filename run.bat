@echo off
setlocal ENABLEEXTENSIONS ENABLEDELAYEDEXECUTION
cd /d "%~dp0"

echo ==== Starting BatchBench ====
set "VENV_PY=.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo [ERROR] No local venv found. Run setup.bat first.
    pause
    exit /b 1
)

echo Checking Python dependencies ...
"%VENV_PY%" -c "import importlib.util, sys; mods=['flask','werkzeug','dotenv','PIL','numpy','torch','transformers','huggingface_hub','safetensors','timm','hf_xet']; missing=[m for m in mods if importlib.util.find_spec(m) is None]; print('Missing: ' + ', '.join(missing)) if missing else None; sys.exit(1 if missing else 0)"
if errorlevel 1 (
    echo [WARN] Missing dependencies detected. Installing from requirements.txt ...
    "%VENV_PY%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] pip install failed.
        pause
        exit /b 1
    )
)

set FLASK_APP=app.py
set FLASK_RUN_PORT=5000
echo Open in browser: http://127.0.0.1:%FLASK_RUN_PORT%
"%VENV_PY%" -m flask run --host 127.0.0.1 --port %FLASK_RUN_PORT%
pause
