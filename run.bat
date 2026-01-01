    @echo off
    setlocal ENABLEEXTENSIONS ENABLEDELAYEDEXECUTION

    echo ==== Starting BatchBench ====
    if not exist .venv (
        echo [ERROR] No .venv found. Run setup.bat first.
        pause
        exit /b 1
    )
call .venv\Scripts\activate
echo Checking Python dependencies ...
python -c "import importlib.util, sys; mods=['flask','werkzeug','dotenv','PIL','numpy','torch','transformers','huggingface_hub','safetensors','timm','hf_xet']; missing=[m for m in mods if importlib.util.find_spec(m) is None]; print('Missing: ' + ', '.join(missing)) if missing else None; sys.exit(1 if missing else 0)"
if errorlevel 1 (
    echo [WARN] Missing dependencies detected. Installing from requirements.txt ...
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] pip install failed.
        pause
        exit /b 1
    )
)
set FLASK_APP=app.py
set FLASK_RUN_PORT=5000
set APP_URL=http://127.0.0.1:%FLASK_RUN_PORT%
start "" /b powershell -NoProfile -Command "Start-Sleep -Seconds 2; Start-Process '%APP_URL%'"
python -m flask run
pause
