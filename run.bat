\
    @echo off
    setlocal ENABLEEXTENSIONS ENABLEDELAYEDEXECUTION

    echo ==== Starting BatchBench ====
    if not exist .venv (
        echo [ERROR] No .venv found. Run setup.bat first.
        pause
        exit /b 1
    )
    call .venv\Scripts\activate
    set FLASK_APP=app.py
    set FLASK_RUN_PORT=5000
    python -m flask run
    pause
