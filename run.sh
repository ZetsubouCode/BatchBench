#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==== Starting BatchBench ===="
VENV_PY=".venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  echo "[ERROR] No local venv found. Run ./setup.sh first."
  exit 1
fi

echo "Checking Python dependencies ..."
if ! "$VENV_PY" -c "import importlib.util, sys; mods=['flask','werkzeug','dotenv','PIL','numpy','torch','transformers','huggingface_hub','safetensors','timm','hf_xet']; missing=[m for m in mods if importlib.util.find_spec(m) is None]; print('Missing: ' + ', '.join(missing)) if missing else None; sys.exit(1 if missing else 0)"; then
  echo "[WARN] Missing dependencies detected. Installing from requirements.txt ..."
  "$VENV_PY" -m pip install -r requirements.txt
fi

export FLASK_APP="app.py"
export FLASK_RUN_PORT="${FLASK_RUN_PORT:-5000}"
APP_URL="http://127.0.0.1:${FLASK_RUN_PORT}"
echo "Open in browser: ${APP_URL}"

if command -v xdg-open >/dev/null 2>&1; then
  (sleep 2; xdg-open "${APP_URL}" >/dev/null 2>&1 || true) &
elif command -v open >/dev/null 2>&1; then
  (sleep 2; open "${APP_URL}" >/dev/null 2>&1 || true) &
fi

"$VENV_PY" -m flask run --host 127.0.0.1 --port "${FLASK_RUN_PORT}"
