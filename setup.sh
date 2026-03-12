#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo
echo "==== BatchBench Setup (Linux/macOS) ===="

PYTHON_CMD=""
if command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD="python"
fi

if [[ -z "$PYTHON_CMD" ]]; then
  echo "[ERROR] Python 3.11+ not found. Install Python, then re-run setup.sh."
  exit 1
fi

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Creating virtual environment .venv ..."
  "$PYTHON_CMD" -m venv .venv
fi

VENV_PY=".venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  echo "[ERROR] Venv Python not found: $VENV_PY"
  exit 1
fi

echo "Upgrading pip ..."
"$VENV_PY" -m pip install --upgrade pip

echo "Installing requirements ..."
"$VENV_PY" -m pip install -r requirements.txt

if [[ ! -f ".env" ]]; then
  echo "Creating .env from .env.example ..."
  cp .env.example .env
fi

echo
echo "Setup complete."
echo "Next: run ./run.sh"
