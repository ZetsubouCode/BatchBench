from __future__ import annotations

import os
import sys
import threading
import time
import webbrowser
from pathlib import Path


_DLL_DIRECTORY_HANDLES = []


def _exe_log_path() -> Path:
    return Path(sys.executable).resolve().parent / "_work" / "logs" / "exe_runtime.log"


def _write_exe_log(message: str) -> None:
    try:
        path = _exe_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(message.rstrip() + "\n")
    except Exception:
        pass


def _bootstrap_frozen_torch_dlls() -> None:
    root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    candidates = [
        root / "torch" / "lib",
        root / "_internal" / "torch" / "lib",
        Path(sys.executable).resolve().parent / "_internal" / "torch" / "lib",
    ]
    added = False
    for candidate in candidates:
        if not candidate.exists():
            continue
        text = str(candidate)
        current = os.environ.get("PATH", "")
        if text not in current.split(os.pathsep):
            os.environ["PATH"] = text + (os.pathsep + current if current else "")
        add_dll_directory = getattr(os, "add_dll_directory", None)
        if add_dll_directory is not None:
            try:
                _DLL_DIRECTORY_HANDLES.append(add_dll_directory(text))
                _write_exe_log(f"torch DLL directory added: {text}")
            except Exception:
                _write_exe_log(f"torch DLL directory failed: {text}")
        added = True
    if not added:
        _write_exe_log("EXE bootstrap: no bundled torch DLL directory; offline tagger uses external worker Python")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    _write_exe_log(f"EXE bootstrap root: {root}")
    _write_exe_log(f"EXE bootstrap _MEIPASS: {getattr(sys, '_MEIPASS', '')}")


_bootstrap_frozen_torch_dlls()

from PIL import Image

from app import batchbench_local_url, run_batchbench_server
from services.paths import resource_path


def _diagnose_torch_import() -> int:
    _write_exe_log("diagnose_torch: start external worker check")
    import subprocess

    candidates = []
    override = os.environ.get("BATCHBENCH_OFFLINE_TAGGER_PYTHON")
    if override:
        candidates.append(Path(override))
    exe_dir = Path(sys.executable).resolve().parent
    for root in (Path.cwd(), exe_dir.parent.parent, exe_dir.parent):
        candidates.append(root / ".venv" / "Scripts" / "python.exe")
        candidates.append(root / "venv" / "Scripts" / "python.exe")
    seen = set()
    for python_exe in candidates:
        try:
            resolved = python_exe.resolve()
        except Exception:
            resolved = python_exe.absolute()
        key = str(resolved).lower()
        if key in seen or not resolved.exists():
            continue
        seen.add(key)
        _write_exe_log(f"diagnose_torch: trying worker python {resolved}")
        proc = subprocess.run(
            [
                str(resolved),
                "-c",
                "from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification; import torch; print(torch.__version__)",
            ],
            text=True,
            capture_output=True,
        )
        _write_exe_log(f"diagnose_torch: exit={proc.returncode} stdout={proc.stdout.strip()} stderr={proc.stderr.strip()}")
        if proc.returncode == 0:
            return 0
    _write_exe_log("diagnose_torch: no working external Python found")
    return 1


def _load_tray_icon_image():
    icon_path = resource_path("static", "icons", "icon.ico")
    try:
        with Image.open(icon_path) as image:
            return image.convert("RGBA").copy()
    except Exception:
        return Image.new("RGBA", (64, 64), (59, 130, 246, 255))


def _run_without_tray() -> None:
    run_batchbench_server(open_browser=True)


def _run_with_tray() -> None:
    try:
        import pystray
    except Exception:
        _run_without_tray()
        return

    stop_event = threading.Event()
    server_error = []

    def server_main() -> None:
        try:
            run_batchbench_server(open_browser=True, stop_event=stop_event)
        except Exception as exc:
            server_error.append(exc)
            stop_event.set()

    server_thread = threading.Thread(target=server_main, name="BatchBenchServer", daemon=False)
    server_thread.start()

    def open_batchbench(_icon=None, _item=None) -> None:
        try:
            webbrowser.open(batchbench_local_url())
        except Exception:
            pass

    def exit_batchbench(icon, _item=None) -> None:
        stop_event.set()
        try:
            icon.stop()
        except Exception:
            pass

    icon = pystray.Icon(
        "BatchBench",
        _load_tray_icon_image(),
        "BatchBench",
        menu=pystray.Menu(
            pystray.MenuItem("Open BatchBench", open_batchbench, default=True),
            pystray.MenuItem("Exit BatchBench", exit_batchbench),
        ),
    )

    def monitor_server() -> None:
        while server_thread.is_alive() and not stop_event.is_set():
            time.sleep(0.25)
        try:
            icon.stop()
        except Exception:
            pass

    threading.Thread(target=monitor_server, name="BatchBenchTrayMonitor", daemon=True).start()
    icon.run()
    stop_event.set()
    server_thread.join(timeout=5.0)
    if server_thread.is_alive():
        os._exit(0)
    if server_error:
        raise server_error[0]


if __name__ == "__main__":
    if "--diagnose-torch" in sys.argv:
        raise SystemExit(_diagnose_torch_import())
    _run_with_tray()
