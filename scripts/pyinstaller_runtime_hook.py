from __future__ import annotations

import os
import sys
from pathlib import Path


_DLL_DIRECTORY_HANDLES = []


def _log(message: str) -> None:
    try:
        path = Path(sys.executable).resolve().parent / "_work" / "logs" / "exe_runtime.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(message.rstrip() + "\n")
    except Exception:
        pass


def _add_dll_dir(path: Path) -> bool:
    if not path.exists():
        return False
    text = str(path)
    current = os.environ.get("PATH", "")
    parts = current.split(os.pathsep) if current else []
    if text not in parts:
        os.environ["PATH"] = text + (os.pathsep + current if current else "")
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is not None:
        try:
            _DLL_DIRECTORY_HANDLES.append(add_dll_directory(text))
            _log(f"runtime hook DLL directory added: {text}")
        except Exception:
            _log(f"runtime hook DLL directory failed: {text}")
    return True


def _bootstrap_torch_dll_search() -> None:
    root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    candidates = [
        root / "torch" / "lib",
        root / "_internal" / "torch" / "lib",
        Path(sys.executable).resolve().parent / "_internal" / "torch" / "lib",
    ]
    added = False
    for candidate in candidates:
        added = _add_dll_dir(candidate) or added
    if not added:
        _log("runtime hook: no bundled torch DLL directory; offline tagger uses external worker Python")

    # PyTorch CPU wheels may load Intel OpenMP from torch\lib. In a frozen app,
    # another loaded extension can initialize OpenMP first; this avoids a hard
    # DLL initialization failure when the duplicate runtime is harmless.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    _log(f"runtime hook root: {root}")
    _log(f"runtime hook _MEIPASS: {getattr(sys, '_MEIPASS', '')}")


_bootstrap_torch_dll_search()
