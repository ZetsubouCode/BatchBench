import os
from pathlib import Path
from typing import Any, List, Optional

def readable_path(p: str) -> Path:
    raw = str(p or "").strip().strip('"').strip("'")
    if not raw:
        return Path("")
    expanded = os.path.expandvars(os.path.expanduser(raw))
    return Path(expanded)


def readable_path_or_none(p: Any) -> Optional[Path]:
    raw = str(p or "").strip().strip('"').strip("'")
    if not raw:
        return None
    expanded = os.path.expandvars(os.path.expanduser(raw))
    return Path(expanded)

def ensure_out_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def log_join(lines: List[str]) -> str:
    return "\n".join(lines)

def windows_drives():
    drives = []
    for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        root = f"{c}:\\"
        if Path(root).exists():
            drives.append(root)
    return drives


def default_browse_root() -> Path:
    if os.name == "nt":
        drives = windows_drives()
        if drives:
            return Path(drives[0])
    return Path("/")
