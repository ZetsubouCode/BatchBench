from pathlib import Path
from typing import List

def readable_path(p: str) -> Path:
    return Path(p).expanduser()

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
