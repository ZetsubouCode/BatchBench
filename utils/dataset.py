from pathlib import Path
from typing import List, Tuple

def split_tags(text: str): 
    return [t.strip() for t in text.split(",") if t.strip()]

def join_tags(tags: List[str]): 
    return ", ".join(tags)

def collect_pairs(folder: Path, image_exts: List[str]) -> List[Tuple[Path, Path]]:
    files = list(folder.iterdir())
    stems_txt = {p.stem for p in files if p.suffix.lower() == ".txt"}
    out = []
    for p in files:
        if p.is_file() and p.suffix.lower() in image_exts and p.stem in stems_txt:
            out.append((p, p.with_suffix(".txt")))
    out.sort(key=lambda t: t[0].stem.lower())
    return out
