# rename_sorter.py
from typing import Tuple, List
from pathlib import Path
import re, shutil

from utils.io import readable_path, log_join  # already used by your other tools

VALID_EXTS = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
SUBFOLDER_RE = re.compile(r'^(\d+)(?:_(byname|bydate))?$', re.IGNORECASE)

def _natural_key(s: str):
    # e.g. "img2" < "img10"
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def _parse_subfolder(name: str):
    """
    Accepts '12_byname', '7_bydate', or '25'.
    Returns (number:int, sort_by:str['name'|'date']) or None if invalid.
    Default (no suffix) -> 'date'
    """
    m = SUBFOLDER_RE.match(name)
    if not m:
        return None
    num = int(m.group(1))
    suffix = (m.group(2) or '').lower()
    sort_by = 'name' if suffix == 'byname' else 'date'
    return num, sort_by

def _iter_sorted_images(folder: Path, sort_by: str) -> List[Path]:
    items: List[Path] = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]
    if sort_by == 'date':
        items.sort(key=lambda p: p.stat().st_mtime)  # oldest -> newest
    else:
        items.sort(key=lambda p: _natural_key(p.name))
    return items

def _unique_dest(dst_dir: Path, base: str, ext: str) -> Path:
    """Ensure we don't overwrite: base.ext, base-1.ext, base-2.ext, ..."""
    candidate = dst_dir / f"{base}{ext}"
    if not candidate.exists():
        return candidate
    i = 1
    while True:
        alt = dst_dir / f"{base}-{i}{ext}"
        if not alt.exists():
            return alt
        i += 1

def handle(form, ctx) -> Tuple[str, str]:
    active_tab = "rename"

    root = readable_path(form.get("rs_root", ""))
    out_raw = (form.get("rs_out", "") or "").strip()
    move_mode = bool(form.get("rs_move"))
    overwrite = bool(form.get("rs_overwrite"))
    verbose = True  # keep logs visible like other tools

    lines: List[str] = []

    if not root.exists() or not root.is_dir():
        lines.append("Root folder not found or not a directory.")
        return active_tab, log_join(lines)

    out_dir = readable_path(out_raw) if out_raw else (root / "sorted_images")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect valid subfolders
    subs = []
    for p in sorted(root.iterdir(), key=lambda x: x.name):
        if not p.is_dir():
            continue
        parsed = _parse_subfolder(p.name)
        if parsed is None:
            continue
        subs.append((p, parsed[0], parsed[1]))
    subs.sort(key=lambda t: t[1])  # sort by numeric prefix

    if not subs:
        lines.append("No valid subfolders found (expected '<number>[_byname|_bydate]').")
        return active_tab, log_join(lines)

    total = 0
    for sub_path, sub_num, sort_by in subs:
        images = _iter_sorted_images(sub_path, sort_by)
        if not images:
            if verbose: lines.append(f"[{sub_path.name}] No images.")
            continue

        if verbose:
            lines.append(f"[{sub_path.name}] mode={sort_by}, files={len(images)}")

        for idx, img in enumerate(images, start=1):
            base = f"{sub_num}_{idx:03d}"
            ext = img.suffix.lower()
            dst = out_dir / f"{base}{ext}"

            if dst.exists() and not overwrite:
                dst = _unique_dest(out_dir, base, ext)

            try:
                if move_mode:
                    shutil.move(str(img), str(dst))
                    action = "Moved"
                else:
                    shutil.copy2(str(img), str(dst))
                    action = "Copied"
                if verbose: lines.append(f"{action}: {img} -> {dst}")
                total += 1
            except Exception as e:
                lines.append(f"ERROR on {img.name}: {e}")

    lines.append(f"Done. Wrote {total} file(s) to: {out_dir}")
    return active_tab, log_join(lines)
