# services/group_renamer.py
from typing import Tuple, List, Iterable
from pathlib import Path
import shutil
import re
from utils.io import readable_path, log_join

def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r'\d+|\D+', s)]

def _list_images(folder: Path, exts: set) -> List[Path]:
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]

def _order(items: Iterable[Path], how: str) -> List[Path]:
    how = (how or "name").lower()
    if how == "mtime":
        return sorted(items, key=lambda p: (p.stat().st_mtime, p.name.lower()))
    if how == "ctime":
        return sorted(items, key=lambda p: (p.stat().st_ctime, p.name.lower()))
    # default: natural name
    return sorted(items, key=lambda p: _natural_key(p.name))

def _ensure_unique(target: Path) -> Path:
    if not target.exists():
        return target
    stem, ext = target.stem, target.suffix
    bump = 1
    while True:
        candidate = target.with_name(f"{stem}-{bump}{ext}")
        if not candidate.exists():
            return candidate
        bump += 1

def _copy_or_move(src: Path, dst: Path, move: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))

def handle(form, ctx) -> Tuple[str, str]:
    active_tab = "rename"

    root = readable_path(form.get("rn_root",""))
    out_dir_in = (form.get("rn_out","") or "").strip()
    out_dir = readable_path(out_dir_in) if out_dir_in else (root / "_renamed")

    exts = set(e.strip().lower() for e in (form.get("rn_exts",".png,.jpg,.jpeg,.webp") or "").split(",") if e.strip())
    include_txt = bool(form.get("rn_include_txt"))
    move_instead = bool(form.get("rn_move_instead"))
    dry_run = bool(form.get("rn_dry_run"))

    pad = max(1, int(form.get("rn_pad","3")))
    suffix_pad = max(0, int(form.get("rn_suffix_pad","0")))
    sep = (form.get("rn_sep","_") or "_")
    start = max(1, int(form.get("rn_start","1")))

    order_top = form.get("rn_top_order","name")
    order_folder = form.get("rn_folder_order","name")
    order_inside = form.get("rn_inside_order","name")

    lines: List[str] = []

    if not root.exists() or not root.is_dir():
        lines.append("Root folder not found.")
        return active_tab, log_join(lines)

    # Collect top-level files and immediate subfolders
    top_images = _order(_list_images(root, exts), order_top)
    subfolders = _order([p for p in root.iterdir() if p.is_dir()], order_folder)

    seq = start
    planned = []

    def pref(n: int) -> str:
        return f"{n:0{pad}d}"

    def part(n: int) -> str:
        return f"{n:0{suffix_pad}d}" if suffix_pad > 0 else str(n)

    # Plan top-level images (plain sequence: 001, 002, ...)
    for img in top_images:
        dst_img = out_dir / f"{pref(seq)}{img.suffix.lower()}"
        if include_txt:
            txt_src = img.with_suffix(".txt")
            txt_dst = out_dir / f"{pref(seq)}.txt" if txt_src.exists() else None
        else:
            txt_src = txt_dst = None
        planned.append(("top", img, dst_img, txt_src, txt_dst))
        seq += 1

    # Plan subfolder images (group as: 003_1, 003_2, … then 004_1, 004_2, …)
    for folder in subfolders:
        imgs = _order(_list_images(folder, exts), order_inside)
        if not imgs:
            continue
        prefix = pref(seq)
        k = 1
        for img in imgs:
            stem = f"{prefix}{sep}{part(k)}"
            dst_img = out_dir / f"{stem}{img.suffix.lower()}"
            if include_txt:
                txt_src = img.with_suffix(".txt")
                txt_dst = out_dir / f"{stem}.txt" if txt_src.exists() else None
            else:
                txt_src = txt_dst = None
            planned.append(("group", img, dst_img, txt_src, txt_dst))
            k += 1
        seq += 1

    # Log the plan
    lines.append(f"Top-level images: {len(top_images)} · Subfolders: {len(subfolders)}")
    lines.append(f"Output: {out_dir}")
    lines.append(f"Numbering: pad={pad}, suffix_pad={suffix_pad}, start={start}, sep='{sep}'")
    lines.append(f"Order: top={order_top}, folder={order_folder}, inside={order_inside}")
    lines.append(f"Mode: {'MOVE' if move_instead else 'COPY'} · Include .txt: {include_txt}")
    lines.append(f"Planned renames: {len(planned)}")
    for kind, src, dst, txt_src, txt_dst in planned[:20]:  # preview first 20
        lines.append(f"  {kind:5s}  {src.name} -> {dst.name}" + (f"  (+txt)" if txt_dst else ""))

    if dry_run:
        lines.append("Dry run: no files written.")
        return active_tab, log_join(lines)

    # Execute
    done = 0
    for _, src, dst, txt_src, txt_dst in planned:
        dst_u = _ensure_unique(dst)
        try:
            _copy_or_move(src, dst_u, move_instead)
            done += 1
        except Exception as e:
            lines.append(f"[ERROR] {src.name}: {e}")
            continue
        if txt_src and txt_dst:
            txt_dst_u = _ensure_unique(txt_dst)
            try:
                _copy_or_move(txt_src, txt_dst_u, move_instead)
            except Exception as e:
                lines.append(f"[WARN] {txt_src.name}: {e}")

    lines.append(f"Done. {done} image(s) processed into: {out_dir}")
    return active_tab, log_join(lines)
