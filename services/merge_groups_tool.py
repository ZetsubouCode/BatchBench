from typing import Tuple, List
from pathlib import Path
from utils.io import readable_path, log_join
from utils.merge_groups_core import group_by_prefix, merge_many  # note: no list_pngs import

def _list_images(folder: Path, glob_pat: str, exts: List[str]) -> List[Path]:
    # Normalize extensions like ["png","jpg",".webp"] -> {".png",".jpg",".webp"}
    extset = { (e if e.startswith(".") else f".{e}").lower().strip() for e in exts if e.strip() }
    return sorted(
        [p for p in folder.glob(glob_pat) if p.is_file() and p.suffix.lower() in extset],
        key=lambda p: p.name.lower()
    )

def handle(form, ctx) -> Tuple[str, str]:
    active_tab = "merge"
    folder = readable_path(form.get("merge_folder",""))
    out_dir_raw = (form.get("merge_out_dir","") or "").strip()
    out_dir = readable_path(out_dir_raw) if out_dir_raw else None
    glob_pat = form.get("merge_glob","*_*.*").strip() or "*_*.*"

    # NEW: read extensions from the form (default .png)
    exts_raw = (form.get("merge_exts", ".png") or ".png").strip()
    exts = [e.strip().lower() for e in exts_raw.split(",") if e.strip()] or [".png"]

    skip_single = bool(form.get("merge_skip_single"))
    reverse = bool(form.get("merge_reverse"))
    orientation = form.get("merge_orientation","v")
    resize = form.get("merge_resize","auto")
    align = (form.get("merge_align","center") or "center").lower()
    gap = int(form.get("merge_gap","0") or 0)
    bg = (form.get("merge_bg","#FFFFFF") or "#FFFFFF").strip()
    overwrite = bool(form.get("merge_overwrite"))
    dry_run = bool(form.get("merge_dry_run"))
    verbose = True

    lines: List[str] = []
    if not folder.exists() or not folder.is_dir():
        lines.append("Source folder not found.")
        return active_tab, log_join(lines)

    # CHANGED: use _list_images instead of list_pngs
    files = _list_images(folder, glob_pat, exts)
    groups = group_by_prefix(files)

    if not groups:
        lines.append(f"No matching files like <prefix>_<number> with extensions {', '.join(exts)} found.")
        return active_tab, log_join(lines)

    if verbose:
        lines.append(f"Source: {folder}")
        lines.append(f"Groups found: {len(groups)} | Orientation: {orientation} | Resize: {resize} | Gap: {gap}px")

    out_dir = out_dir or (folder / "combined")
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    planned = []
    for prefix, pairs in sorted(groups.items(), key=lambda kv: kv[0].lower()):
        ordered = [p for _, p in pairs]
        if reverse:
            ordered = list(reversed(ordered))
        if skip_single and len(ordered) == 1:
            if verbose: lines.append(f"Skip single: {prefix}")
            continue
        out_path = out_dir / f"{prefix}.png"  # keep merged outputs as PNG
        planned.append((prefix, ordered, out_path))

    if verbose:
        lines.append(f"Groups to merge: {len(planned)} (writing to {out_dir})")
        for prefix, ordered, out_path in planned:
            names = ", ".join(p.name for p in ordered)
            lines.append(f"  {prefix} -> {out_path.name}  [{len(ordered)} parts]  :: {names}")

    if dry_run:
        return active_tab, log_join(lines)

    made = 0
    for idx, (prefix, ordered, out_path) in enumerate(planned, start=1):
        if out_path.exists() and not overwrite:
            if verbose: lines.append(f"[{idx}/{len(planned)}] Skip (exists): {out_path.name}")
            continue
        try:
            merged = merge_many(ordered, orientation, align, gap, bg, resize)
            merged.save(out_path)
            merged.close()
            made += 1
            if verbose: lines.append(f"[{idx}/{len(planned)}] Saved -> {out_path}")
        except Exception as e:
            lines.append(f"[{idx}/{len(planned)}] ERROR merging {prefix}: {e}")

    lines.append(f"Done. Wrote {made} file(s) to: {out_dir}")
    return active_tab, log_join(lines)
