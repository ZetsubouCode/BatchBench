from typing import Tuple, List
from pathlib import Path
import re
from PIL import Image

from utils.io import readable_path, ensure_out_dir, log_join
from utils.merge_groups_core import merge_many


def _list_images(folder: Path, glob_pat: str, exts: List[str]) -> List[Path]:
    extset = {(e if e.startswith(".") else f".{e}").lower().strip() for e in exts if e.strip()}
    return sorted(
        [p for p in folder.glob(glob_pat) if p.is_file() and p.suffix.lower() in extset],
        key=lambda p: p.name.lower(),
    )


def _row_is_white(pix, y: int, w: int, threshold: int, ratio: float) -> bool:
    white = 0
    for x in range(w):
        if pix[x, y] >= threshold:
            white += 1
    return (white / w) >= ratio


def _fill_small_gaps(rows: List[bool], max_gap: int) -> List[bool]:
    if max_gap <= 0:
        return rows
    out = rows[:]
    i = 0
    n = len(rows)
    while i < n:
        if out[i]:
            i += 1
            continue
        start = i
        while i < n and not out[i]:
            i += 1
        end = i - 1
        if start > 0 and i < n and (end - start + 1) <= max_gap:
            for k in range(start, end + 1):
                out[k] = True
    return out


def _find_stripes(
    img: Image.Image, threshold: int, min_height: int, ratio: float, max_gap: int
) -> List[Tuple[int, int]]:
    gray = img.convert("L")
    pix = gray.load()
    w, h = gray.size
    rows = [_row_is_white(pix, y, w, threshold, ratio) for y in range(h)]
    rows = _fill_small_gaps(rows, max_gap)

    stripes: List[Tuple[int, int]] = []
    y = 0
    while y < h:
        if not rows[y]:
            y += 1
            continue
        start = y
        while y < h and rows[y]:
            y += 1
        end = y - 1
        if (end - start + 1) >= min_height:
            stripes.append((start, end))
    return stripes


def _segments_from_stripes(img_height: int, stripes: List[Tuple[int, int]], min_panel: int):
    segments: List[Tuple[int, int]] = []
    last = 0
    for start, end in stripes:
        if start - last >= min_panel:
            segments.append((last, start))
        last = end + 1
    if img_height - last >= min_panel:
        segments.append((last, img_height))
    return segments


def _chapter_dirs(folder: Path) -> List[Path]:
    pat = re.compile(r"^(?:chapter\s*)?\d+", re.IGNORECASE)
    subs = [d for d in folder.iterdir() if d.is_dir()]
    numbered = [d for d in subs if pat.match(d.name)]
    return sorted(numbered or subs, key=lambda p: p.name.lower())


def _targets(folder: Path, glob_pat: str, exts: List[str]) -> List[Tuple[Path, List[Path]]]:
    targets: List[Tuple[Path, List[Path]]] = []
    here = _list_images(folder, glob_pat, exts)
    if here:
        targets.append((folder, here))
    for sub in _chapter_dirs(folder):
        imgs = _list_images(sub, glob_pat, exts)
        if imgs:
            targets.append((sub, imgs))
    return targets


def handle(form, ctx) -> Tuple[str, str]:
    active_tab = "webtoon"
    folder = readable_path(form.get("wt_folder", ""))
    out_dir_raw = (form.get("wt_out_dir", "") or "").strip()
    out_dir = readable_path(out_dir_raw) if out_dir_raw else None
    glob_pat = form.get("wt_glob", "*.*").strip() or "*.*"

    exts_raw = (form.get("wt_exts", ".png,.jpg,.jpeg,.webp") or ".png,.jpg,.jpeg,.webp").strip()
    exts = [e.strip().lower() for e in exts_raw.split(",") if e.strip()] or [".png"]

    resize_mode = form.get("wt_resize", "match-width")
    white_threshold = int(form.get("wt_white_threshold", "245") or 245)
    row_ratio = float(form.get("wt_row_ratio", "98") or 98)
    min_stripe = int(form.get("wt_min_stripe", "12") or 12)
    max_gap = int(form.get("wt_max_gap", "2") or 2)
    min_panel = int(form.get("wt_min_panel", "128") or 128)

    save_strip = bool(form.get("wt_save_strip"))
    overwrite = bool(form.get("wt_overwrite"))
    dry_run = bool(form.get("wt_dry_run"))

    # Clamp inputs to sane ranges
    white_threshold = max(0, min(255, white_threshold))
    row_ratio = max(0.0, min(100.0, row_ratio)) / 100.0
    min_stripe = max(1, min_stripe)
    max_gap = max(0, max_gap)
    min_panel = max(1, min_panel)
    resize_mode = resize_mode if resize_mode in ("match-width", "none") else "match-width"

    lines: List[str] = []
    if not folder.exists() or not folder.is_dir():
        lines.append("Source folder not found.")
        return active_tab, log_join(lines)

    targets = _targets(folder, glob_pat, exts)
    if not targets:
        lines.append(f"No images found with pattern '{glob_pat}' and extensions {', '.join(exts)}.")
        return active_tab, log_join(lines)

    lines.append(f"Root: {folder} | Chapters detected: {len(targets)}")
    for i, (chap_path, pages) in enumerate(targets, start=1):
        lines.append(f"[{i}/{len(targets)}] {chap_path.name}: {len(pages)} page(s)")
        target_out = (out_dir / chap_path.name) if out_dir else (chap_path / "_panels")
        if not dry_run:
            ensure_out_dir(target_out)

        try:
            merged = merge_many(pages, "v", "center", 0, "#FFFFFF", resize_mode)
            lines.append(f"  Merged size: {merged.width}x{merged.height}px (gapless stack)")

            stripes = _find_stripes(merged, white_threshold, min_stripe, row_ratio, max_gap)
            segments = _segments_from_stripes(merged.height, stripes, min_panel)

            lines.append(
                f"  Stripe rule: >= {min_stripe}px tall, row >= {int(row_ratio*100)}% >= {white_threshold}"
            )
            lines.append(f"  Stripes found: {len(stripes)} | Planned slices: {len(segments)} (>= {min_panel}px)")
            for idx, (start, end) in enumerate(stripes, start=1):
                lines.append(f"    Stripe {idx}: y={start}..{end} (h={end-start+1})")

            if dry_run:
                merged.close()
                lines.append("  Dry run: no files written.")
                continue

            if save_strip:
                strip_path = target_out / f"{chap_path.name}_strip.png"
                if strip_path.exists() and not overwrite:
                    lines.append(f"  Skip strip (exists): {strip_path.name}")
                else:
                    merged.save(strip_path)
                    lines.append(f"  Saved strip -> {strip_path.name}")

            pad = max(3, len(str(len(segments)))) if segments else 3
            saved = 0
            for idx, (top, bottom) in enumerate(segments, start=1):
                if bottom <= top:
                    continue
                out_path = target_out / f"{chap_path.name}_{idx:0{pad}d}.png"
                if out_path.exists() and not overwrite:
                    lines.append(f"  Skip slice (exists): {out_path.name}")
                    continue
                try:
                    piece = merged.crop((0, top, merged.width, bottom))
                    piece.save(out_path)
                    piece.close()
                    saved += 1
                    lines.append(f"  Saved slice {idx} -> {out_path.name} (y={top}..{bottom-1})")
                except Exception as e:
                    lines.append(f"  [ERROR] slice {idx} ({out_path.name}): {e}")
            merged.close()
            lines.append(f"  Done: {saved} slice(s) into {target_out}")
        except Exception as e:
            lines.append(f"[ERROR] {chap_path.name}: {e}")

    if dry_run:
        lines.append("Dry run finished: no files were saved.")
    return active_tab, log_join(lines)
